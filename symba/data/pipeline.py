"""The single route from raw line to tensor (01 P1).

There is exactly one path here. If a stage fails, the build fails; nothing
falls back to a placeholder record, and nothing downstream reads a field that
an earlier stage did not write.
"""

import random
import time

import numpy as np
import torch

from ..config import Config
from .ast_parse import parse_amp_record
from .canonical import canonicalise_record
from .dataset import AmplitudeDataset, make_loader
from .graph import build_graph_record
from .load import load_theory
from .normalize import normalize_record
from .serialize import target_tokens
from .splits import (assert_disjoint, assign_template_classes, split_records)
from .vocab import Vocab


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Bundle:
    """Everything one experiment needs: splits, vocabularies, loaders."""

    def __init__(self, records, train, val, test, vocabs, datasets, loaders,
                 lengths, stats, generator=None, decode_budget=None):
        self.records = records
        self.train, self.val, self.test = train, val, test
        self.graph_vocab, self.amp_vocab, self.target_vocab = vocabs
        self.datasets = datasets
        self.loaders = loaders
        self.lengths = lengths
        self.stats = stats
        # The shuffle stream, exposed so a run can reset it. A bundle is shared
        # between arms, so by the second arm the generator has already been
        # advanced by the first one's epochs.
        self.generator = generator
        # What the decoder is allowed to emit. Separate from ``lengths``: see
        # ``seed_run`` and the note on sizing below.
        self.decode_budget = decode_budget


def seed_run(cfg, bundle=None):
    """Make one (arm, seed) run a function of its config and seed alone.

    ``build`` seeds once, but a job trains several arms against a cached
    bundle, so by the time the second arm constructs its model the global RNG
    has been advanced by every model, dropout draw and shuffle before it. The
    same (arm, seed) therefore produced different initial weights depending on
    where it sat in the job - measured, a max absolute difference of 0.53 in
    the decoder embedding.

    Two consequences, both bad. Numbers were not reproducible from
    ``config + seed``, which is exactly what ``config.py`` claims. And arms
    could not be compared, because they differed in initialisation as well as
    in the factor under test; resetting here gives common random numbers
    across arms, which is free variance reduction on a corpus with none to
    spare.

    Call immediately before constructing the model.
    """
    set_seed(cfg.train.seed)
    if bundle is not None and bundle.generator is not None:
        bundle.generator.manual_seed(cfg.train.seed)


def build(cfg: Config, verify_canonical: bool = False, verbose: bool = True):
    """Run S1-S6 and return a :class:`Bundle`."""
    t0 = time.time()
    set_seed(cfg.train.seed)

    records = load_theory(cfg.data.root, cfg.data.theory)

    for record in records:
        normalize_record(record)
        build_graph_record(record)
        parse_amp_record(record, segment=True)
        canonicalise_record(record, verify=verify_canonical)

        if cfg.data.target == "canonical":
            record["target_tokens"] = target_tokens(record["canon_num"],
                                                    record["canon_den"])
        else:
            # Raw mode exists only so the canonical-vs-raw comparison in
            # 01 SS6.1 can be run; it re-serialises the raw string through the
            # same grammar so the two arms differ in one thing only.
            from .ast_parse import amp_to_prefix
            record["target_tokens"] = amp_to_prefix(record["sq_amp_std"])

        if cfg.data.amp_representation == "raw":
            # The physics-free control of 03 SS3.4: characters instead of the
            # AST. Both views must change together, or segmentation would feed
            # the encoder the parsed diagrams anyway.
            record["amp_tokens"] = list(record["amp_std"])
            record["amp_segments"] = [record["amp_tokens"]]

    segment_amp = resolve_segmentation(cfg.data.segment_amp, records)
    if not segment_amp:
        for record in records:
            record["amp_segments"] = [record["amp_tokens"]]

    classes = assign_template_classes(records)

    train, val, test = split_records(
        records, cfg.data.split_protocol, cfg.data.val_frac,
        cfg.data.test_frac, cfg.train.seed)
    assert_disjoint(train, val, test)

    # G8: vocabularies see the training split and nothing else.
    #
    # The amplitude vocabulary is built over ``amp_segments``, which is the
    # stream the math encoder is actually fed. Building it over ``amp_tokens``
    # missed the ``<diagrams>`` placeholder that amp_to_segments substitutes
    # for the diagram sum in the context segment, so every QCD record fed the
    # encoder an ``<unk>`` there - and G7 never saw it, because G7 measured
    # ``amp_tokens`` rather than what the model reads. With segmentation off
    # amp_segments is ``[amp_tokens]``, so one path covers both settings.
    graph_vocab = Vocab((r["graph_tokens"] for r in train), reserve_digits=False)
    amp_vocab = Vocab((s for r in train for s in r["amp_segments"]))
    target_vocab = Vocab((r["target_tokens"] for r in train))

    def _streams(part, field):
        if field == "amp_segments":
            return (s for r in part for s in r[field])
        return (r[field] for r in part)

    oov = {}
    for name, part in (("val", val), ("test", test)):
        for field, vocab in (("graph_tokens", graph_vocab),
                             ("amp_segments", amp_vocab),
                             ("target_tokens", target_vocab)):
            rate, unseen = vocab.oov_rate(_streams(part, field))
            oov[f"{name}.{field}"] = {"rate": rate,
                                      "unseen": dict(unseen.most_common(10))}

    datasets, loaders = {}, {}
    generator = torch.Generator().manual_seed(cfg.train.seed)

    # Two different quantities, previously conflated into one.
    #
    # ``decode_budget`` is what the decoder is ALLOWED TO EMIT, and it comes
    # from the training split alone. Taking it over all three splits let
    # held-out target lengths cap the decoder - test-set information reaching
    # the thing under test.
    #
    # ``lengths`` sizes the positional TABLES, and is the observed maximum over
    # whatever is present. An embedding row that no training gradient ever
    # touches carries no information about a held-out target's content, so this
    # is not the leak; and sizing it from the training split alone would make a
    # long held-out sequence index off the end of the table. A held-out target
    # longer than ``decode_budget`` is a finding, not a crash: the model
    # structurally cannot emit it, so it scores as a miss on its own, and the
    # count is reported below rather than raising. Under leave-one-template-out
    # a held-out class longer than anything in training would otherwise kill
    # the fold.
    train_ds = AmplitudeDataset(train, graph_vocab, amp_vocab, target_vocab)
    decode_budget = _budget(train_ds.lengths()[2])

    datasets["train"] = train_ds
    for name, part in (("val", val), ("test", test)):
        datasets[name] = AmplitudeDataset(part, graph_vocab, amp_vocab,
                                          target_vocab)

    lengths = tuple(max(datasets[n].lengths()[i] for n in datasets) + 8
                    for i in range(3))
    segment_len = max(datasets[n].segment_length() for n in datasets) + 8

    unreachable = {
        name: sum(1 for item in datasets[name].items
                  if item["target"].size(0) > decode_budget)
        for name in ("val", "test")
    }

    for name in ("train", "val", "test"):
        loaders[name] = make_loader(datasets[name], cfg.train.batch_size,
                                    shuffle=(name == "train"),
                                    generator=generator)

    stats = {
        "theory": cfg.data.theory,
        "n_records": len(records),
        "n_templates": len(classes),
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "vocab": {"graph": len(graph_vocab), "amp": len(amp_vocab),
                  "target": len(target_vocab)},
        # Positional-table sizes: observed maxima plus a fixed cushion.
        "table_sizes": {"graph": lengths[0], "amp": lengths[1],
                        "target": lengths[2]},
        # What the decoder may emit. Training split only.
        "decode_budget": decode_budget,
        "train_max_lengths": dict(zip(("graph", "amp", "target"),
                                      train_ds.lengths())),
        # Held-out targets longer than the decode budget. The model cannot
        # emit these, so they score as misses; reported, never raised.
        "unreachable_targets": unreachable,
        "oov": oov,
        "segment_amp": segment_amp,
        "segment_len": segment_len,
        "build_seconds": round(time.time() - t0, 1),
    }

    if verbose:
        print(f"[{cfg.data.theory}] {stats['n_records']} records, "
              f"{stats['n_templates']} templates | "
              f"train/val/test {len(train)}/{len(val)}/{len(test)} | "
              f"vocab g={len(graph_vocab)} a={len(amp_vocab)} "
              f"t={len(target_vocab)} | tables {lengths} | "
              f"decode budget {decode_budget} | {stats['build_seconds']}s")
        worst = max(oov.values(), key=lambda d: d["rate"])
        if worst["rate"] > 0:
            print(f"  OOV up to {worst['rate']:.3%}: {worst['unseen']}")
        if any(unreachable.values()):
            print(f"  targets longer than the decode budget (scored as "
                  f"misses): {unreachable}")

    bundle = Bundle(records, train, val, test,
                    (graph_vocab, amp_vocab, target_vocab),
                    datasets, loaders, lengths, stats,
                    generator=generator, decode_budget=decode_budget)
    bundle.segment_amp = segment_amp
    bundle.segment_len = segment_len
    return bundle


# Cushion over the longest training target, matching the +8 the positional
# tables carry. Deliberately small: this IS the decode budget, and early in
# training the model does not emit EOS, so every junk
# sequence runs to the full budget and is then handed to sympy - whose cost
# grows superlinearly with nesting depth. A generous cushion buys nothing (the
# training maximum equals the corpus maximum on this corpus) and makes the
# first evaluations several times slower. A held-out sequence past the cushion
# raises in AmplitudeDataset rather than being clipped (01 P2).
BUDGET_CUSHION = 8


def _budget(train_max: int) -> int:
    return train_max + BUDGET_CUSHION


def resolve_segmentation(setting, records) -> bool:
    """Decide whether per-diagram encoding is worth it on this corpus.

    Attention cost goes as length squared, and segments are batched as a
    rectangle, so the comparison is ``n_diagrams * longest_diagram^2`` against
    ``full_length^2``. On QCD the first is ~50x smaller; on QED it is slightly
    larger, which matches the measured epoch times.
    """
    if isinstance(setting, bool):
        return setting
    if setting != "auto":
        raise ValueError(f"segment_amp must be True, False or 'auto', "
                         f"got {setting!r}")

    flat = segmented = 0
    for record in records:
        flat += len(record["amp_tokens"]) ** 2
        segments = record["amp_segments"]
        segmented += len(segments) * max(len(s) for s in segments) ** 2
    return segmented < flat
