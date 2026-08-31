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
from .canonical import canonicalise_record, verify_equivalence
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
                 lengths, stats):
        self.records = records
        self.train, self.val, self.test = train, val, test
        self.graph_vocab, self.amp_vocab, self.target_vocab = vocabs
        self.datasets = datasets
        self.loaders = loaders
        self.lengths = lengths
        self.stats = stats


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
            record["amp_tokens"] = list(record["amp_std"])

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
    graph_vocab = Vocab((r["graph_tokens"] for r in train), reserve_digits=False)
    amp_vocab = Vocab((r["amp_tokens"] for r in train))
    target_vocab = Vocab((r["target_tokens"] for r in train))

    oov = {}
    for name, part in (("val", val), ("test", test)):
        for field, vocab in (("graph_tokens", graph_vocab),
                             ("amp_tokens", amp_vocab),
                             ("target_tokens", target_vocab)):
            rate, unseen = vocab.oov_rate(r[field] for r in part)
            oov[f"{name}.{field}"] = {"rate": rate,
                                      "unseen": dict(unseen.most_common(10))}

    datasets, loaders = {}, {}
    generator = torch.Generator().manual_seed(cfg.train.seed)
    for name, part in (("train", train), ("val", val), ("test", test)):
        datasets[name] = AmplitudeDataset(part, graph_vocab, amp_vocab,
                                          target_vocab)
        loaders[name] = make_loader(datasets[name], cfg.train.batch_size,
                                    shuffle=(name == "train"),
                                    generator=generator)

    lengths = tuple(max(datasets[n].lengths()[i] for n in datasets)
                    for i in range(3))

    stats = {
        "theory": cfg.data.theory,
        "n_records": len(records),
        "n_templates": len(classes),
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "vocab": {"graph": len(graph_vocab), "amp": len(amp_vocab),
                  "target": len(target_vocab)},
        "max_lengths": {"graph": lengths[0], "amp": lengths[1],
                        "target": lengths[2]},
        "oov": oov,
        "segment_amp": segment_amp,
        "build_seconds": round(time.time() - t0, 1),
    }

    if verbose:
        print(f"[{cfg.data.theory}] {stats['n_records']} records, "
              f"{stats['n_templates']} templates | "
              f"train/val/test {len(train)}/{len(val)}/{len(test)} | "
              f"vocab g={len(graph_vocab)} a={len(amp_vocab)} "
              f"t={len(target_vocab)} | max len {lengths} | "
              f"{stats['build_seconds']}s")
        worst = max(oov.values(), key=lambda d: d["rate"])
        if worst["rate"] > 0:
            print(f"  OOV up to {worst['rate']:.3%}: {worst['unseen']}")

    bundle = Bundle(records, train, val, test,
                    (graph_vocab, amp_vocab, target_vocab),
                    datasets, loaders, lengths, stats)
    bundle.segment_amp = segment_amp
    return bundle


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
