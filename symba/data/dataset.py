"""Raw corpus -> preprocessed records -> train / val / test DataLoaders."""

import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ..config import PAD
from .ast_parse import amp_to_prefix, amp_to_segments
from .canonical import canonicalise
from .graph import FeynmanGraph
from .load import load_theory
from .normalize import standardize, strip_keywords
from .serialize import target_tokens
from .vocab import Vocab


def preprocess(record):
    """Add the token streams the model reads to a raw record.

    graph_tokens    the Feynman diagram: legs, vertices, propagators
    amp_tokens      the amplitude as a prefix-notation syntax tree
    amp_segments    the same amplitude, one sequence per Feynman diagram
    target_tokens   the canonical squared amplitude in prefix notation
    """
    graph = FeynmanGraph(strip_keywords(record["interaction"]),
                         strip_keywords(record["vertices"]))
    amp = standardize(record["amp"])
    record["graph_tokens"] = graph.to_tokens()
    record["amp_tokens"] = amp_to_prefix(amp)
    record["amp_segments"] = amp_to_segments(amp)
    if "sq_amp" in record:
        record["target_tokens"] = target_tokens(*canonicalise(standardize(record["sq_amp"])))
    return record


def worth_segmenting(records):
    """Per-diagram encoding pays when it lowers the attention cost: the sum of
    squared lengths, with each record's diagrams padded to its longest one."""
    flat = sum(len(r["amp_tokens"]) ** 2 for r in records)
    per_diagram = sum(len(r["amp_segments"]) * max(map(len, r["amp_segments"])) ** 2
                      for r in records)
    return per_diagram < flat


def split(records, val_frac, test_frac, seed):
    """Random ``(train, val, test)`` split of the records."""
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    n_test, n_val = round(test_frac * len(order)), round(val_frac * len(order))
    pick = lambda ids: [records[i] for i in ids]
    return pick(order[n_test + n_val:]), pick(order[n_test:n_test + n_val]), pick(order[:n_test])


def encode(record, graph_vocab, amp_vocab, target_vocab=None):
    item = {"graph": torch.tensor(graph_vocab.encode(record["graph_tokens"])),
            "segments": [torch.tensor(amp_vocab.encode(s)) for s in record["amp_segments"]]}
    if target_vocab is not None:
        item["target"] = torch.tensor(target_vocab.encode(record["target_tokens"]))
    return item


def _pad(tensors):
    ids = torch.full((len(tensors), max(t.size(0) for t in tensors)), PAD, dtype=torch.long)
    for i, t in enumerate(tensors):
        ids[i, :t.size(0)] = t
    return ids


def collate(items):
    """Pad items into a batch. Segments become ``(batch, diagrams, length)``."""
    n_seg = max(len(it["segments"]) for it in items)
    seg_len = max(s.size(0) for it in items for s in it["segments"])
    segments = torch.full((len(items), n_seg, seg_len), PAD, dtype=torch.long)
    for i, it in enumerate(items):
        for j, s in enumerate(it["segments"]):
            segments[i, j, :s.size(0)] = s

    batch = {"graph": _pad([it["graph"] for it in items]), "segments": segments}
    batch["graph_mask"] = batch["graph"].eq(PAD)
    batch["segments_mask"] = segments.eq(PAD)
    if "target" in items[0]:
        batch["target"] = _pad([it["target"] for it in items])
    if "index" in items[0]:
        batch["index"] = torch.tensor([it["index"] for it in items])
    return batch


class BucketSampler(torch.utils.data.Sampler):
    """Batches of similar-sized records, so padding stays small.

    For training the records are shuffled, cut into pools of 8 batches and
    sorted by size inside each pool, so batches still change every epoch.
    """

    def __init__(self, sizes, batch_size, shuffle, generator=None):
        self.sizes, self.batch_size = sizes, batch_size
        self.shuffle, self.generator = shuffle, generator

    def __iter__(self):
        n = len(self.sizes)
        if self.shuffle:
            order, pool = torch.randperm(n, generator=self.generator).tolist(), 8 * self.batch_size
        else:
            order, pool = list(range(n)), n
        order = [i for s in range(0, n, pool)
                 for i in sorted(order[s:s + pool], key=self.sizes.__getitem__)]
        batches = [order[i:i + self.batch_size] for i in range(0, n, self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=self.generator)]
        return iter(batches)

    def __len__(self):
        return -(-len(self.sizes) // self.batch_size)


@dataclass
class Data:
    train: list
    val: list
    test: list
    graph_vocab: Vocab
    amp_vocab: Vocab
    target_vocab: Vocab
    loaders: dict
    table_sizes: tuple      # positional-table rows: graph, amplitude segment, target
    decode_budget: int      # most tokens the decoder may emit
    segment_amp: bool
    stats: dict


def build(cfg, verbose=True):
    records = [preprocess(r) for r in load_theory(cfg.data.root, cfg.data.theory)]
    segment_amp = cfg.data.segment_amp
    if segment_amp is None:
        segment_amp = worth_segmenting(records)
    if not segment_amp:
        for r in records:
            r["amp_segments"] = [r["amp_tokens"]]

    train, val, test = split(records, cfg.data.val_frac, cfg.data.test_frac, cfg.train.seed)

    # Vocabularies see the training split only.
    graph_vocab = Vocab((r["graph_tokens"] for r in train), reserve_digits=False)
    amp_vocab = Vocab(s for r in train for s in r["amp_segments"])
    target_vocab = Vocab(r["target_tokens"] for r in train)

    generator = torch.Generator().manual_seed(cfg.train.seed)
    loaders, items = {}, {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        items[name] = [dict(encode(r, graph_vocab, amp_vocab, target_vocab), index=i)
                       for i, r in enumerate(part)]
        sizes = [(len(it["segments"]), max(s.size(0) for s in it["segments"]))
                 for it in items[name]]
        sampler = BucketSampler(sizes, cfg.train.batch_size, name == "train", generator)
        loaders[name] = DataLoader(items[name], batch_sampler=sampler, collate_fn=collate)

    every = [it for part in items.values() for it in part]
    table_sizes = (max(it["graph"].size(0) for it in every) + 8,
                   max(s.size(0) for it in every for s in it["segments"]) + 8,
                   max(it["target"].size(0) for it in every) + 8)
    # The decoder may emit up to the longest *training* target plus a margin;
    # a longer held-out target simply cannot be produced and counts as wrong.
    decode_budget = max(it["target"].size(0) for it in items["train"]) + 8

    stats = {
        "records": len(records), "train": len(train), "val": len(val), "test": len(test),
        "vocab": {"graph": len(graph_vocab), "amplitude": len(amp_vocab),
                  "target": len(target_vocab)},
        "longest": {"sq_amp_characters": max(len(r["sq_amp"]) for r in records),
                    "amp_tokens": max(len(r["amp_tokens"]) for r in records),
                    "diagram_tokens": max(len(s) for r in records for s in r["amp_segments"]),
                    "target_tokens": max(len(r["target_tokens"]) for r in records)},
        "segment_amp": segment_amp,
    }
    if verbose:
        print(f"[{cfg.data.theory}] {len(records)} records | train/val/test "
              f"{len(train)}/{len(val)}/{len(test)} | vocab {stats['vocab']} | "
              f"per-diagram encoding: {segment_amp}", flush=True)

    return Data(train, val, test, graph_vocab, amp_vocab, target_vocab, loaders,
                table_sizes, decode_budget, segment_amp, stats)
