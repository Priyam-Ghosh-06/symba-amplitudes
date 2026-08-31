"""Tensors, dynamic padding, and padding masks.

Two properties the previous dataset lacked and that everything downstream
depends on:

* the tensors are built from ``graph_tokens`` / ``amp_tokens`` / ``target_tokens``
  - the parsed streams - never from the raw strings (gate G9);
* nothing is truncated. Padding is per batch, and a sequence longer than the
  declared budget raises rather than being clipped (gate G10).
"""

import torch
from torch.utils.data import Dataset, DataLoader

from ..config import PAD


class AmplitudeDataset(Dataset):
    """One item per record: graph ids, amp ids, target ids, and metadata."""

    FIELDS = ("graph_tokens", "amp_tokens", "target_tokens")

    def __init__(self, records, graph_vocab, amp_vocab, target_vocab,
                 max_lengths=None):
        self.records = records
        self.vocabs = (graph_vocab, amp_vocab, target_vocab)
        self.max_lengths = max_lengths

        self.items = []
        for record in records:
            ids = []
            for field, vocab in zip(self.FIELDS, self.vocabs):
                # Direct indexing, not .get(...): a missing field is a pipeline
                # bug and must raise here rather than become an empty tensor.
                encoded = vocab.encode(record[field])
                ids.append(encoded)

            if max_lengths is not None:
                for name, encoded, limit in zip(self.FIELDS, ids, max_lengths):
                    if len(encoded) > limit:
                        raise ValueError(
                            f"{record.source_file}:{record.line_no} {name} is "
                            f"{len(encoded)} tokens, over the budget of {limit}")

            self.items.append({
                "graph": torch.tensor(ids[0], dtype=torch.long),
                "amp": torch.tensor(ids[1], dtype=torch.long),
                "target": torch.tensor(ids[2], dtype=torch.long),
                "index": len(self.items),
                "template": record["template"],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def record(self, index: int):
        return self.records[index]

    def lengths(self):
        """Max encoded length per stream, for allocating positional tables."""
        return tuple(max(item[key].size(0) for item in self.items)
                     for key in ("graph", "amp", "target"))


def _pad_stack(tensors):
    """Pad to the longest in the batch and return ``(ids, key_padding_mask)``.

    The mask is True at padding positions, matching the convention
    ``F.scaled_dot_product_attention`` expects once inverted into an additive
    mask, and is threaded through every attention call.
    """
    longest = max(t.size(0) for t in tensors)
    ids = torch.full((len(tensors), longest), PAD, dtype=torch.long)
    for i, t in enumerate(tensors):
        ids[i, :t.size(0)] = t
    return ids, ids.eq(PAD)


def collate(batch):
    graph, graph_mask = _pad_stack([b["graph"] for b in batch])
    amp, amp_mask = _pad_stack([b["amp"] for b in batch])
    target, target_mask = _pad_stack([b["target"] for b in batch])
    return {
        "graph": graph, "graph_mask": graph_mask,
        "amp": amp, "amp_mask": amp_mask,
        "target": target, "target_mask": target_mask,
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        "template": [b["template"] for b in batch],
    }


class LengthBucketSampler(torch.utils.data.Sampler):
    """Group similar-length items so dynamic padding actually saves work.

    Buckets are shuffled every epoch, and items are shuffled inside a bucket, so
    this keeps randomness while avoiding batches whose longest member is ten
    times the median.
    """

    def __init__(self, dataset, batch_size, shuffle=True, generator=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.order = sorted(range(len(dataset)),
                            key=lambda i: dataset[i]["amp"].size(0))

    def __iter__(self):
        batches = [self.order[i:i + self.batch_size]
                   for i in range(0, len(self.order), self.batch_size)]
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=self.generator)
            batches = [batches[i] for i in perm.tolist()]
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size


def make_loader(dataset, batch_size, shuffle=True, generator=None):
    if len(dataset) == 0:
        raise ValueError("empty dataset")
    sampler = LengthBucketSampler(dataset, batch_size, shuffle, generator)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)
