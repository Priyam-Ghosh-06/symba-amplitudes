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

            # One tensor per diagram. With segmentation off this is a single
            # segment holding the whole amplitude, so the two paths share code.
            segments = [torch.tensor(amp_vocab.encode(seg), dtype=torch.long)
                        for seg in record["amp_segments"]]

            self.items.append({
                "graph": torch.tensor(ids[0], dtype=torch.long),
                "amp": torch.tensor(ids[1], dtype=torch.long),
                "amp_segments": segments,
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


def _pad_segments(batch):
    """Stack per-diagram segments into ``(B, S, L)`` plus its padding mask.

    S is the largest diagram count in the batch and L the longest segment, so
    the encoder attends within a diagram over L rather than across the whole
    concatenated amplitude. Empty slots are all-PAD and fully masked.
    """
    n_seg = max(len(b["amp_segments"]) for b in batch)
    seg_len = max(t.size(0) for b in batch for t in b["amp_segments"])

    ids = torch.full((len(batch), n_seg, seg_len), PAD, dtype=torch.long)
    for i, item in enumerate(batch):
        for j, segment in enumerate(item["amp_segments"]):
            ids[i, j, :segment.size(0)] = segment
    return ids, ids.eq(PAD)


def collate(batch):
    graph, graph_mask = _pad_stack([b["graph"] for b in batch])
    amp, amp_mask = _pad_stack([b["amp"] for b in batch])
    amp_seg, amp_seg_mask = _pad_segments(batch)
    target, target_mask = _pad_stack([b["target"] for b in batch])
    return {
        "graph": graph, "graph_mask": graph_mask,
        "amp": amp, "amp_mask": amp_mask,
        "amp_segments": amp_seg, "amp_segments_mask": amp_seg_mask,
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

        def cost(i):
            # Segments are padded to a rectangle per batch, so the thing to
            # group by is (diagram count, longest diagram) - sorting on the
            # flat length would put a 2-diagram record next to a 14-diagram one
            # and pad the whole batch up to the larger.
            item = dataset[i]
            segments = item["amp_segments"]
            return (len(segments), max(t.size(0) for t in segments))

        self.order = sorted(range(len(dataset)), key=cost)

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
