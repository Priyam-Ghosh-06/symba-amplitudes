"""Run a trained model on new amplitudes.

Inference has to apply *exactly* the preprocessing training used, or the model
sees a different input distribution than it was fitted on. So this reuses the
same S2-S5 functions rather than reimplementing them, and the only thing it
skips is the target side, which does not exist for an unseen amplitude.
"""

import torch

from .checkpoint import load as load_checkpoint
from .config import PAD
from .data.ast_parse import amp_to_prefix, amp_to_segments
from .data.graph import FeynmanGraph
from .data.normalize import standardize, strip_keywords
from .data.serialize import from_prefix, is_well_formed
from .eval.decode import ConstraintMask, beam_search


def prepare_record(interaction: str, vertices: str, amp: str) -> dict:
    """Raw fields -> the parsed streams the encoders consume (S2, S4, S5)."""
    interaction_clean = strip_keywords(interaction)
    vertices_clean = strip_keywords(vertices)
    amp_std = standardize(amp)

    graph = FeynmanGraph(interaction_clean, vertices_clean)
    if not graph.is_connected():
        raise ValueError("Feynman graph is disconnected")

    return {
        "graph_tokens": graph.to_tokens(),
        "amp_tokens": amp_to_prefix(amp_std),
        "amp_segments": amp_to_segments(amp_std),
        "amp_std": amp_std,
        "graph": graph,
    }


def parse_line(line: str):
    """One SYMBA corpus line -> ``(interaction, vertices, amp, sq_amp)``."""
    parts = line.split(" : ")
    if len(parts) != 4:
        raise ValueError(f"expected 4 ' : '-separated fields, got {len(parts)}")
    return tuple(p.strip() for p in parts)


def _pad(sequences, device):
    longest = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), longest), PAD, dtype=torch.long,
                     device=device)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return ids, ids.eq(PAD)


class Predictor:
    """A loaded checkpoint plus the decoding it was evaluated with."""

    def __init__(self, checkpoint_path: str, device=None,
                 beam_width: int = None, constrained: bool = None):
        self.device = device or torch.device("cpu")
        (self.model, self.cfg, vocabs, self.meta) = load_checkpoint(
            checkpoint_path, self.device)
        self.graph_vocab, self.amp_vocab, self.target_vocab = vocabs
        self.constraint = ConstraintMask(self.target_vocab)

        self.beam_width = (beam_width if beam_width is not None
                           else self.cfg.train.beam_width)
        self.constrained = (constrained if constrained is not None
                            else self.cfg.train.constrained_decoding)
        self.max_len = self.meta["decode_budget"]
        self.segment_amp = self.meta["segment_amp"]

    def _batch(self, prepared: list) -> dict:
        device = self.device
        graph, graph_mask = _pad(
            [self.graph_vocab.encode(r["graph_tokens"]) for r in prepared],
            device)
        amp, amp_mask = _pad(
            [self.amp_vocab.encode(r["amp_tokens"]) for r in prepared], device)

        if self.segment_amp:
            encoded = [[self.amp_vocab.encode(s) for s in r["amp_segments"]]
                       for r in prepared]
        else:
            encoded = [[self.amp_vocab.encode(r["amp_tokens"])]
                       for r in prepared]

        n_seg = max(len(e) for e in encoded)
        seg_len = max(len(s) for e in encoded for s in e)
        seg_ids = torch.full((len(prepared), n_seg, seg_len), PAD,
                             dtype=torch.long, device=device)
        for i, segments in enumerate(encoded):
            for j, segment in enumerate(segments):
                seg_ids[i, j, :len(segment)] = torch.tensor(
                    segment, dtype=torch.long, device=device)

        return {
            "graph": graph, "graph_mask": graph_mask,
            "amp": amp, "amp_mask": amp_mask,
            "amp_segments": seg_ids, "amp_segments_mask": seg_ids.eq(PAD),
        }

    @torch.no_grad()
    def predict(self, records: list, batch_size: int = 8) -> list:
        """``[{interaction, vertices, amp}, ...]`` -> predictions.

        Each result carries the decoded token sequence, the sympy expression it
        denotes, and whether it parsed. With constrained decoding on, it always
        parses - that is the point of the constraint.
        """
        prepared = [prepare_record(r["interaction"], r["vertices"], r["amp"])
                    for r in records]

        results = []
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start:start + batch_size]
            batch = self._batch(chunk)
            decoded = beam_search(
                self.model, batch, self.target_vocab,
                beam_width=self.beam_width, max_len=self.max_len,
                length_penalty=self.cfg.train.length_penalty,
                constrained=self.constrained, constraint=self.constraint)

            for ids in decoded:
                tokens = self.target_vocab.decode(ids[1:])
                well_formed = is_well_formed(tokens)
                expression = None
                if well_formed:
                    try:
                        expression = from_prefix(tokens)
                    except Exception:
                        well_formed = False
                results.append({
                    "tokens": tokens,
                    "prefix": " ".join(tokens),
                    "expression": expression,
                    "well_formed": well_formed,
                })
        return results
