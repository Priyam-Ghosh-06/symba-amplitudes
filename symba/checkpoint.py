"""Saving and reloading a trained model.

Training previously kept the selected weights in memory and threw them away
when the process exited, so there was no artefact to run inference from. A
checkpoint has to carry everything needed to rebuild the model *and* to turn a
raw amplitude into the tensors it expects: the three vocabularies, the sequence
lengths the positional tables were sized from, and whether the amplitude was
encoded per diagram.
"""

import os

import torch

from .config import Config
from .data.vocab import Vocab
from .model.model import AmplitudeModel

# 2: carries ``decode_budget`` separately from ``lengths``. Version 1
# checkpoints predate the pipeline-correctness pass and are not loadable; they
# were produced by code with a different decode budget, so silently accepting
# them would reproduce neither their numbers nor the current ones.
FORMAT_VERSION = 2


def _vocab_state(vocab: Vocab) -> dict:
    return {"itos": list(vocab.itos)}


def _vocab_from_state(state: dict) -> Vocab:
    """Rebuild a vocabulary from its token list, preserving every id."""
    vocab = Vocab.__new__(Vocab)
    vocab.itos = list(state["itos"])
    vocab.stoi = {token: i for i, token in enumerate(vocab.itos)}
    vocab.counts = {}
    return vocab


def save(path: str, model: AmplitudeModel, cfg: Config, bundle,
         metrics: dict = None):
    """Write a self-contained checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "format_version": FORMAT_VERSION,
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
        "lengths": list(bundle.lengths),
        "decode_budget": int(bundle.decode_budget),
        "segment_amp": bool(bundle.segment_amp),
        "segment_len": int(bundle.segment_len),
        "graph_vocab": _vocab_state(bundle.graph_vocab),
        "amp_vocab": _vocab_state(bundle.amp_vocab),
        "target_vocab": _vocab_state(bundle.target_vocab),
        "data_stats": bundle.stats,
        "metrics": metrics or {},
    }, path)
    return path


def load(path: str, device=None):
    """Rebuild ``(model, cfg, vocabs, meta)`` from a checkpoint."""
    device = device or torch.device("cpu")
    # weights_only=False: the payload holds config dicts and vocab lists, not
    # just tensors. Only load checkpoints you produced.
    blob = torch.load(path, map_location=device, weights_only=False)

    version = blob.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: checkpoint format {version}, expected {FORMAT_VERSION}")

    cfg = Config.from_dict(blob["config"])
    graph_vocab = _vocab_from_state(blob["graph_vocab"])
    amp_vocab = _vocab_from_state(blob["amp_vocab"])
    target_vocab = _vocab_from_state(blob["target_vocab"])

    model = AmplitudeModel(cfg.model, graph_vocab, amp_vocab, target_vocab,
                           tuple(blob["lengths"]),
                           segment_amp=blob["segment_amp"],
                           segment_len=blob["segment_len"])
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()

    meta = {
        "lengths": tuple(blob["lengths"]),
        "decode_budget": blob["decode_budget"],
        "segment_amp": blob["segment_amp"],
        "segment_len": blob["segment_len"],
        "data_stats": blob.get("data_stats", {}),
        "metrics": blob.get("metrics", {}),
    }
    return model, cfg, (graph_vocab, amp_vocab, target_vocab), meta
