"""Save a trained model, load it back, and predict squared amplitudes."""

import torch

from .config import Config
from .data.dataset import collate, encode, preprocess
from .data.serialize import from_prefix, is_well_formed
from .data.vocab import Vocab
from .decode import beam_search
from .model.model import AmplitudeModel


def save_checkpoint(path, model, cfg, data):
    torch.save({"config": cfg.to_dict(),
                "model": model.state_dict(),
                "vocabs": [data.graph_vocab.itos, data.amp_vocab.itos, data.target_vocab.itos],
                "table_sizes": list(data.table_sizes),
                "decode_budget": data.decode_budget}, path)


class Predictor:
    def __init__(self, path, device="cpu"):
        blob = torch.load(path, map_location=device, weights_only=True)
        self.cfg = Config.from_dict(blob["config"])
        self.vocabs = [Vocab.from_itos(itos) for itos in blob["vocabs"]]
        self.model = AmplitudeModel(self.cfg.model, *self.vocabs, blob["table_sizes"])
        self.model.load_state_dict(blob["model"])
        self.model.to(device).eval()
        self.decode_budget = blob["decode_budget"]
        self.device = device

    def predict(self, interaction, vertices, amp):
        """The predicted squared amplitude as a sympy expression (None if the
        output is not a valid expression), and its prefix tokens."""
        record = preprocess({"interaction": interaction, "vertices": vertices, "amp": amp})
        if not self.cfg.data.segment_amp:
            record["amp_segments"] = [record["amp_tokens"]]
        batch = collate([encode(record, *self.vocabs[:2])])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        ids = beam_search(self.model, batch, self.vocabs[2], self.cfg.train.beam_width,
                          self.decode_budget, self.cfg.train.length_penalty)[0]
        tokens = self.vocabs[2].decode(ids[1:])
        return (from_prefix(tokens) if is_well_formed(tokens) else None), tokens
