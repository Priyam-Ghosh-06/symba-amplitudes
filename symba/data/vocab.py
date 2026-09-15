"""Token <-> id mapping."""

from collections import Counter

from ..config import EOS, PAD, SOS, SPECIAL_TOKENS, UNK
from .serialize import DIGITS, type_ids


class Vocab:
    """Specials first, then digits (fixed ids), then tokens by frequency."""

    def __init__(self, token_lists, reserve_digits=True):
        counts = Counter()
        for tokens in token_lists:
            counts.update(tokens)
        fixed = SPECIAL_TOKENS + (DIGITS if reserve_digits else [])
        rest = sorted((t for t in counts if t not in fixed), key=lambda t: (-counts[t], t))
        self._set(fixed + rest)

    @classmethod
    def from_itos(cls, itos):
        vocab = cls([], reserve_digits=False)
        vocab._set(list(itos))
        return vocab

    def _set(self, itos):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [SOS] + [self.stoi.get(t, UNK) for t in tokens] + [EOS]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            if i not in (PAD, SOS, UNK):
                out.append(self.itos[i])
        return out

    def type_ids(self):
        return type_ids(self.itos)
