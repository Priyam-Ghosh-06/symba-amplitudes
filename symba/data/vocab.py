"""S6 - vocabulary, built from the training split only.

The previous tokeniser counted tokens over the whole corpus and split two cells
later (02 SS2.3). Here construction takes a single list and the caller is the
training split; gate G8 asserts that ordering, and gate G7 reports the OOV rate
on val/test rather than hiding it behind an ``<unk>``.
"""

from collections import Counter

from ..config import PAD, SOS, EOS, UNK, SPECIAL_TOKENS
from .serialize import DIGITS, type_ids


class Vocab:
    """Deterministic token/id mapping.

    Ids are assigned specials first, then digits (so a digit id never moves when
    the corpus changes), then the remaining symbols sorted by ``(-count, token)``
    - frequency for interpretability, token name to break ties reproducibly.
    """

    def __init__(self, tokens_iter, reserve_digits: bool = True):
        counts = Counter()
        for tokens in tokens_iter:
            counts.update(tokens)

        self.itos = list(SPECIAL_TOKENS)
        if reserve_digits:
            self.itos += DIGITS

        for token in sorted(counts, key=lambda t: (-counts[t], t)):
            if token not in self.itos:
                self.itos.append(token)

        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.counts = counts

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, add_bos_eos: bool = True) -> list:
        ids = [self.stoi.get(t, UNK) for t in tokens]
        return [SOS] + ids + [EOS] if add_bos_eos else ids

    def decode(self, ids, strip_special: bool = True) -> list:
        out = []
        for i in ids:
            if i == EOS:
                break
            if strip_special and i in (PAD, SOS, UNK):
                continue
            out.append(self.itos[i])
        return out

    def type_ids(self) -> list:
        """Type id per vocabulary entry, for the type embedding channel."""
        return type_ids(self.itos)

    def oov_rate(self, tokens_iter):
        """Gate G7: fraction of tokens in a held-out split absent from here."""
        total = missing = 0
        unseen = Counter()
        for tokens in tokens_iter:
            for token in tokens:
                total += 1
                if token not in self.stoi:
                    missing += 1
                    unseen[token] += 1
        return (missing / total if total else 0.0), unseen
