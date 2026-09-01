"""Embeddings: identity, type, and positional role.

A token like ``s_13`` carries an identity, a *type* (Mandelstam vs mass vs
operator vs digit) known exactly from the grammar, and a structural role. The
type channel is free supervision that a byte-level model throws away, and it is
what makes "do the experts partition by symbol class" a measurable question
rather than a rhetorical one (03 SS4.1).
"""

import torch
import torch.nn as nn

from ..config import TOKEN_TYPES


class TokenEmbedding(nn.Module):
    """Identity + optional type + learned position.

    Positional tables are sized from the data, not from a 2048 default that
    would allocate ~1M rows per table and never index past 300 (02 SS3.10).
    """

    def __init__(self, vocab_size, d_model, max_len, type_ids=None,
                 dropout=0.1, scheme="plain"):
        super().__init__()
        self.scheme = scheme
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        if type_ids is not None:
            self.register_buffer("type_of_token",
                                 torch.tensor(type_ids, dtype=torch.long))
            self.type_embed = nn.Embedding(len(TOKEN_TYPES), d_model)
        else:
            self.type_of_token = None
            self.type_embed = None

        if scheme == "role_filler":
            # Kept as an arm. Note it is affine: with no nonlinearity between
            # them, C(Ax + Br) collapses to (CA)x + (CB)r, so the extra factor
            # buys no expressivity (02 SS3.5). Reported as such.
            self.A_e = nn.Linear(d_model, d_model, bias=False)
            self.B_e = nn.Linear(d_model, d_model, bias=False)
            self.C_e = nn.Linear(d_model, d_model, bias=True)
        elif scheme == "tpr":
            # A real binding: the role multiplicatively gates the filler, so
            # unbinding with the role recovers the filler. This is the property
            # additive composition destroys and that TPR exists to provide.
            self.role_gate = nn.Linear(d_model, d_model, bias=False)
        elif scheme != "plain":
            raise ValueError(f"unknown embedding scheme {scheme!r}")

    @property
    def filler_embed(self):
        """The matrix weight tying should target."""
        return self.token_embed

    def forward(self, ids, offset: int = 0):
        B, L = ids.shape
        positions = (torch.arange(L, device=ids.device) + offset
                     ).unsqueeze(0).expand(B, -1)

        filler = self.token_embed(ids)
        if self.type_embed is not None:
            filler = filler + self.type_embed(self.type_of_token[ids])
        role = self.pos_embed(positions)

        if self.scheme == "role_filler":
            x = self.C_e(self.A_e(filler) + self.B_e(role))
        elif self.scheme == "tpr":
            x = filler * torch.tanh(self.role_gate(role))
        else:
            x = filler + role

        return self.dropout(self.norm(x))

