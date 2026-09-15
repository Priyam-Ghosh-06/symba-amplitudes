"""Token embeddings.

Each token has a *filler* - what it is: its identity plus its grammar type -
and a *role* - where it is. ``plain`` adds the two. ``role_filler`` maps each
through its own matrix before combining, ``C(A filler + B role)``, so what a
token is and where it sits are learned separately.
"""

import torch
import torch.nn as nn

from ..config import TOKEN_TYPES


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, type_ids, dropout, scheme="role_filler"):
        super().__init__()
        if scheme not in ("plain", "role_filler"):
            raise ValueError(f"unknown embedding {scheme!r}")
        self.scheme = scheme
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.register_buffer("type_of_token", torch.tensor(type_ids, dtype=torch.long))
        self.type_embed = nn.Embedding(len(TOKEN_TYPES), d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        if scheme == "role_filler":
            self.A = nn.Linear(d_model, d_model, bias=False)
            self.B = nn.Linear(d_model, d_model, bias=False)
            self.C = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, offset=0):
        positions = torch.arange(ids.size(1), device=ids.device) + offset
        filler = self.token_embed(ids) + self.type_embed(self.type_of_token[ids])
        role = self.pos_embed(positions)[None]
        if self.scheme == "role_filler":
            x = self.C(self.A(filler) + self.B(role))
        else:
            x = filler + role
        return self.dropout(self.norm(x))
