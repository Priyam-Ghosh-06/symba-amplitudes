r"""Three named attention operators, and padding masks on all of them.

The proposal's motivation for XSA - "a token should not trivially attend to
itself" - is a statement about the attention *weights*, which is
``xsa_mask``: zero the diagonal of :math:`QK^\top` before the softmax and let
the rest renormalise. The code that was written is ``xsa_proj``: after the
softmax, remove the component of the output along that position's own value
vector. These are different maps with different fixed points, so both exist as
arms and the claim gets tested against the operator the argument is about
(03 SS4.3).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_KINDS = ("vanilla", "xsa_proj", "xsa_mask")


def _expand_key_padding_mask(mask, num_heads, q_len):
    """``(B, S)`` boolean pad mask to an additive ``(B, H, L, S)`` mask."""
    if mask is None:
        return None
    B, S = mask.shape
    additive = torch.zeros(B, 1, 1, S, dtype=torch.float32, device=mask.device)
    additive = additive.masked_fill(mask[:, None, None, :], float("-inf"))
    return additive.expand(B, num_heads, q_len, S)


class MultiHeadAttention(nn.Module):
    """Self- or cross-attention with a selectable exclusivity operator."""

    def __init__(self, d_model, num_heads, kind="vanilla", dropout=0.1,
                 cross=False):
        super().__init__()
        if kind not in ATTENTION_KINDS:
            raise ValueError(f"unknown attention kind {kind!r}")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")

        self.kind = kind
        self.cross = cross
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout_p = dropout

        if cross:
            self.q_proj = nn.Linear(d_model, d_model)
            self.kv_proj = nn.Linear(d_model, d_model * 2)
        else:
            self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def _shape(self, x, B, L):
        return x.reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)

    def forward(self, x, memory=None, key_padding_mask=None, causal=False,
                need_weights=False):
        B, L, D = x.shape

        if self.cross:
            q = self._shape(self.q_proj(x), B, L)
            M = memory.size(1)
            kv = self.kv_proj(memory).reshape(B, M, 2, self.num_heads,
                                              self.d_head)
            k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        else:
            qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            M = L

        attn_mask = _expand_key_padding_mask(key_padding_mask, self.num_heads, L)

        if self.kind == "xsa_mask" and not self.cross:
            # Forbid every position from attending to itself, then renormalise.
            diagonal = torch.eye(L, dtype=torch.bool, device=x.device)
            diag_mask = torch.zeros(1, 1, L, L, dtype=torch.float32,
                                    device=x.device)
            diag_mask = diag_mask.masked_fill(diagonal[None, None], float("-inf"))
            attn_mask = diag_mask if attn_mask is None else attn_mask + diag_mask

        if causal:
            causal_mask = torch.zeros(1, 1, L, M, dtype=torch.float32,
                                      device=x.device)
            causal_mask = causal_mask.masked_fill(
                torch.ones(L, M, dtype=torch.bool, device=x.device)
                .triu(1)[None, None], float("-inf"))
            attn_mask = causal_mask if attn_mask is None else attn_mask + causal_mask

        weights = None
        if need_weights:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            if attn_mask is not None:
                scores = scores + attn_mask
            weights = F.softmax(scores, dim=-1)
            # A fully masked row (an all-pad query) softmaxes to NaN; those rows
            # are dropped by the loss mask anyway, so zero them for safety.
            weights = torch.nan_to_num(weights)
            out = torch.matmul(weights, v)
        else:
            if attn_mask is not None:
                # A row that is entirely -inf produces NaN; keep one legal slot.
                fully_masked = torch.isinf(attn_mask).all(dim=-1, keepdim=True)
                attn_mask = attn_mask.masked_fill(fully_masked, 0.0)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0)

        if self.kind == "xsa_proj":
            # Y <- Y - <Y, v> / (||v||^2) v, with a *relative* epsilon so the
            # projection is not systematically under-applied at initialisation
            # when the value vectors are still small (02 SS3.11).
            if not self.cross:
                dot = (out * v).sum(dim=-1, keepdim=True)
                norm_sq = (v * v).sum(dim=-1, keepdim=True)
                eps = 1e-6 * norm_sq.mean().clamp(min=1e-12)
                out = out - (dot / (norm_sq + eps)) * v

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), weights
