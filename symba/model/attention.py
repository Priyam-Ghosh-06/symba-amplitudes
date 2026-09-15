"""Multi-head attention, with padding masks, a KV cache for decoding, and
optional exclusive self-attention (XSA).

XSA removes from each output the component along the token's own value vector,
``y <- y - (<y, v> / |v|^2) v``, so a token's new representation has to come
from the other tokens rather than from itself.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _padding_mask(mask, num_heads, q_len):
    """``(B, S)`` boolean pad mask -> additive ``(B, H, L, S)`` mask."""
    if mask is None:
        return None
    B, S = mask.shape
    additive = torch.zeros(B, 1, 1, S, device=mask.device)
    additive = additive.masked_fill(mask[:, None, None, :], float("-inf"))
    return additive.expand(B, num_heads, q_len, S)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, kind="vanilla", dropout=0.1, cross=False):
        super().__init__()
        if kind not in ("vanilla", "xsa"):
            raise ValueError(f"unknown attention {kind!r}")
        self.kind, self.cross, self.dropout = kind, cross, dropout
        self.num_heads, self.d_head = num_heads, d_model // num_heads
        if cross:
            self.q_proj = nn.Linear(d_model, d_model)
            self.kv_proj = nn.Linear(d_model, 2 * d_model)
        else:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, memory=None, key_padding_mask=None, causal=False,
                need_weights=False, cache=None):
        """With a ``cache`` dict, ``x`` is just the newest decoder position:
        earlier keys and values are reused and the memory is projected once."""
        B, L, D = x.shape
        H, d = self.num_heads, self.d_head

        if self.cross:
            q = self.q_proj(x).reshape(B, L, H, d).transpose(1, 2)
            if cache is not None and "kv" in cache:
                k, v = cache["kv"]
            else:
                kv = self.kv_proj(memory).reshape(B, memory.size(1), 2, H, d)
                k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
                if cache is not None:
                    cache["kv"] = (k, v)
        else:
            q, k, v = self.qkv_proj(x).reshape(B, L, 3, H, d).permute(2, 0, 3, 1, 4).unbind(0)
            if cache is not None:
                if "kv" in cache:
                    k = torch.cat([cache["kv"][0], k], dim=2)
                    v = torch.cat([cache["kv"][1], v], dim=2)
                cache["kv"] = (k, v)
                causal = False              # every cached key precedes the new query
        M = k.size(2)

        mask = _padding_mask(key_padding_mask, H, L)
        if causal:
            future = torch.ones(L, M, dtype=torch.bool, device=x.device).triu(1)
            causal_mask = torch.zeros(1, 1, L, M, device=x.device).masked_fill(future, float("-inf"))
            mask = causal_mask if mask is None else mask + causal_mask

        weights = None
        if need_weights:
            scores = q @ k.transpose(-2, -1) / math.sqrt(d)
            if mask is not None:
                scores = scores + mask
            weights = torch.nan_to_num(scores.softmax(-1))
            out = weights @ v
        else:
            if mask is not None:
                # A fully padded row (an empty diagram slot) would give NaN.
                mask = mask.masked_fill(torch.isinf(mask).all(-1, keepdim=True), 0.0)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)

        if self.kind == "xsa" and not self.cross:
            v_self = v[:, :, -L:]
            norm_sq = (v_self * v_self).sum(-1, keepdim=True)
            eps = 1e-6 * norm_sq.mean().clamp(min=1e-12)
            out = out - (out * v_self).sum(-1, keepdim=True) / (norm_sq + eps) * v_self

        return self.out_proj(out.transpose(1, 2).reshape(B, L, D)), weights
