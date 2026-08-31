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
                need_weights=False, cache=None):
        """``cache`` is a dict this layer may read from and write to.

        During incremental decoding ``x`` is the single newest position. Self
        attention appends its K/V to ``cache['self']`` and attends over the
        whole run; cross attention projects the memory once and reuses it,
        since the memory is fixed for the entire generation. Without this the
        decoder recomputes the full prefix at every step, which is O(W*T^2)
        work where O(T) suffices (02 SS4.2).
        """
        B, L, D = x.shape
        incremental = cache is not None

        if self.cross:
            q = self._shape(self.q_proj(x), B, L)
            if incremental and "cross_kv" in cache:
                k, v = cache["cross_kv"]
            else:
                M = memory.size(1)
                kv = self.kv_proj(memory).reshape(B, M, 2, self.num_heads,
                                                  self.d_head)
                k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
                if incremental:
                    cache["cross_kv"] = (k, v)
            M = k.size(2)
        else:
            qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            if incremental:
                if "self_kv" in cache:
                    past_k, past_v = cache["self_kv"]
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)
                cache["self_kv"] = (k, v)
                # Every cached key precedes the new query, so the causal mask
                # is already satisfied by construction.
                causal = False
            M = k.size(2)

        attn_mask = _expand_key_padding_mask(key_padding_mask, self.num_heads, L)

        if self.kind == "xsa_mask" and not self.cross:
            # Forbid every position from attending to itself, then renormalise.
            # Under a KV cache the queries are the last L of the M keys, so the
            # "self" key for query i sits at column M - L + i.
            self_index = torch.arange(L, device=x.device) + (M - L)
            diagonal = F.one_hot(self_index, M).bool()
            diag_mask = torch.zeros(1, 1, L, M, dtype=torch.float32,
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
                # Align the value vectors with the queries: under a KV cache
                # v holds the whole run while out holds only the new positions.
                v_self = v[:, :, -L:]
                dot = (out * v_self).sum(dim=-1, keepdim=True)
                norm_sq = (v_self * v_self).sum(dim=-1, keepdim=True)
                eps = 1e-6 * norm_sq.mean().clamp(min=1e-12)
                out = out - (dot / (norm_sq + eps)) * v_self

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), weights
