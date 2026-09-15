"""Two encoders and a cross-attention decoder.

    Feynman graph tokens -------> graph encoder --.
                                                   +--> memory --> decoder --> sq_amp tokens
    amplitude, one sequence ----> math encoder ---'
    per Feynman diagram           (shared weights)

The decoder cross-attends over the graph and all diagrams at once; a learned
modality embedding tells it which memory positions come from which encoder.
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .embed import TokenEmbedding
from .ffn import MoEFFN, build_ffn


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.attention, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = build_ffn(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, pad_mask):
        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask=pad_mask)[0])
        return x + self.ffn(self.norm2(x))


class Encoder(nn.Module):
    def __init__(self, cfg, vocab, max_len, n_layers):
        super().__init__()
        self.embed = TokenEmbedding(len(vocab), cfg.d_model, max_len, vocab.type_ids(),
                                    cfg.dropout, cfg.embedding)
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, ids, pad_mask):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """Causal self-attention, cross-attention over the encoder memory, FFN."""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, "vanilla", cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, "vanilla",
                                             cfg.dropout, cross=True)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.ffn = build_ffn(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, memory_mask, need_weights=False, cache=None):
        self_cache = cache.setdefault("self", {}) if cache is not None else None
        cross_cache = cache.setdefault("cross", {}) if cache is not None else None
        x = x + self.dropout(self.self_attn(self.norm1(x), causal=True, cache=self_cache)[0])
        crossed, weights = self.cross_attn(self.norm2(x), memory=memory,
                                           key_padding_mask=memory_mask,
                                           need_weights=need_weights, cache=cross_cache)
        x = x + self.dropout(crossed)
        return x + self.ffn(self.norm3(x)), weights


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=module.embedding_dim ** -0.5)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0)


class AmplitudeModel(nn.Module):
    def __init__(self, cfg, graph_vocab, amp_vocab, target_vocab, table_sizes):
        super().__init__()
        if not (cfg.use_graph or cfg.use_math):
            raise ValueError("at least one encoder must be enabled")
        graph_len, segment_len, target_len = table_sizes

        self.graph_encoder = (Encoder(cfg, graph_vocab, graph_len, cfg.graph_layers)
                              if cfg.use_graph else None)
        self.math_encoder = (Encoder(cfg, amp_vocab, segment_len, cfg.math_layers)
                             if cfg.use_math else None)
        self.modality_embed = nn.Embedding(2, cfg.d_model)
        self.decoder_embed = TokenEmbedding(len(target_vocab), cfg.d_model, target_len,
                                            target_vocab.type_ids(), cfg.dropout, cfg.embedding)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.dec_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.fc_out = nn.Linear(cfg.d_model, len(target_vocab), bias=False)

        self.apply(_init_weights)
        self.fc_out.weight = self.decoder_embed.token_embed.weight   # tied, after init

    def encode(self, batch):
        """Encoder memory ``(B, M, d)`` and its padding mask ``(B, M)``."""
        parts, masks = [], []
        if self.graph_encoder is not None:
            g = self.graph_encoder(batch["graph"], batch["graph_mask"])
            parts.append(g + self.modality_embed.weight[0])
            masks.append(batch["graph_mask"])
        if self.math_encoder is not None:
            # Diagrams are folded into the batch, so attention runs within one
            # diagram, then laid back out as one memory sequence.
            ids, mask = batch["segments"], batch["segments_mask"]
            B, S, L = ids.shape
            m = self.math_encoder(ids.reshape(B * S, L), mask.reshape(B * S, L))
            parts.append(m.reshape(B, S * L, -1) + self.modality_embed.weight[1])
            masks.append(mask.reshape(B, S * L))
        return torch.cat(parts, 1), torch.cat(masks, 1)

    def decode_step(self, target_in, memory, memory_mask, need_weights=False,
                    cache=None, offset=0):
        """Logits for ``target_in``. With a ``cache``, ``target_in`` is only the
        newest token and ``offset`` its position. Returns the last layer's
        cross-attention weights when ``need_weights``."""
        x = self.decoder_embed(target_in, offset=offset)
        weights = None
        for i, block in enumerate(self.blocks):
            x, weights = block(x, memory, memory_mask, need_weights,
                               cache=cache.setdefault(i, {}) if cache is not None else None)
        return self.fc_out(self.norm(x)), weights

    @staticmethod
    def reorder_cache(cache, index):
        """Beam search reorders hypotheses; their cached keys/values must follow."""
        for layer in cache.values():
            for attn in layer.values():
                attn["kv"] = tuple(t.index_select(0, index) for t in attn["kv"])

    def forward(self, batch):
        memory, memory_mask = self.encode(batch)
        logits, _ = self.decode_step(batch["target"][:, :-1], memory, memory_mask)
        return {"logits": logits, "aux_loss": self.aux_loss()}

    def aux_loss(self):
        losses = [m.aux_loss for m in self.modules() if isinstance(m, MoEFFN)]
        return sum(losses) if losses else torch.zeros(())

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())
