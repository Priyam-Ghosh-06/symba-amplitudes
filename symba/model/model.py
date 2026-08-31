"""Dual-pathway encoder/decoder over the parsed graph and amplitude streams.

    graph tokens -> graph encoder --.
                                     >-- concat + modality tag -> decoder -> target
    amp tokens   -> math encoder  --'

Padding masks are threaded through every attention call, including the decoder's
cross-attention over the fused memory, so no PAD position ever participates as a
key (02 SS3.1).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PAD
from .attention import MultiHeadAttention
from .embed import TokenEmbedding
from .ffn import build_ffn


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.num_heads,
                                       cfg.attention, cfg.attn_dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = build_ffn(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding_mask=None, type_ids=None):
        attended, _ = self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.dropout(attended)
        return x + self.ffn(self.norm2(x), type_ids)


class Encoder(nn.Module):
    def __init__(self, cfg, vocab, max_len):
        super().__init__()
        self.embed = TokenEmbedding(
            len(vocab), cfg.d_model, max_len,
            type_ids=vocab.type_ids() if cfg.use_type_embedding else None,
            dropout=cfg.dropout, scheme=cfg.embedding)
        self.blocks = nn.ModuleList([EncoderBlock(cfg)
                                     for _ in range(cfg.graph_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, ids, key_padding_mask=None, type_ids=None):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, key_padding_mask, type_ids)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads,
                                            "vanilla", cfg.attn_dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads,
                                             "vanilla", cfg.attn_dropout,
                                             cross=True)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.ffn = build_ffn(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, memory_mask=None, need_weights=False,
                type_ids=None, cache=None):
        self_cache = cross_cache = None
        if cache is not None:
            self_cache = cache.setdefault("self_attn", {})
            cross_cache = cache.setdefault("cross_attn", {})

        attended, _ = self.self_attn(self.norm1(x), causal=True,
                                     cache=self_cache)
        x = x + self.dropout(attended)

        crossed, weights = self.cross_attn(
            self.norm2(x), memory=memory, key_padding_mask=memory_mask,
            need_weights=need_weights, cache=cross_cache)
        x = x + self.dropout(crossed)

        return x + self.ffn(self.norm3(x), type_ids), weights


class AmplitudeModel(nn.Module):
    """The full system. Ablation arms are config, not runtime flags.

    ``use_graph`` / ``use_math`` build a model that genuinely lacks the pathway
    rather than zeroing it at call time, so a modality ablation cannot silently
    evaluate the full model with the wrong weights (02 SS3.12).
    """

    def __init__(self, cfg, graph_vocab, amp_vocab, target_vocab, lengths,
                 segment_amp: bool = True):
        super().__init__()
        self.cfg = cfg
        self.cfg_data_segments = segment_amp
        self.target_vocab = target_vocab
        graph_len, amp_len, target_len = lengths

        if not (cfg.use_graph or cfg.use_math):
            raise ValueError("at least one encoder pathway must be enabled")

        self.graph_encoder = (Encoder(cfg, graph_vocab, graph_len + 8)
                              if cfg.use_graph else None)
        self.math_encoder = (Encoder(cfg, amp_vocab, amp_len + 8)
                             if cfg.use_math else None)
        self.amp_len = amp_len
        self.modality_embed = nn.Embedding(2, cfg.d_model)

        self.decoder_embed = TokenEmbedding(
            len(target_vocab), cfg.d_model, target_len + 8,
            type_ids=target_vocab.type_ids() if cfg.use_type_embedding else None,
            dropout=cfg.dropout, scheme=cfg.embedding)
        self.blocks = nn.ModuleList([DecoderBlock(cfg)
                                     for _ in range(cfg.dec_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.fc_out = nn.Linear(cfg.d_model, len(target_vocab), bias=False)

        if cfg.gated_fusion:
            self.fusion_gate = nn.Linear(cfg.d_model * 2, cfg.d_model)

        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            # Tie after init: apply() writes fresh tensors and would otherwise
            # break the shared storage.
            self.fc_out.weight = self.decoder_embed.token_embed.weight

    @staticmethod
    def _init_weights(module):
        """Fan-in aware init, and it covers Conv1d, which the old one missed."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=module.embedding_dim ** -0.5)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _encode_amplitude(self, batch):
        """Encode the amplitude, one Feynman diagram at a time.

        The diagrams are folded into the batch dimension, so self-attention
        runs over a single diagram's length rather than the whole concatenated
        amplitude. On QCD that is 10.7x less attention work and drops the
        longest attended sequence from 2859 tokens to 239, which is what makes
        the theory trainable on CPU at all.

        The encoded diagrams are then laid back out as one memory sequence, so
        the decoder still cross-attends over the entire amplitude and nothing
        downstream has to know segmentation happened.
        """
        if not self.cfg_data_segments or "amp_segments" not in batch:
            return self.math_encoder(batch["amp"], batch["amp_mask"]), \
                   batch["amp_mask"]

        ids = batch["amp_segments"]                 # (B, S, L)
        mask = batch["amp_segments_mask"]
        B, S, L = ids.shape

        flat_ids = ids.reshape(B * S, L)
        flat_mask = mask.reshape(B * S, L)
        encoded = self.math_encoder(flat_ids, flat_mask)
        return encoded.reshape(B, S * L, -1), mask.reshape(B, S * L)

    def encode(self, batch):
        """Run whichever encoders exist and fuse them into one memory."""
        parts, masks, modality = [], [], []

        if self.graph_encoder is not None:
            g = self.graph_encoder(batch["graph"], batch["graph_mask"])
            parts.append(g)
            masks.append(batch["graph_mask"])
            modality.append(torch.zeros(g.size(1), dtype=torch.long,
                                        device=g.device))
        if self.math_encoder is not None:
            m, m_mask = self._encode_amplitude(batch)
            parts.append(m)
            masks.append(m_mask)
            modality.append(torch.ones(m.size(1), dtype=torch.long,
                                       device=m.device))

        memory = torch.cat(parts, dim=1)
        memory_mask = torch.cat(masks, dim=1)
        modality_ids = torch.cat(modality).unsqueeze(0).expand(memory.size(0), -1)
        memory = memory + self.modality_embed(modality_ids)

        graph_len = parts[0].size(1) if self.graph_encoder is not None else 0
        return memory, memory_mask, graph_len

    def decode_step(self, target_in, memory, memory_mask, need_weights=False,
                    cache=None, offset=0):
        """One decoder pass.

        With ``cache`` supplied, ``target_in`` is just the newest token and
        ``offset`` is its position, so the positional embedding still sees the
        true index. Without it the whole prefix is re-run, which is what
        training needs.
        """
        x = self.decoder_embed(target_in, offset=offset)
        weights = None
        for i, block in enumerate(self.blocks):
            block_cache = None
            if cache is not None:
                block_cache = cache.setdefault(i, {})
            x, w = block(x, memory, memory_mask, need_weights,
                         cache=block_cache)
            if w is not None:
                weights = w
        return self.fc_out(self.norm(x)), weights

    @staticmethod
    def new_cache() -> dict:
        """Fresh incremental-decoding state for one generation."""
        return {}

    @staticmethod
    def reorder_cache(cache: dict, index: torch.Tensor):
        """Reindex the batch dimension of every cached tensor.

        Beam search permutes hypotheses at every step. The cached keys and
        values are per hypothesis, so they have to travel with it - otherwise a
        surviving beam continues from another beam's history, silently.
        """
        for node in cache.values():
            for attn in node.values():
                for key, (k, v) in list(attn.items()):
                    attn[key] = (k.index_select(0, index),
                                 v.index_select(0, index))

    def forward(self, batch, need_weights=False):
        memory, memory_mask, graph_len = self.encode(batch)
        logits, weights = self.decode_step(batch["target"][:, :-1], memory,
                                           memory_mask, need_weights)
        return {"logits": logits, "attn": weights, "graph_len": graph_len,
                "aux_loss": self.aux_loss()}

    def aux_loss(self):
        """Summed MoE load-balancing loss across every FFN in the model."""
        total = None
        for module in self.modules():
            value = getattr(module, "aux_loss", None)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                total = value if total is None else total + value
        return total if total is not None else torch.zeros(())

    def moe_stats(self) -> list:
        return [m.stats for m in self.modules()
                if getattr(m, "stats", None)]

    def n_parameters(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if p.requires_grad and p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total
