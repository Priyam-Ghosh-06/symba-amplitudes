"""Feed-forward layers: a dense MLP, or a mixture of experts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(d_model, dim_feedforward, dropout):
    return nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))


class MoEFFN(nn.Module):
    """Noisy top-k mixture of experts with a load-balancing loss (Shazeer et al., 2017).

    A router scores every token against every expert; each token is processed
    by its top-k experts and their outputs are mixed by the router weights.
    """

    def __init__(self, d_model, dim_feedforward, n_experts, top_k, dropout, aux_weight):
        super().__init__()
        self.n_experts, self.top_k, self.aux_weight = n_experts, top_k, aux_weight
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(_mlp(d_model, dim_feedforward, dropout)
                                     for _ in range(n_experts))
        self.aux_loss = torch.zeros(())
        self.last_expert = None          # top-1 expert of every token, for analysis

    def forward(self, x):
        B, L, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.router(flat)
        if self.training:
            logits = logits + torch.randn_like(logits) / self.n_experts
        probs = logits.softmax(-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)
        gates = top_probs / top_probs.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token, slot = (top_idx == e).nonzero(as_tuple=True)
            if token.numel():
                out.index_add_(0, token, expert(flat[token]) * gates[token, slot, None])

        # Importance x load: penalises the router for favouring a few experts.
        importance = probs.mean(0)
        load = F.one_hot(top_idx, self.n_experts).float().sum(1).mean(0)
        self.aux_loss = self.aux_weight * self.n_experts * (importance * load).sum()
        self.last_expert = top_idx[:, 0].detach()
        return out.reshape(B, L, D)


def build_ffn(cfg):
    if cfg.ffn == "dense":
        return _mlp(cfg.d_model, cfg.dim_feedforward, cfg.dropout)
    if cfg.ffn == "moe":
        return MoEFFN(cfg.d_model, cfg.dim_feedforward, cfg.n_experts, cfg.top_k,
                      cfg.dropout, cfg.moe_aux_weight)
    raise ValueError(f"unknown ffn {cfg.ffn!r}")
