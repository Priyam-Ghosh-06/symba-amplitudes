"""Feed-forward: a dense MLP, or an instrumented Mixture of Experts.

The MoE here is testable in a way the previous one was not. It has the standard
importance+load auxiliary loss, it reports per-expert utilisation and routing
entropy, and dispatch is one grouped pass per expert instead of
``top_k x n_experts`` masked passes (02 SS3.3, SS3.4).

Honest prior from 03 SS4.2: at 324 examples and 30 target functions, capacity is
not the binding constraint, so MoE is expected to move the number by about
nothing. It is here as a hypothesis with a measurement attached, not a feature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseFFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.aux_loss = torch.tensor(0.0)
        self.stats = {}

    def forward(self, x, type_ids=None):
        self.aux_loss = x.new_zeros(())
        return self.net(x)


class MoEFFN(nn.Module):
    """Noisy top-k routing with load balancing and utilisation logging."""

    def __init__(self, d_model, dim_feedforward, n_experts=4, top_k=2,
                 dropout=0.1, aux_weight=0.01):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.aux_weight = aux_weight

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])
        self.aux_loss = torch.tensor(0.0)
        self.stats = {}

    def forward(self, x, type_ids=None):
        B, L, D = x.shape
        flat = x.reshape(B * L, D)

        logits = self.router(flat)
        if self.training:
            logits = logits + torch.randn_like(logits) / self.n_experts

        # Softmax over ALL experts first, then renormalise the selected subset,
        # so unselected experts still receive router gradient (GShard/Switch
        # convention; the previous code softmaxed the top-k only).
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)
        gates = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # One grouped pass per expert: sort tokens by assignment, slice, apply.
        output = torch.zeros_like(flat)
        flat_expert = top_idx.reshape(-1)                    # (N * top_k,)
        flat_gate = gates.reshape(-1, 1)
        flat_token = (torch.arange(flat.size(0), device=x.device)
                      .repeat_interleave(self.top_k))

        order = torch.argsort(flat_expert)
        sorted_expert = flat_expert[order]
        boundaries = torch.searchsorted(
            sorted_expert,
            torch.arange(self.n_experts + 1, device=x.device))

        for e in range(self.n_experts):
            lo, hi = int(boundaries[e]), int(boundaries[e + 1])
            if lo == hi:
                continue
            slots = order[lo:hi]
            tokens = flat_token[slots]
            out_e = self.experts[e](flat[tokens]) * flat_gate[slots]
            output.index_add_(0, tokens, out_e)

        self.aux_loss = self._balance_loss(probs, top_idx)
        self._record_stats(probs, top_idx, type_ids, B, L)
        return output.reshape(B, L, D)

    def _balance_loss(self, probs, top_idx):
        """Importance + load auxiliary loss (Shazeer et al.).

        Without it there is nothing penalising router collapse, and the claim
        that experts specialise cannot be made either way.
        """
        importance = probs.mean(dim=0)
        one_hot = F.one_hot(top_idx, self.n_experts).float().sum(dim=1)
        load = one_hot.mean(dim=0)
        return self.aux_weight * self.n_experts * (importance * load).sum()

    @torch.no_grad()
    def _record_stats(self, probs, top_idx, type_ids, B, L):
        counts = torch.bincount(top_idx.reshape(-1),
                                minlength=self.n_experts).float()
        utilisation = counts / counts.sum().clamp(min=1)
        entropy = -(probs.clamp(min=1e-9).log() * probs).sum(-1).mean()

        self.stats = {
            "utilisation": utilisation.tolist(),
            "routing_entropy": float(entropy),
            "max_utilisation": float(utilisation.max()),
        }

        if type_ids is not None:
            # I(token type ; expert) - the direct measurement of Claim 3.
            types = type_ids.reshape(-1)
            top1 = top_idx[:, 0]
            n_types = int(types.max()) + 1
            joint = torch.zeros(n_types, self.n_experts, device=probs.device)
            joint.index_put_((types, top1),
                             torch.ones_like(types, dtype=torch.float),
                             accumulate=True)
            joint = joint / joint.sum().clamp(min=1)
            p_type = joint.sum(1, keepdim=True)
            p_expert = joint.sum(0, keepdim=True)
            nonzero = joint > 0
            mutual = (joint[nonzero]
                      * (joint[nonzero]
                         / (p_type.expand_as(joint)[nonzero]
                            * p_expert.expand_as(joint)[nonzero])).log()).sum()
            self.stats["mi_type_expert"] = float(mutual)


def build_ffn(cfg, d_model=None):
    d_model = d_model or cfg.d_model
    if cfg.ffn == "dense":
        return DenseFFN(d_model, cfg.dim_feedforward, cfg.dropout)
    if cfg.ffn == "moe":
        return MoEFFN(d_model, cfg.dim_feedforward, cfg.n_experts, cfg.top_k,
                      cfg.dropout, cfg.moe_aux_weight)
    raise ValueError(f"unknown ffn {cfg.ffn!r}")
