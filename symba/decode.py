"""Beam search with a grammar constraint.

At every step, tokens that cannot legally continue the prefix expression are
masked before the softmax, so the decoder can only produce well-formed
expressions, and it closes every open operator before the length budget runs
out. Finished beams are ranked by length-normalised score (Wu et al., 2016).
"""

import torch
import torch.nn.functional as F

from .config import EOS, PAD, SOS
from .data.serialize import ARITY, DIGITS, INT_NEG, INT_POS, PrefixState


class ConstraintMask:
    def __init__(self, vocab):
        itos = vocab.itos
        special = lambda t: t.startswith("<") and t.endswith(">")
        self.size = len(itos)
        self.is_digit = torch.tensor([t in DIGITS for t in itos])
        self.is_int = torch.tensor([t in (INT_POS, INT_NEG) for t in itos])
        self.is_operator = torch.tensor([t in ARITY for t in itos])
        self.is_operand = torch.tensor([t not in ARITY and t not in DIGITS
                                        and t not in (INT_POS, INT_NEG) and not special(t)
                                        for t in itos])

    def allowed(self, state, digits_in_run, remaining):
        """Legal next tokens, given the prefix state and the steps left."""
        allowed = torch.zeros(self.size, dtype=torch.bool)
        if state.in_integer:
            allowed |= self.is_digit
            if digits_in_run == 0:
                return allowed               # an INT marker needs at least one digit
            outstanding = state.needed - 1   # as if the integer closed here
        else:
            outstanding = state.needed

        if outstanding > 0:
            allowed |= self.is_operand
            # An operator or an integer needs spare budget to be closed later.
            if remaining > outstanding + 1:
                allowed |= self.is_operator | self.is_int
        elif state.started:
            allowed[EOS] = True
        return allowed


def _advance(state, token, digits_in_run):
    if token in DIGITS:
        return state.advance(token), digits_in_run + 1
    return state.advance(token), 0


@torch.no_grad()
def beam_search(model, batch, vocab, beam_width=4, max_len=160, length_penalty=0.7,
                constrained=True, use_cache=True):
    """Decode every sample in ``batch``; returns one token-id list per sample."""
    model.eval()
    memory, memory_mask = model.encode(batch)
    device = memory.device
    B, W = memory.size(0), beam_width
    constraint = ConstraintMask(vocab) if constrained else None

    memory = memory.repeat_interleave(W, dim=0)
    memory_mask = memory_mask.repeat_interleave(W, dim=0)
    sequences = torch.full((B * W, 1), SOS, dtype=torch.long, device=device)
    scores = torch.full((B, W), float("-inf"), device=device)
    scores[:, 0] = 0.0
    scores = scores.reshape(-1)
    finished = torch.zeros(B * W, dtype=torch.bool, device=device)
    states = [PrefixState() for _ in range(B * W)]
    digit_runs = [0] * (B * W)
    cache = {} if use_cache else None

    for step in range(max_len - 1):
        if cache is None:
            logits, _ = model.decode_step(sequences, memory, memory_mask)
        else:
            logits, _ = model.decode_step(sequences[:, -1:], memory, memory_mask,
                                          cache=cache, offset=sequences.size(1) - 1)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

        if constrained:
            remaining = max_len - 1 - step
            mask = torch.stack([constraint.allowed(states[i], digit_runs[i], remaining)
                                for i in range(B * W)]).to(device)
            log_probs = log_probs.masked_fill(~mask, float("-inf"))

        # A finished beam can only extend with PAD, at no cost.
        log_probs[finished] = float("-inf")
        log_probs[finished, PAD] = 0.0

        V = log_probs.size(-1)
        top_scores, top_flat = (scores[:, None] + log_probs).reshape(B, W * V).topk(W, dim=-1)
        source = (torch.arange(B, device=device)[:, None] * W + top_flat // V).reshape(-1)
        tokens = (top_flat % V).reshape(-1)
        sequences = torch.cat([sequences[source], tokens[:, None]], dim=1)
        scores = top_scores.reshape(-1)
        if cache is not None:
            model.reorder_cache(cache, source)

        new_states, new_runs, new_finished = [], [], []
        for src, token_id in zip(source.tolist(), tokens.tolist()):
            if finished[src] or token_id in (EOS, PAD):
                new_states.append(states[src])
                new_runs.append(digit_runs[src])
                new_finished.append(bool(finished[src]) or token_id == EOS)
                continue
            try:
                state, run = _advance(states[src], vocab.itos[token_id], digit_runs[src])
            except ValueError:
                state, run = states[src], digit_runs[src]
            new_states.append(state)
            new_runs.append(run)
            new_finished.append(False)
        states, digit_runs = new_states, new_runs
        finished = torch.tensor(new_finished, dtype=torch.bool, device=device)
        if bool(finished.all()):
            break

    lengths = (sequences != PAD).sum(dim=1).float()
    penalty = ((5.0 + lengths) / 6.0) ** length_penalty if length_penalty else 1.0
    best = (scores / penalty).reshape(B, W).argmax(dim=-1)
    return [sequences[b * W + int(best[b])].tolist() for b in range(B)]
