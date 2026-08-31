"""Free-running decoding: greedy and beam, with length normalisation and an
optional grammar constraint.

Two things the previous decoder got wrong and that matter more than any
architecture choice:

* beams were ranked by raw cumulative log-probability, so a beam that emits EOS
  early always outranks a longer correct one - a systematic bias toward short
  output (02 SS4.1). Here scores are length-normalised with the GNMT penalty.
* nothing stopped the model emitting a sequence that is not an expression.
  ``constrained=True`` masks any token that cannot legally continue the prefix,
  which makes an unparsable output impossible (03 SS3.2).

Teacher-forced argmax is deliberately not in this module. It is not decoding and
must never be reported as exact match (02 SS1.7).
"""

import torch
import torch.nn.functional as F

from ..config import PAD, SOS, EOS
from ..data.serialize import PrefixState, ARITY, DIGITS, INT_POS, INT_NEG


class ConstraintMask:
    """Precomputes, per vocabulary, which tokens are operators/digits/operands."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.is_digit = torch.tensor(
            [t in DIGITS for t in vocab.itos], dtype=torch.bool)
        self.is_int_marker = torch.tensor(
            [t in (INT_POS, INT_NEG) for t in vocab.itos], dtype=torch.bool)
        self.is_operand = torch.tensor(
            [t not in ARITY and t not in DIGITS and t not in (INT_POS, INT_NEG)
             and not (t.startswith("<") and t.endswith(">"))
             for t in vocab.itos], dtype=torch.bool)
        self.is_operator = torch.tensor(
            [t in ARITY for t in vocab.itos], dtype=torch.bool)
        self.is_special = torch.tensor(
            [t.startswith("<") and t.endswith(">") for t in vocab.itos],
            dtype=torch.bool)

    def allowed(self, state: PrefixState, digits_in_run: int, device,
                remaining: int = None):
        """Boolean vector over the vocabulary of legal next tokens.

        ``remaining`` is the number of steps left in the length budget. Every
        outstanding operand needs at least one token to close, and an operator
        adds to that debt, so once the debt equals the budget only operands stay
        legal. Without this the decoder can be perfectly well-formed at every
        step and still hit the length limit mid-expression, emitting a prefix
        that does not parse.
        """
        allowed = torch.zeros(len(self.vocab.itos), dtype=torch.bool,
                              device=device)

        if state.in_integer:
            allowed |= self.is_digit.to(device)
            if digits_in_run == 0:
                return allowed            # an INT marker must be followed by a digit
            # A digit run may also end here, so fall through to the normal case
            # using the operand count that closing the integer would leave.
            outstanding = state.needed - 1
        else:
            outstanding = state.needed

        if outstanding > 0:
            allowed |= self.is_operand.to(device)
            # An operator raises the debt by (arity - 1); an integer costs two
            # tokens to close one operand. Both need slack in the budget.
            if remaining is None or remaining > outstanding + 1:
                allowed |= self.is_operator.to(device)
            if remaining is None or remaining > outstanding + 1:
                allowed |= self.is_int_marker.to(device)
            if not bool(allowed.any()):
                allowed |= self.is_operand.to(device)
        elif state.started:
            allowed[EOS] = True

        return allowed


def _advance(state, token, digits_in_run):
    """Apply one token to a prefix state, tracking the digit run length."""
    if token in DIGITS:
        return state.advance(token), digits_in_run + 1
    return state.advance(token), 0


@torch.no_grad()
def beam_search(model, batch, vocab, beam_width=4, max_len=160,
                length_penalty=0.7, constrained=True, constraint=None,
                use_cache=True):
    """Batched beam search. Returns a list of token-id lists, one per sample.

    Beams live in the batch dimension, so one decoder call advances every beam
    of every sample at once. The encoder runs once for the whole batch.
    """
    model.eval()
    device = batch["graph"].device
    memory, memory_mask, _ = model.encode(batch)
    B = memory.size(0)
    W = beam_width

    if constrained and constraint is None:
        constraint = ConstraintMask(vocab)

    memory = memory.repeat_interleave(W, dim=0)
    memory_mask = memory_mask.repeat_interleave(W, dim=0)

    sequences = torch.full((B * W, 1), SOS, dtype=torch.long, device=device)
    scores = torch.full((B, W), float("-inf"), device=device)
    scores[:, 0] = 0.0
    scores = scores.reshape(-1)
    finished = torch.zeros(B * W, dtype=torch.bool, device=device)

    states = [PrefixState() for _ in range(B * W)]
    digit_runs = [0] * (B * W)

    # Incremental decoding: each step feeds only the newest token and reuses the
    # cached self-attention K/V and the cross-attention projection of the fixed
    # memory. Re-running the whole prefix every step is O(W*T^2) (02 SS4.2).
    # ``use_cache=False`` keeps the O(T^2) recompute path, which exists so the
    # gate test can assert the two produce the same sequences.
    cache = model.new_cache() if use_cache else None

    for step in range(max_len - 1):
        if cache is None:
            logits, _ = model.decode_step(sequences, memory, memory_mask)
        else:
            logits, _ = model.decode_step(
                sequences[:, -1:], memory, memory_mask,
                cache=cache, offset=sequences.size(1) - 1)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

        if constrained:
            remaining = max_len - 1 - step
            mask = torch.stack([
                constraint.allowed(states[i], digit_runs[i], device, remaining)
                for i in range(B * W)])
            log_probs = log_probs.masked_fill(~mask, float("-inf"))

        # A finished beam only ever extends with PAD, at no score cost, so it
        # competes against live beams on its final normalised score.
        log_probs[finished] = float("-inf")
        log_probs[finished, PAD] = 0.0

        candidate = scores.unsqueeze(1) + log_probs
        candidate = candidate.reshape(B, W * log_probs.size(-1))
        top_scores, top_flat = candidate.topk(W, dim=-1)

        beam_index = top_flat // log_probs.size(-1)
        token_index = top_flat % log_probs.size(-1)

        global_beam = (torch.arange(B, device=device).unsqueeze(1) * W
                       + beam_index).reshape(-1)
        sequences = torch.cat(
            [sequences[global_beam], token_index.reshape(-1, 1)], dim=1)
        scores = top_scores.reshape(-1)
        if cache is not None:
            model.reorder_cache(cache, global_beam)

        new_states, new_runs, new_finished = [], [], []
        for i, (src, token_id) in enumerate(zip(global_beam.tolist(),
                                                token_index.reshape(-1).tolist())):
            was_finished = bool(finished[src])
            token = vocab.itos[token_id]
            if was_finished or token_id in (EOS, PAD):
                new_states.append(states[src])
                new_runs.append(digit_runs[src])
                new_finished.append(True if token_id == EOS or was_finished
                                    else False)
            else:
                try:
                    state, run = _advance(states[src], token, digit_runs[src])
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
    normalised = (scores / penalty).reshape(B, W)
    best = normalised.argmax(dim=-1)

    outputs = []
    for b in range(B):
        row = sequences[b * W + int(best[b])].tolist()
        outputs.append(row)
    return outputs


@torch.no_grad()
def greedy(model, batch, vocab, max_len=160, constrained=True, constraint=None):
    """Beam search with width 1 - free-running, not teacher-forced."""
    return beam_search(model, batch, vocab, beam_width=1, max_len=max_len,
                       length_penalty=0.0, constrained=constrained,
                       constraint=constraint)
