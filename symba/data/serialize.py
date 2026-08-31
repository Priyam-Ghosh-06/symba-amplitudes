"""S5 - typed prefix serialisation of the canonical target.

Prefix (Polish) notation removes every bracket, so the only thing a decoder has
to get right is the operator sequence and its operands. Integers are written
Lample-style as a sign marker followed by digit tokens, which keeps the
vocabulary at ~30 symbols regardless of how large a coefficient gets.

Every token also carries a *type* taken from the grammar rather than learned
(01 SS5). The type channel costs 8 embedding rows and buys constrained decoding,
a per-type error breakdown, and a measurable answer to "do the MoE experts
partition by symbol class".
"""

import sympy
from sympy import Add, Mul, Pow, Integer, Rational, Symbol

from ..config import TYPE_TO_ID
from .canonical import symbol_dimension

INT_POS, INT_NEG = "INT+", "INT-"
DIGITS = [str(d) for d in range(10)]

# Arity drives both the round-trip parser and the constrained decoder.
ARITY = {"+": 2, "*": 2, "/": 2, "^": 2, "NEG": 1}


def token_type(token: str) -> str:
    """Grammar-derived type for one target token."""
    if token in ARITY:
        return "OP"
    if token in (INT_POS, INT_NEG):
        return "OP"
    if token in DIGITS:
        return "DIGIT"
    if token.startswith("<") and token.endswith(">"):
        return "SPECIAL"
    if token == "reg_prop":
        return "REGPROP"
    if token.startswith("s_"):
        return "MANDELSTAM"
    if token.startswith("m_"):
        return "MASS"
    if token in ("e", "g"):
        return "COUPLING"
    return "STRUCT"


def type_ids(tokens) -> list:
    return [TYPE_TO_ID[token_type(t)] for t in tokens]


# --- sympy -> prefix ---------------------------------------------------------

def _int_tokens(value: int) -> list:
    marker = INT_POS if value >= 0 else INT_NEG
    return [marker] + list(str(abs(int(value))))


def _fold(op: str, parts: list) -> list:
    """Right-fold a variadic operator into nested binary applications."""
    out = parts[-1]
    for part in reversed(parts[:-1]):
        out = [op] + part + out
    return out


def to_prefix(expr) -> list:
    """Deterministic prefix serialisation of a sympy expression.

    Arguments of commutative operators are sorted by ``default_sort_key`` so
    that two structurally equal expressions always serialise identically.
    """
    expr = sympy.sympify(expr)

    if isinstance(expr, Symbol):
        return [expr.name]

    if expr.is_Integer:
        return _int_tokens(int(expr))

    if isinstance(expr, Rational):
        numerator, denominator = expr.p, expr.q
        if denominator == 1:
            return _int_tokens(numerator)
        return ["/"] + _int_tokens(numerator) + _int_tokens(denominator)

    if isinstance(expr, Add):
        return _fold("+", [to_prefix(a) for a in
                           sorted(expr.args, key=sympy.default_sort_key)])

    if isinstance(expr, Mul):
        coeff, rest = expr.as_coeff_Mul()
        parts = []
        if coeff == -1 and rest is not sympy.S.One:
            return ["NEG"] + to_prefix(rest)
        if coeff != 1:
            parts.append(to_prefix(coeff))
        factors = rest.args if rest.is_Mul else (rest,)
        parts += [to_prefix(f) for f in
                  sorted(factors, key=sympy.default_sort_key)]
        return _fold("*", parts)

    if isinstance(expr, Pow):
        return ["^"] + to_prefix(expr.base) + to_prefix(expr.exp)

    raise ValueError(f"cannot serialise {expr!r} of type {type(expr).__name__}")


def target_tokens(numerator, denominator) -> list:
    """Serialise the canonical target as ``/ N D``."""
    return ["/"] + to_prefix(numerator) + to_prefix(denominator)


# --- prefix -> sympy ---------------------------------------------------------

def from_prefix(tokens):
    """Inverse of :func:`to_prefix`. Raises on a malformed sequence.

    Used by gate G5/G13 (round-trip) and by the symbolic-equivalence metric,
    which has to turn a generated token sequence back into an expression.
    """
    pos = [0]

    def parse():
        if pos[0] >= len(tokens):
            raise ValueError("truncated prefix expression")
        token = tokens[pos[0]]
        pos[0] += 1

        if token in ARITY:
            args = [parse() for _ in range(ARITY[token])]
            if token == "+":
                return args[0] + args[1]
            if token == "*":
                return args[0] * args[1]
            if token == "/":
                return args[0] / args[1]
            if token == "^":
                return args[0] ** args[1]
            return -args[0]

        if token in (INT_POS, INT_NEG):
            digits = []
            while pos[0] < len(tokens) and tokens[pos[0]] in DIGITS:
                digits.append(tokens[pos[0]])
                pos[0] += 1
            if not digits:
                raise ValueError(f"{token} with no digits")
            value = int("".join(digits))
            return Integer(value if token == INT_POS else -value)

        if token in DIGITS:
            raise ValueError(f"bare digit {token!r} outside an integer")

        return Symbol(token)

    expr = parse()
    if pos[0] != len(tokens):
        raise ValueError(f"{len(tokens) - pos[0]} trailing tokens")
    return expr


def is_well_formed(tokens) -> bool:
    """Parse-validity metric (01 SS6.1 metric 3) - does it parse at all?"""
    try:
        from_prefix(tokens)
        return True
    except (ValueError, TypeError, RecursionError):
        return False


# --- constrained decoding ----------------------------------------------------

class PrefixState:
    """Tracks how many operands a partial prefix sequence still owes.

    A prefix expression is complete exactly when the first token has been
    consumed and the outstanding-operand count reaches zero. That single
    counter is enough to forbid every ill-formed continuation: emitting EOS
    early, emitting an operand when none is expected, or running past the end.
    """

    def __init__(self):
        self.needed = 1          # the whole expression is one operand
        self.in_integer = False  # inside a digit run, so digits stay legal
        self.started = False

    def copy(self) -> "PrefixState":
        other = PrefixState()
        other.needed = self.needed
        other.in_integer = self.in_integer
        other.started = self.started
        return other

    def advance(self, token: str) -> "PrefixState":
        state = self.copy()
        state.started = True

        if token in DIGITS:
            if not state.in_integer:
                raise ValueError("digit outside an integer")
            return state          # digits extend the current integer

        if state.in_integer:
            state.in_integer = False
            state.needed -= 1     # the integer just closed, one operand filled

        if token in ARITY:
            state.needed += ARITY[token] - 1
        elif token in (INT_POS, INT_NEG):
            # An integer is one operand, like a symbol, but it closes when the
            # digit run ends rather than immediately - the decrement happens in
            # the ``in_integer`` branch above (and in is_complete/legal, which
            # both look one step ahead).
            state.in_integer = True
        else:
            state.needed -= 1     # a symbol is a complete operand
        return state

    def is_complete(self) -> bool:
        needed = self.needed - 1 if self.in_integer else self.needed
        return self.started and needed == 0

    def legal(self, token: str, min_digits_seen: bool = True) -> bool:
        """Can ``token`` legally follow the sequence seen so far?"""
        if token in DIGITS:
            return self.in_integer
        if self.in_integer and not min_digits_seen:
            return False          # INT marker must be followed by a digit
        outstanding = self.needed - 1 if self.in_integer else self.needed
        return outstanding > 0
