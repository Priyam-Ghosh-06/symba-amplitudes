"""Canonical squared amplitude <-> typed prefix token sequence.

Prefix notation needs no brackets. Integers are written as a sign marker
followed by digit tokens (``INT+ 1 6`` is 16), so a coefficient of any size
uses the same ten digit tokens. Every token also has a grammar type
(operator, digit, mass, Mandelstam variable, ...).
"""

import sympy
from sympy import Add, Integer, Mul, Pow, Rational, Symbol

from ..config import TYPE_TO_ID

INT_POS, INT_NEG = "INT+", "INT-"
DIGITS = [str(d) for d in range(10)]
ARITY = {"+": 2, "*": 2, "/": 2, "^": 2, "NEG": 1}


def token_type(token):
    if token in ARITY or token in (INT_POS, INT_NEG):
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


def type_ids(tokens):
    return [TYPE_TO_ID[token_type(t)] for t in tokens]


# --- sympy -> prefix ---------------------------------------------------------

def _int_tokens(value):
    return [INT_POS if value >= 0 else INT_NEG] + list(str(abs(int(value))))


def _fold(op, parts):
    """Right-fold a variadic operator into nested binary applications."""
    out = parts[-1]
    for part in reversed(parts[:-1]):
        out = [op] + part + out
    return out


def to_prefix(expr):
    """Deterministic prefix form: arguments of + and * are sorted, so equal
    expressions always give the same tokens."""
    expr = sympy.sympify(expr)
    if isinstance(expr, Symbol):
        return [expr.name]
    if expr.is_Integer:
        return _int_tokens(int(expr))
    if isinstance(expr, Rational):
        return ["/"] + _int_tokens(expr.p) + _int_tokens(expr.q)
    if isinstance(expr, Add):
        return _fold("+", [to_prefix(a) for a in sorted(expr.args, key=sympy.default_sort_key)])
    if isinstance(expr, Mul):
        coeff, rest = expr.as_coeff_Mul()
        if coeff == -1 and rest is not sympy.S.One:
            return ["NEG"] + to_prefix(rest)
        parts = [to_prefix(coeff)] if coeff != 1 else []
        factors = rest.args if rest.is_Mul else (rest,)
        parts += [to_prefix(f) for f in sorted(factors, key=sympy.default_sort_key)]
        return _fold("*", parts)
    if isinstance(expr, Pow):
        return ["^"] + to_prefix(expr.base) + to_prefix(expr.exp)
    raise ValueError(f"cannot serialise {expr!r}")


def target_tokens(numerator, denominator):
    """The canonical target as ``/ numerator denominator``."""
    return ["/"] + to_prefix(numerator) + to_prefix(denominator)


# --- prefix -> sympy ---------------------------------------------------------

def from_prefix(tokens):
    """Inverse of :func:`to_prefix`. Raises on a malformed sequence."""
    pos = 0

    def parse():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("truncated prefix expression")
        token = tokens[pos]
        pos += 1
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
            start = pos
            while pos < len(tokens) and tokens[pos] in DIGITS:
                pos += 1
            if pos == start:
                raise ValueError(f"{token} with no digits")
            value = int("".join(tokens[start:pos]))
            return Integer(value if token == INT_POS else -value)
        if token in DIGITS:
            raise ValueError(f"bare digit {token!r}")
        return Symbol(token)

    expr = parse()
    if pos != len(tokens):
        raise ValueError(f"{len(tokens) - pos} trailing tokens")
    return expr


def is_well_formed(tokens):
    """Does the token sequence parse as exactly one prefix expression?"""
    state = PrefixState()
    for token in tokens:
        try:
            state = state.advance(token)
        except ValueError:
            return False
        if state.needed < 0:
            return False
    return state.is_complete()


class PrefixState:
    """How many operands a partial prefix sequence still owes.

    The sequence is complete when the count reaches zero. Tracking this one
    number is enough to forbid every ill-formed continuation, which is what the
    constrained decoder uses.
    """

    def __init__(self):
        self.needed = 1          # the whole expression is one operand
        self.in_integer = False  # inside a digit run
        self.digits = 0          # digits seen in the current run
        self.started = False

    def copy(self):
        other = PrefixState()
        other.needed, other.in_integer = self.needed, self.in_integer
        other.digits, other.started = self.digits, self.started
        return other

    def advance(self, token):
        state = self.copy()
        state.started = True
        if token in DIGITS:
            if not state.in_integer:
                raise ValueError("digit outside an integer")
            state.digits += 1
            return state
        if state.in_integer:     # the digit run just ended: one operand filled
            state.in_integer = False
            state.needed -= 1
        if token in ARITY:
            state.needed += ARITY[token] - 1
        elif token in (INT_POS, INT_NEG):
            state.in_integer = True
            state.digits = 0
        else:
            state.needed -= 1    # a symbol is a complete operand
        return state

    def is_complete(self):
        if self.in_integer and self.digits == 0:
            return False
        needed = self.needed - 1 if self.in_integer else self.needed
        return self.started and needed == 0
