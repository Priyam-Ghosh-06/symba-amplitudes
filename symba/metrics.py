"""Is a predicted squared amplitude correct?"""

import random
from fractions import Fraction

from .data.serialize import ARITY, DIGITS, INT_NEG, INT_POS, is_well_formed


def _value(tokens, point, rng):
    """Exact value of a prefix expression, each symbol set from ``point``."""
    pos = 0

    def parse():
        nonlocal pos
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
                if args[1].denominator != 1 or abs(args[1]) > 16:
                    raise ValueError("exponent out of range")
                return args[0] ** int(args[1])
            return -args[0]
        if token in (INT_POS, INT_NEG):
            start = pos
            while pos < len(tokens) and tokens[pos] in DIGITS:
                pos += 1
            value = int("".join(tokens[start:pos]))
            return Fraction(value if token == INT_POS else -value)
        if token not in point:
            point[token] = Fraction(rng.randint(1, 10**6), rng.randint(1, 10**6))
        return point[token]

    return parse()


def symbolically_equal(prediction, reference, trials=2):
    """Do two prefix token lists denote the same rational function?

    Both are evaluated exactly, in rational arithmetic, at random points. Two
    different rational functions agree at a random point with vanishing
    probability (Schwartz-Zippel lemma), and unlike symbolic simplification
    this takes microseconds however wrong the prediction is.
    """
    if prediction == reference:
        return True
    if not is_well_formed(prediction) or len(prediction) > 2 * len(reference):
        return False
    rng = random.Random(0)
    try:
        for _ in range(trials):
            point = {}
            if _value(prediction, point, rng) != _value(reference, point, rng):
                return False
        return True
    except (ZeroDivisionError, ValueError):
        return False
