"""Canonical form of the squared amplitude.

MARTY writes the squared amplitude unexpanded and uncancelled, so one function
can appear as many different strings, some thousands of characters long. The
target is put over a common denominator, cancelled and expanded with sympy,
which makes it short and makes equal answers identical.
"""

import re

from sympy import cancel, expand, factor, fraction, together
from sympy.parsing.sympy_parser import parse_expr

_IMAGINARY = re.compile(r"(?<![A-Za-z_0-9])i(?![A-Za-z_0-9])")


def to_sympy(expression):
    """Parse a MARTY expression string (``^`` is power, ``i`` the imaginary unit)."""
    return parse_expr(_IMAGINARY.sub("I", expression.replace("^", "**")))


def canonicalise(sq_amp):
    """``(numerator, denominator)``: numerator expanded, denominator factored."""
    numerator, denominator = fraction(cancel(together(expand(to_sympy(sq_amp)))))
    return expand(numerator), factor(denominator)
