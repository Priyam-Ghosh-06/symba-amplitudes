"""S3 - canonicalise the target, and check the mass-dimension invariant.

MARTY emits nested, unsimplified output. ``expand -> together -> cancel`` takes
the QCD target from a 2872-character tail down to 258 and merges 19 raw strings
that were algebraically identical to another raw string (01 SS0.4). Both forms are
kept: the raw string is what makes a number comparable to SYMBA, the canonical
form is what the model is trained on.
"""

import re

import sympy
from sympy import Symbol, expand, cancel, together, fraction, factor
from sympy.parsing.sympy_parser import parse_expr

from ..config import MASS_DIMENSION

# ``i`` is always the imaginary unit in this corpus, never an index (01 SS0 and
# the n-gram tests in the tokenisation notebook).
_IMAGINARY_RE = re.compile(r"(?<![A-Za-z_0-9])i(?![A-Za-z_0-9])")
_CARET_RE = re.compile(r"\^")

_MANDELSTAM_RE = re.compile(r"^s_\d+$")
_MASS_RE = re.compile(r"^m_")
_COUPLINGS = {"e", "g"}


def symbol_dimension(name: str) -> int:
    """Mass dimension of one free symbol."""
    if name == "reg_prop":
        return MASS_DIMENSION["reg_prop"]
    if _MANDELSTAM_RE.match(name):
        return MASS_DIMENSION["mandelstam"]
    if _MASS_RE.match(name):
        return MASS_DIMENSION["mass"]
    if name in _COUPLINGS:
        return MASS_DIMENSION["coupling"]
    raise ValueError(f"unknown symbol {name!r} - no mass dimension defined")


def to_sympy(expression: str):
    """Parse a MARTY expression string into a sympy expression."""
    text = _CARET_RE.sub("**", expression)
    text = _IMAGINARY_RE.sub("I", text)
    return parse_expr(text, evaluate=True)


def canonicalise(sq_amp: str):
    """Return ``(numerator, denominator)`` in canonical form.

    ``numerator`` is fully expanded (at most 5 monomials in QED, 8 in QCD);
    ``denominator`` is factored (under 37 characters in both corpora).
    """
    e = expand(to_sympy(sq_amp))
    n, d = fraction(cancel(together(e)))
    return expand(n), factor(d)


def monomial_dimensions(numerator) -> list:
    """Mass dimension of every monomial in an expanded numerator."""
    terms = numerator.as_ordered_terms() if numerator.is_Add else [numerator]

    dims = []
    for term in terms:
        total = 0
        _coeff, rest = term.as_coeff_Mul()
        factors = rest.as_ordered_factors() if rest.is_Mul else [rest]
        for f in factors:
            base, exponent = f.as_base_exp()
            if base.is_Number:
                continue
            if not isinstance(base, Symbol):
                raise ValueError(f"unexpected factor {f!r} in {term!r}")
            if not exponent.is_Integer:
                raise ValueError(f"non-integer exponent in {term!r}")
            total += symbol_dimension(base.name) * int(exponent)
        dims.append(total)
    return dims


def check_mass_dimension(numerator, expected: int = 4) -> bool:
    """Gate G4: the expanded numerator is homogeneous of mass dimension 4.

    Measured exact in 360/360 QED and 234/234 QCD records. A violation means
    either a new process class or a bug in the canonicalisation, and both are
    worth stopping for.
    """
    dims = monomial_dimensions(numerator)
    return bool(dims) and all(d == expected for d in dims)


def verify_equivalence(numerator, denominator, raw: str) -> bool:
    """Gate G3: the canonical form denotes the same function as the raw string."""
    return sympy.simplify(numerator / denominator - to_sympy(raw)) == 0


def canonicalise_record(record, verify: bool = False):
    """Attach ``canon_num`` / ``canon_den`` and the dimension flag to a Record."""
    numerator, denominator = canonicalise(record["sq_amp_std"])
    record["canon_num"] = numerator
    record["canon_den"] = denominator
    record["canon_expr"] = numerator / denominator
    record["mass_dim_ok"] = check_mass_dimension(numerator)

    if not record["mass_dim_ok"]:
        raise AssertionError(
            f"{record.source_file}:{record.line_no} numerator is not "
            f"mass-dimension-4 homogeneous: {monomial_dimensions(numerator)}")

    if verify and not verify_equivalence(numerator, denominator,
                                         record["sq_amp_std"]):
        raise AssertionError(
            f"{record.source_file}:{record.line_no} canonical form is not "
            f"symbolically equal to the raw target")

    return record


# --- template classes (01 SS0.6) ---------------------------------------------

def leg_role_form(expr, mass_order: list):
    """Substitute each mass symbol for its leg role, ``m_e -> M1``.

    ``mass_order`` is the order the masses first appear in the record's legs, so
    two records that differ only in flavour map to the same expression. This is
    what collapses 360 QED records to 30 distinct functions.
    """
    subs = {Symbol(name): Symbol(f"M{i + 1}")
            for i, name in enumerate(mass_order)}
    return sympy.simplify(expr.subs(subs, simultaneous=True))


def template_key(expr, mass_order: list) -> str:
    """A hashable identifier for the template class of one record."""
    return sympy.srepr(sympy.cancel(leg_role_form(expr, mass_order)))
