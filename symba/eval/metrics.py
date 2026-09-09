"""The metric suite of 01 SS6.1, each reported with n and a Wilson interval.

A single opaque percentage is replaced by a decomposition: did it produce a
well-formed expression at all, is the expression dimensionally consistent, did
it pick the right propagator channel, did it get the monomial structure but the
wrong coefficient. That decomposition is what turns "66.7%" into a diagnosis.
"""

import math
from collections import defaultdict

import sympy

from ..data.canonical import check_mass_dimension
from ..data.serialize import (from_prefix, is_well_formed,
                              operator_count)


def wilson(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval. Correct at small n, unlike the normal one."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denominator = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarise(flags: list) -> dict:
    successes = int(sum(flags))
    p, lo, hi = wilson(successes, len(flags))
    return {"value": p, "ci_low": lo, "ci_high": hi,
            "n": len(flags), "successes": successes}


_EXPR_CACHE = {}
_FRACTION_CACHE = {}

# An untrained or half-trained decoder emits deeply nested rationals such as
# ``/ / / / ... m_b m_b ...``. Those are perfectly well-formed, so the parser
# accepts them, but ``together``/``expand`` on them is combinatorial and can run
# for minutes on a single sample. A reference target is ~40 operations; anything
# an order of magnitude past that cannot be equal to one, so it is scored wrong
# without being expanded. Bail-outs are counted and reported rather than hidden.
COMPLEXITY_CAP = 600
# A prediction with more than this multiple of the reference's operator count
# is scored wrong without being parsed.
OPERATOR_SLACK = 3
_bailouts = {"count": 0}


def bailout_count() -> int:
    """How many predictions were rejected on complexity rather than compared."""
    return _bailouts["count"]



# Upper bound on the monomial count of expand(expr) that is still worth
# computing. Reference numerators carry at most 8 monomials, so this is three
# orders of magnitude of slack for a prediction that could still be right.
TERM_CAP = 2000


def expansion_terms(expr, cap: int = TERM_CAP) -> int:
    """Upper bound on the number of monomials in ``expand(expr)``.

    ``count_ops`` bounds the size of the *written* expression, which does not
    bound the cost of expanding it: a product of k binomials has ``count_ops``
    11 for every k while its expansion grows as 2**k, and OPERATOR_SLACK
    permits roughly k = 60. So the old cap never bound the quantity that
    actually blows up.

    One pass over the tree, short-circuited as soon as the bound passes ``cap``,
    so the pathological case is the cheap case. A negative integer power is a
    single factor - ``expand`` does not distribute over 1/(a + b) - so it
    counts as one term rather than recursing into the base.
    """
    if expr.is_Atom:
        return 1
    if expr.is_Add:
        total = 0
        for arg in expr.args:
            total += expansion_terms(arg, cap)
            if total > cap:
                return total
        return total
    if expr.is_Mul:
        total = 1
        for arg in expr.args:
            total *= expansion_terms(arg, cap)
            if total > cap:
                return total
        return total
    if expr.is_Pow:
        base, exponent = expr.as_base_exp()
        if not exponent.is_Integer:
            return cap + 1
        power = int(exponent)
        if power < 0:
            return 1
        base_terms = expansion_terms(base, cap)
        if base_terms <= 1:
            return 1
        total = 1
        for _ in range(power):
            total *= base_terms
            if total > cap:
                return total
        return total
    return 1


def _too_complex(expr) -> bool:
    try:
        if sympy.count_ops(expr) > COMPLEXITY_CAP:
            _bailouts["count"] += 1
            return True
        if expansion_terms(expr) > TERM_CAP:
            _bailouts["count"] += 1
            return True
    except Exception:
        _bailouts["count"] += 1
        return True
    return False


def _safe_expr(tokens):
    """Parse a token list to sympy, memoised - references repeat every eval."""
    key = tuple(tokens)
    if key in _EXPR_CACHE:
        return _EXPR_CACHE[key]
    try:
        expr = from_prefix(tokens)
    except Exception:
        expr = None
    if len(_EXPR_CACHE) < 20_000:
        _EXPR_CACHE[key] = expr
    return expr


def clear_caches():
    """Drop the memo tables and sympy's global cache.

    sympy caches aggressively by design. Scoring thousands of distinct junk
    expressions per run grows that cache without bound, which is what took the
    first QED job from 733 MB to 3.7 GB.
    """
    _EXPR_CACHE.clear()
    _FRACTION_CACHE.clear()
    sympy.core.cache.clear_cache()


def _as_fraction(expr):
    """``(expanded numerator, denominator)`` for a rational expression."""
    if expr is None or _too_complex(expr):
        return (None, None)
    key = sympy.srepr(expr)
    if key in _FRACTION_CACHE:
        return _FRACTION_CACHE[key]
    try:
        # ``cancel`` matters, and its absence was a real defect. The reference
        # is built by canonical.py as expand -> together -> cancel; a
        # prediction was only put through ``together``. A prediction that is
        # algebraically equal but not in lowest terms - say
        # (s_12^2 - s_13^2) / ((s_12 - s_13) * s_14) against
        # (s_12 + s_13) / s_14 - then kept a denominator the reference does
        # not have, and scored ``equal`` while failing the channel test. That
        # broke the identity symbolic EM = structure x coefficient downward.
        # Normalising both sides the same way is the fix.
        numerator, denominator = sympy.fraction(
            sympy.cancel(sympy.together(expr)))
        result = (sympy.expand(numerator), sympy.expand(denominator))
    except Exception:
        result = (None, None)
    if len(_FRACTION_CACHE) < 20_000:
        _FRACTION_CACHE[key] = result
    return result


def symbolically_equal(a, b) -> bool:
    """Exact equality of two rational functions, by cross-multiplication.

    ``simplify(a - b) == 0`` is correct but costs seconds per call, and this
    runs on every validation sample at every evaluation. For a ratio of
    polynomials, ``Na*Db - Nb*Da == 0`` after expansion is exact and about two
    orders of magnitude cheaper.
    """
    if a is None or b is None:
        return False
    if a == b:
        return True
    na, da = _as_fraction(a)
    nb, db = _as_fraction(b)
    if na is None or nb is None:
        return False
    try:
        return sympy.expand(na * db - nb * da) == 0
    except Exception:
        return False


def _denominator_of(expr):
    return _as_fraction(expr)[1]


def same_channel(pred_expr, ref_expr) -> bool:
    """Do the two denominators describe the same propagator channel?

    Projective, not literal: a channel is a set of poles, so denominators that
    differ by a non-zero scalar are the same channel. ``2*s_14`` and ``s_14``
    are one channel written two ways, and a strict ``expand(Dp - Dr) == 0``
    called that a miss.
    """
    dp, dr = _denominator_of(pred_expr), _denominator_of(ref_expr)
    if dp is None or dr is None:
        return False
    try:
        if sympy.expand(dp - dr) == 0:
            return True
        ratio = sympy.cancel(dp / dr)
        return bool(ratio.is_number) and ratio != 0
    except Exception:
        return False


def _monomial_set(expr):
    numerator, _d = _as_fraction(expr)
    if numerator is None:
        return None
    terms = numerator.as_ordered_terms() if numerator.is_Add else [numerator]
    return {str(t.as_coeff_Mul()[1]) for t in terms}


def evaluate_predictions(predictions, references, templates=None) -> dict:
    """Score a batch of decoded token lists against their ground truth.

    ``predictions`` and ``references`` are lists of token lists (specials
    already stripped).
    """
    raw_em, symbolic_em, well_formed, dim_ok = [], [], [], []
    channel_ok, monomial_f1 = [], []
    structure_ok, coeff_given_structure = [], []
    per_class = defaultdict(list)
    bailouts_before = bailout_count()

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        raw_em.append(pred == ref)

        # Well-formedness is decided syntactically, with no sympy involved.
        well_formed.append(is_well_formed(pred))

        # Only build a sympy object for a prediction that could plausibly equal
        # the reference. A sequence several times more complex than the target
        # cannot be equal to it, and parsing it is where the time goes.
        budget = OPERATOR_SLACK * max(1, operator_count(ref))
        if not well_formed[-1] or operator_count(pred) > budget:
            if well_formed[-1]:
                _bailouts["count"] += 1
            pred_expr = None
        else:
            pred_expr = _safe_expr(pred)

        ref_expr = _safe_expr(ref)
        if pred_expr is None or ref_expr is None:
            symbolic_em.append(False)
            dim_ok.append(False)
            channel_ok.append(False)
            monomial_f1.append(0.0)
            structure_ok.append(False)
        else:
            equal = symbolically_equal(pred_expr, ref_expr)
            symbolic_em.append(bool(equal))

            numerator, _d = _as_fraction(pred_expr)
            try:
                dim_ok.append(numerator is not None
                              and check_mass_dimension(numerator))
            except Exception:
                dim_ok.append(False)

            channel_ok.append(same_channel(pred_expr, ref_expr))

            pred_mono = _monomial_set(pred_expr) or set()
            ref_mono = _monomial_set(ref_expr) or set()
            if pred_mono or ref_mono:
                overlap = len(pred_mono & ref_mono)
                precision = overlap / len(pred_mono) if pred_mono else 0.0
                recall = overlap / len(ref_mono) if ref_mono else 0.0
                f1 = (2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
            else:
                f1 = 1.0
            monomial_f1.append(f1)

            # The decomposition of 01 SS6.1 metric 6: "wrong structure" and
            # "wrong number" are different failures and must be counted
            # separately. Structure is the denominator channel plus the
            # monomial support of the numerator; given both, the only thing
            # left to get wrong is the coefficients, so the conditional rate
            # is exactly the coefficient accuracy.
            #
            # The previous form was ``equal and pred_mono == ref_mono``, which
            # is implied by ``equal`` and so reproduced symbolic exact match
            # in all 46 archived runs - it measured nothing.
            same_structure = bool(channel_ok[-1]) and pred_mono == ref_mono
            structure_ok.append(same_structure)
            if same_structure:
                coeff_given_structure.append(bool(equal))

        if templates is not None:
            per_class[templates[i]].append(bool(symbolic_em[-1]))

    result = {
        # Token-sequence equality against the canonical target. This is the
        # direct analogue of SYMBA's sequence accuracy, but on the canonical
        # representation, NOT on MARTY's raw output string - the two are not
        # numerically comparable and the old name ("raw_exact_match") claimed
        # they were. Because to_prefix is a normal form, this is equal to
        # symbolic EM on canonical targets by construction, and it was in all
        # 46 archived runs; it separates only under data.target="raw".
        "sequence_exact_match": summarise(raw_em),
        "symbolic_exact_match": summarise(symbolic_em),
        "parse_validity": summarise(well_formed),
        "mass_dimension_validity": summarise(dim_ok),
        "channel_accuracy": summarise(channel_ok),
        # Fraction whose denominator channel AND numerator monomial support
        # are both right - the "did it find the functional form" half.
        "structure_exact": summarise(structure_ok),
        # Conditional on the structure being right: did it get the numbers
        # right too. n is the number of structurally correct predictions, so
        # symbolic EM = structure_exact x coefficient_exact.
        "coefficient_exact": summarise(coeff_given_structure),
        "monomial_f1": {"value": (sum(monomial_f1) / len(monomial_f1)
                                  if monomial_f1 else 0.0),
                        "n": len(monomial_f1)},
        # Predictions too complex to be compared, scored wrong. High early in
        # training, and should fall to zero as the model learns to close an
        # expression instead of nesting operators.
        "complexity_bailouts": bailout_count() - bailouts_before,
        # Per-record outcomes, in split order. Arms at the same seed score the
        # identical test records in the identical order, so keeping this makes
        # a paired test (McNemar) and error-overlap analysis available on runs
        # that already exist, instead of needing a rerun to ask a question
        # nobody thought of at run time. It is a few dozen booleans.
        "per_record": {
            "symbolic": [bool(x) for x in symbolic_em],
            "structure": [bool(x) for x in structure_ok],
            "channel": [bool(x) for x in channel_ok],
            "template": list(templates) if templates is not None else None,
        },
    }

    if per_class:
        result["per_template"] = {
            key[:32]: summarise(flags) for key, flags in sorted(per_class.items())
        }
        class_means = [sum(f) / len(f) for f in per_class.values()]
        result["macro_template_accuracy"] = {
            "value": sum(class_means) / len(class_means),
            "n_classes": len(class_means),
        }

    return result


def format_report(name: str, metrics: dict) -> str:
    """One-line-per-metric text block with n and CI, for the run log."""
    lines = [f"  {name}"]
    for key in ("symbolic_exact_match", "sequence_exact_match", "parse_validity",
                "mass_dimension_validity", "channel_accuracy",
                "structure_exact", "coefficient_exact"):
        if key not in metrics:
            continue
        m = metrics[key]
        lines.append(f"    {key:<26} {m['value'] * 100:6.2f}%  "
                     f"[{m['ci_low'] * 100:5.1f}, {m['ci_high'] * 100:5.1f}]  "
                     f"n={m['n']}")
    if "monomial_f1" in metrics:
        lines.append(f"    {'monomial_f1':<26} "
                     f"{metrics['monomial_f1']['value'] * 100:6.2f}%")
    if "macro_template_accuracy" in metrics:
        m = metrics["macro_template_accuracy"]
        lines.append(f"    {'macro_template_accuracy':<26} "
                     f"{m['value'] * 100:6.2f}%  classes={m['n_classes']}")
    return "\n".join(lines)
