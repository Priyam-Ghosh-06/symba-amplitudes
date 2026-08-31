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


def reset_bailouts():
    _bailouts["count"] = 0


def _too_complex(expr) -> bool:
    try:
        if sympy.count_ops(expr) > COMPLEXITY_CAP:
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
        numerator, denominator = sympy.fraction(sympy.together(expr))
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
    if _too_complex(a):
        return False
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
    channel_ok, monomial_f1, coeff_ok = [], [], []
    per_class = defaultdict(list)
    bailouts_before = bailout_count()

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        raw_match = pred == ref
        raw_em.append(raw_match)

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
            coeff_ok.append(False)
        else:
            equal = symbolically_equal(pred_expr, ref_expr)
            symbolic_em.append(bool(equal))

            numerator, _d = _as_fraction(pred_expr)
            try:
                dim_ok.append(numerator is not None
                              and check_mass_dimension(numerator))
            except Exception:
                dim_ok.append(False)

            channel_ok.append(
                sympy.expand(_denominator_of(pred_expr)
                             - _denominator_of(ref_expr)) == 0
                if _denominator_of(pred_expr) is not None
                and _denominator_of(ref_expr) is not None else False)

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
            coeff_ok.append(bool(equal) and pred_mono == ref_mono)

        if templates is not None:
            per_class[templates[i]].append(bool(symbolic_em[-1]))

    result = {
        "raw_exact_match": summarise(raw_em),
        "symbolic_exact_match": summarise(symbolic_em),
        "parse_validity": summarise(well_formed),
        "mass_dimension_validity": summarise(dim_ok),
        "channel_accuracy": summarise(channel_ok),
        "coefficient_exact": summarise(coeff_ok),
        "monomial_f1": {"value": (sum(monomial_f1) / len(monomial_f1)
                                  if monomial_f1 else 0.0),
                        "n": len(monomial_f1)},
        # Predictions too complex to be compared, scored wrong. High early in
        # training, and should fall to zero as the model learns to close an
        # expression instead of nesting operators.
        "complexity_bailouts": bailout_count() - bailouts_before,
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
    for key in ("symbolic_exact_match", "raw_exact_match", "parse_validity",
                "mass_dimension_validity", "channel_accuracy",
                "coefficient_exact"):
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
