r"""S2 - normalisation.

Dummy indices such as ``%\sigma_18923`` carry no information beyond which other
occurrences they match, so they are renumbered by order of first appearance.
Physical symbols (``s_12``, ``m_e``) have no ``%`` prefix and are untouched.
"""

import re

_DUMMY_INDEX_RE = re.compile(r"(%[a-zA-Z\\]+_)(\d+)")
_BACKSLASH_RE = re.compile(r"(?<!%)\\")


def standardize(expression: str) -> str:
    r"""``%\sigma_18923 -> %\sigma_d1``, numbered by first occurrence.

    Idempotent: the pattern only matches a numeric suffix, and the output
    suffix is ``d<N>``, so re-running is a no-op (gate G2).
    """
    mapping: dict = {}
    counter = [1]

    def replacer(match):
        prefix, index_str = match.group(1), match.group(2)
        if index_str not in mapping:
            mapping[index_str] = f"d{counter[0]}"
            counter[0] += 1
        return f"{prefix}{mapping[index_str]}"

    return _DUMMY_INDEX_RE.sub(replacer, expression)


def strip_keywords(text: str) -> str:
    """Remove structural English from interaction/vertex strings, keep numbers."""
    text = (text.replace("Interaction:", "")
                .replace("Vertex ", "")
                .replace("Vertex", ""))
    return " ".join(text.split())


def assert_backslash_invariant(text: str, where: str = ""):
    r"""Every ``\`` in the corpus is an index marker preceded by ``%``.

    Measured true for 5712/5712 occurrences (01 §S2). Re-checked at load so a
    new dataset that breaks the assumption fails loudly instead of quietly
    producing a different tokenisation.
    """
    match = _BACKSLASH_RE.search(text)
    if match:
        start = max(0, match.start() - 20)
        raise AssertionError(
            f"unescaped backslash at {where or 'input'} "
            f"pos {match.start()}: ...{text[start:match.start() + 20]!r}")


def normalize_record(record):
    """Populate the normalised fields on a Record in place."""
    assert_backslash_invariant(record.amp, "amp")
    assert_backslash_invariant(record.sq_amp, "sq_amp")

    record["interaction_clean"] = strip_keywords(record.interaction)
    record["vertices_clean"] = strip_keywords(record.vertices)
    record["amp_std"] = standardize(record.amp)
    record["sq_amp_std"] = standardize(record.sq_amp)
    return record
