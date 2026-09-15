r"""Dummy-index normalisation.

MARTY names contracted indices with arbitrary numbers (``%\sigma_18923``). They
only matter through which occurrences match, so they are renumbered in order of
first appearance: ``%\sigma_18923 -> %\sigma_d1``. Physical symbols such as
``s_12`` or ``m_e`` carry no ``%`` and are left alone.
"""

import re

_DUMMY_INDEX = re.compile(r"(%[a-zA-Z\\]+_)(\d+)")


def standardize(expression):
    mapping = {}

    def replace(match):
        prefix, index = match.groups()
        if index not in mapping:
            mapping[index] = f"d{len(mapping) + 1}"
        return prefix + mapping[index]

    return _DUMMY_INDEX.sub(replace, expression)


def strip_keywords(text):
    """Drop the English keywords from the interaction and vertex strings."""
    text = text.replace("Interaction:", "").replace("Vertex", "")
    return " ".join(text.split())
