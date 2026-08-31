"""S1 - parse and validate the raw corpus.

One line is ``interaction : vertices : amp : sq_amp``. Everything here either
succeeds or raises; there is no fallback record (01 P2).
"""

import os
import re
from dataclasses import dataclass, field
from typing import List

_TREE_LEVEL_RE = re.compile(r"TreeLevel-(\d+)")


@dataclass
class Record:
    interaction: str
    vertices: str
    amp: str
    sq_amp: str
    source_file: str
    tree_level: int
    line_no: int
    # Filled in by later stages; absent means the stage has not run, and reading
    # it raises rather than silently yielding a default (02 §6 rule 2).
    extra: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.extra[key]

    def __setitem__(self, key, value):
        self.extra[key] = value

    def __contains__(self, key):
        return key in self.extra


def _count_legs(interaction: str) -> int:
    """External legs, treating ``AntiPart X`` as one leg."""
    body = interaction.replace("Interaction:", "")
    tokens = [t for t in body.replace(" to ", " ").split() if t]
    return sum(1 for t in tokens if t != "AntiPart")


def load_theory(root: str, theory: str, expect_legs: int = 4) -> List[Record]:
    """Load every ``.txt`` for one theory in sorted filename order.

    Sorted order matters: the split is derived from record content, but any
    order-dependent downstream step (vocabulary ties, logging) must be stable
    across machines (02 §2.1).
    """
    theory = theory.upper()
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.endswith(".txt") and theory in name.upper():
                paths.append(os.path.join(dirpath, name))
    paths.sort()

    if not paths:
        raise FileNotFoundError(f"no {theory} .txt files under {root!r}")

    records: List[Record] = []
    for path in paths:
        name = os.path.basename(path)
        level_match = _TREE_LEVEL_RE.search(name)
        tree_level = int(level_match.group(1)) if level_match else -1

        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        non_blank = [(i, ln) for i, ln in enumerate(lines, 1) if ln.strip()]
        for line_no, line in non_blank:
            parts = line.split(" : ")
            if len(parts) != 4:
                raise ValueError(
                    f"{name}:{line_no} has {len(parts)} fields, expected 4")
            interaction, vertices, amp, sq_amp = (p.strip() for p in parts)

            legs = _count_legs(interaction)
            if legs != expect_legs:
                raise ValueError(
                    f"{name}:{line_no} has {legs} external legs, expected "
                    f"{expect_legs}: {interaction!r}")

            records.append(Record(interaction, vertices, amp, sq_amp,
                                  name, tree_level, line_no))

        # G1: one record per non-blank line, asserted rather than assumed.
        produced = sum(1 for r in records if r.source_file == name)
        if produced != len(non_blank):
            raise AssertionError(
                f"{name}: {produced} records from {len(non_blank)} non-blank lines")

    return records
