"""Split protocols (01 SS5) and the template classes they depend on.

Protocol A (record) is the headline number the objective asks for. Protocol B
(template) holds out whole functional forms and is the number that says whether
the model learned physics or a codebook. They are reported together; the gap is
the result.

Splits are assigned by hashing record content, not by shuffling a list, so they
do not depend on filesystem traversal order (02 SS2.1).
"""

import hashlib
from collections import defaultdict

import sympy

from .canonical import template_key

_PARTICLE_TO_MASS = {}          # populated lazily; m_<particle>


def _stable_hash(text: str) -> float:
    """Uniform value in [0, 1) derived from the record's own content."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def leg_mass_order(record) -> list:
    """Mass symbols in external-leg order, deduplicated, restricted to those
    that actually appear in the target.

    This is the abstraction that collapses 360 QED records to 30 functions:
    ``m_e -> M1`` by the leg the mass first belongs to, so two records that
    differ only in flavour become the same expression.
    """
    graph = record["graph"]
    present = {s.name for s in record["canon_expr"].free_symbols}

    order, seen = [], set()
    for particle, _anti in graph.in_legs + graph.out_legs:
        name = f"m_{particle}"
        if name in present and name not in seen:
            seen.add(name)
            order.append(name)

    # Any mass in the target that is not attached to a leg still needs a slot,
    # or the substitution would leave it un-abstracted and split the class.
    for name in sorted(present):
        if name.startswith("m_") and name not in seen:
            seen.add(name)
            order.append(name)
    return order


def assign_template_classes(records) -> dict:
    """Attach ``template`` to every record; return ``{template: [records]}``."""
    classes = defaultdict(list)
    for record in records:
        key = template_key(record["canon_expr"], leg_mass_order(record))
        record["template"] = key
        classes[key].append(record)
    return dict(classes)


def split_records(records, protocol="record", val_frac=0.1, test_frac=0.1,
                  seed=0):
    """Return ``(train, val, test)`` as three disjoint lists.

    ``record``   - protocol A, hash each record independently.
    ``template`` - protocol B, hash the template class so every record of a
                   held-out functional form lands on the same side.
    """
    if protocol == "record":
        key_of = lambda r: f"{seed}|{r.amp}|{r.sq_amp}"
    elif protocol == "template":
        key_of = lambda r: f"{seed}|{r['template']}"
    else:
        raise ValueError(f"unknown split protocol {protocol!r}")

    train, val, test = [], [], []
    for record in records:
        h = _stable_hash(key_of(record))
        if h < test_frac:
            test.append(record)
        elif h < test_frac + val_frac:
            val.append(record)
        else:
            train.append(record)

    return train, val, test


def assert_disjoint(train, val, test):
    """Gate G12: the three index sets are pairwise disjoint."""
    ids = [set(map(id, part)) for part in (train, val, test)]
    names = ("train", "val", "test")
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = ids[i] & ids[j]
            if overlap:
                raise AssertionError(
                    f"{names[i]}/{names[j]} share {len(overlap)} records")
    return True


def class_summary(records) -> dict:
    """Counts per template class, for the per-class error breakdown (03 SS5.2)."""
    counts = defaultdict(int)
    for record in records:
        counts[record["template"]] += 1
    return dict(counts)
