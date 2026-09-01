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

from .canonical import template_key


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
    if protocol == "template":
        return _split_by_class(records, val_frac, test_frac, seed)

    if protocol != "record":
        raise ValueError(f"unknown split protocol {protocol!r}")

    train, val, test = [], [], []
    for record in records:
        h = _stable_hash(f"{seed}|{record.amp}|{record.sq_amp}")
        if h < test_frac:
            test.append(record)
        elif h < test_frac + val_frac:
            val.append(record)
        else:
            train.append(record)

    return train, val, test


def _split_by_class(records, val_frac, test_frac, seed):
    """Grouped split: whole template classes go to one side.

    Classes are ranked by a stable hash and then assigned *by count*, not by a
    hash threshold. With 11 QCD classes a threshold can leave a split empty
    purely by luck; ranking guarantees at least one class per split whenever
    there are three or more.
    """
    by_class = defaultdict(list)
    for record in records:
        by_class[record["template"]].append(record)

    ordered = sorted(by_class, key=lambda t: _stable_hash(f"{seed}|{t}"))
    n = len(ordered)
    if n < 3:
        raise ValueError(f"template split needs at least 3 classes, found {n}")

    n_test = min(n - 2, max(1, round(test_frac * n)))
    n_val = min(n - 1 - n_test, max(1, round(val_frac * n)))

    test_classes = ordered[:n_test]
    val_classes = ordered[n_test:n_test + n_val]
    train_classes = ordered[n_test + n_val:]

    def collect(classes):
        return [r for c in classes for r in by_class[c]]

    return collect(train_classes), collect(val_classes), collect(test_classes)


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

