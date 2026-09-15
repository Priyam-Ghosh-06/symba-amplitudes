"""Read the SYMBA corpus: one record per line, ``interaction : vertices : amp : sq_amp``."""

import os


def parse_line(line):
    parts = line.split(" : ")
    if len(parts) != 4:
        raise ValueError(f"expected 4 ' : '-separated fields, got {len(parts)}")
    return tuple(p.strip() for p in parts)


def load_theory(root, theory):
    """Every record of one theory (QED or QCD), in sorted file order."""
    folder = os.path.join(root, theory.upper())
    files = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))
    if not files:
        raise FileNotFoundError(f"no .txt files in {folder}")

    records = []
    for name in files:
        with open(os.path.join(folder, name), encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                interaction, vertices, amp, sq_amp = parse_line(line)
                records.append({"interaction": interaction, "vertices": vertices,
                                "amp": amp, "sq_amp": sq_amp,
                                "file": name, "line": line_no})
    return records
