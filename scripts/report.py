"""Turn results JSON into the tables of 01 SS6, with n and Wilson intervals.

    python scripts/report.py results/QED_record_seed0.json
    python scripts/report.py results/*.json --markdown
"""

import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METRICS = [
    ("symbolic_exact_match", "symbolic EM"),
    ("raw_exact_match", "raw EM"),
    ("parse_validity", "parse ok"),
    ("mass_dimension_validity", "dim-4 ok"),
    ("channel_accuracy", "channel"),
]


def _cell(metric, markdown=False):
    if metric is None:
        return "     -"
    value = metric["value"] * 100
    if "ci_low" in metric:
        return (f"{value:5.1f} [{metric['ci_low'] * 100:.0f},"
                f"{metric['ci_high'] * 100:.0f}]")
    return f"{value:5.1f}"


def report_file(path, markdown=False):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    lines = []
    theory, protocol = data.get("theory"), data.get("protocol")
    stats = data.get("data", {})
    lines.append(f"\n{'=' * 78}")
    lines.append(f"{theory} / protocol {protocol} / {os.path.basename(path)}")
    if stats:
        split = stats.get("split", {})
        lines.append(f"  {stats.get('n_records')} records, "
                     f"{stats.get('n_templates')} templates | "
                     f"train/val/test {split.get('train')}/{split.get('val')}/"
                     f"{split.get('test')} | vocab {stats.get('vocab')}")
    lines.append("=" * 78)

    header = f"{'run':<34}" + "".join(f"{label:>14}" for _k, label in METRICS)
    lines.append(header)
    lines.append("-" * len(header))

    for seed, baselines in sorted(data.get("baselines", {}).items()):
        for name, metrics in baselines.items():
            row = f"{'[baseline] ' + name:<34}"
            row += "".join(f"{_cell(metrics.get(k)):>14}" for k, _l in METRICS)
            lines.append(row)
    if data.get("baselines"):
        lines.append("-" * len(header))

    for arm, seeds in data.get("arms", {}).items():
        values = []
        for seed, entry in sorted(seeds.items()):
            if "error" in entry:
                lines.append(f"{arm + ' s' + seed:<34}  ERROR {entry['error'][:60]}")
                continue
            test = entry.get("test", {})
            row = f"{arm + ' s' + seed:<34}"
            row += "".join(f"{_cell(test.get(k)):>14}" for k, _l in METRICS)
            params = entry.get("n_parameters")
            minutes = entry.get("minutes")
            row += f"   {params / 1e6:.2f}M  {minutes:.0f}min" if params else ""
            lines.append(row)
            values.append(test.get("symbolic_exact_match", {}).get("value", 0))

        if len(values) > 1:
            mean = statistics.mean(values) * 100
            spread = statistics.pstdev(values) * 100
            lines.append(f"{'  -> mean over seeds':<34}{mean:>13.1f} "
                         f"(sd {spread:.1f})")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    paths = []
    for pattern in args.paths:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    for path in paths:
        if not os.path.exists(path):
            print(f"missing: {path}")
            continue
        print(report_file(path, args.markdown))


if __name__ == "__main__":
    main()
