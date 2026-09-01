"""Predict squared amplitudes with a trained checkpoint.

    # one amplitude, given on the command line
    python scripts/predict.py --checkpoint checkpoints/QED_record_long__full_vanilla_dense__seed0.pt \
        --interaction "..." --vertices "..." --amp "..."

    # a whole file in SYMBA corpus format (4 fields separated by ' : ')
    python scripts/predict.py --checkpoint <ckpt> --input data/Symba/QED/QED-2-to-2-diag-TreeLevel-0.txt --limit 5

    # list what checkpoints exist
    python scripts/predict.py --list

When the input line carries a ground-truth ``sq_amp`` it is scored against the
prediction with the same symbolic-equivalence test used in training, so a file
from the corpus doubles as a spot check.
"""

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symba.data.canonical import canonicalise
from symba.data.normalize import standardize
from symba.data.serialize import target_tokens
from symba.eval.metrics import symbolically_equal
from symba.inference import Predictor, parse_line


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="path to a .pt written by training")
    p.add_argument("--input", help="file of SYMBA-format lines")
    p.add_argument("--interaction")
    p.add_argument("--vertices")
    p.add_argument("--amp")
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N lines of --input")
    p.add_argument("--beam-width", type=int, default=None)
    p.add_argument("--unconstrained", action="store_true",
                   help="disable the grammar constraint (it is on by default)")
    p.add_argument("--json-out", default=None)
    p.add_argument("--list", action="store_true",
                   help="list available checkpoints and exit")
    return p.parse_args()


def list_checkpoints():
    paths = sorted(glob.glob("checkpoints/*.pt"))
    if not paths:
        print("No checkpoints yet. Train one first:\n"
              "  python scripts/run_queue.py --batch core")
        return
    print(f"{len(paths)} checkpoint(s):")
    for path in paths:
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            metrics = blob.get("metrics", {}).get("test", {})
            em = metrics.get("symbolic_exact_match", {}).get("value")
            stats = blob.get("data_stats", {})
            note = (f"  test symbolic EM {em * 100:.1f}%" if em is not None
                    else "  (no test metrics)")
            print(f"  {path}\n      {stats.get('theory', '?')} "
                  f"{stats.get('n_records', '?')} records{note}")
        except Exception as exc:
            print(f"  {path}   [unreadable: {exc}]")


def main():
    args = parse_args()

    if args.list:
        list_checkpoints()
        return

    if not args.checkpoint:
        print("--checkpoint is required (or use --list)")
        sys.exit(2)

    records, truths = [], []
    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                interaction, vertices, amp, sq_amp = parse_line(line)
                records.append({"interaction": interaction,
                                "vertices": vertices, "amp": amp})
                truths.append(sq_amp)
                if args.limit and len(records) >= args.limit:
                    break
    elif args.interaction and args.vertices and args.amp:
        records.append({"interaction": args.interaction,
                        "vertices": args.vertices, "amp": args.amp})
        truths.append(None)
    else:
        print("give either --input FILE or all of "
              "--interaction/--vertices/--amp")
        sys.exit(2)

    predictor = Predictor(args.checkpoint,
                          beam_width=args.beam_width,
                          constrained=not args.unconstrained)
    print(f"loaded {args.checkpoint}\n"
          f"  beam width {predictor.beam_width}, "
          f"constrained={predictor.constrained}, "
          f"segmented amplitude={predictor.segment_amp}\n"
          f"  {len(records)} record(s)\n")

    predictions = predictor.predict(records)

    correct = scored = 0
    payload = []
    for i, (record, prediction) in enumerate(zip(records, predictions), 1):
        print(f"[{i}] {record['interaction'][:78]}")
        print(f"    predicted : {prediction['prefix'][:150]}")
        if prediction["expression"] is not None:
            print(f"    expression: {str(prediction['expression'])[:150]}")

        entry = {"interaction": record["interaction"],
                 "prefix": prediction["prefix"],
                 "expression": str(prediction["expression"]),
                 "well_formed": prediction["well_formed"]}

        truth = truths[i - 1]
        if truth:
            numerator, denominator = canonicalise(standardize(truth))
            reference = numerator / denominator
            match = symbolically_equal(prediction["expression"], reference)
            scored += 1
            correct += bool(match)
            entry["correct"] = bool(match)
            entry["reference"] = str(reference)
            print(f"    truth     : {str(reference)[:150]}")
            print(f"    {'MATCH' if match else 'MISMATCH'}")
        print()
        payload.append(entry)

    if scored:
        print(f"symbolic exact match: {correct}/{scored} "
              f"({correct / scored * 100:.1f}%)")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
