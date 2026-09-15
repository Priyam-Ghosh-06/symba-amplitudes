"""Predict squared amplitudes with a trained model.

    python scripts/predict.py checkpoints/QED_full.pt data/Symba/QED/QED-2-to-2-diag-TreeLevel-0.txt --limit 3

Each input line is in corpus format; its squared amplitude is used only to
check the prediction.
"""

import argparse
import os
import sys

import sympy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from symba.data.canonical import canonicalise
from symba.data.load import parse_line
from symba.data.normalize import standardize
from symba.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("input", help="file of lines 'interaction : vertices : amp : sq_amp'")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    predictor = Predictor(args.checkpoint)
    with open(args.input, encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()][:args.limit]

    correct = 0
    for i, line in enumerate(lines, 1):
        interaction, vertices, amp, sq_amp = parse_line(line)
        predicted, _ = predictor.predict(interaction, vertices, amp)
        numerator, denominator = canonicalise(standardize(sq_amp))
        truth = numerator / denominator
        ok = predicted is not None and sympy.cancel(predicted - truth) == 0
        correct += ok
        print(f"[{i}] {interaction.replace('Interaction:', '').strip()}")
        print(f"    predicted: {predicted}")
        print(f"    truth:     {truth}")
        print(f"    {'correct' if ok else 'wrong'}\n")
    print(f"{correct}/{len(lines)} correct")


if __name__ == "__main__":
    main()
