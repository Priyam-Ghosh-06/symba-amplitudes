"""Train one model and evaluate it on the test set.

    python scripts/train.py --theory QED
    python scripts/train.py --theory QCD --variant no_moe

Writes results/<theory>_<variant>.json and checkpoints/<theory>_<variant>.pt.
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from symba.config import Config
from symba.data.dataset import build
from symba.model.model import AmplitudeModel
from symba.predictor import save_checkpoint
from symba.trainer import evaluate, train_model

# The full model, and one variant per component taken out of it.
VARIANTS = {
    "full": {},
    "graph_only": {"use_math": False},
    "ast_only": {"use_graph": False},
    "no_role_filler": {"embedding": "plain"},
    "no_moe": {"ffn": "dense"},
    "no_xsa": {"attention": "vanilla"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--theory", choices=["QED", "QCD"], default="QED")
    parser.add_argument("--variant", choices=list(VARIANTS), default="full")
    parser.add_argument("--epochs", type=int, help="default 120 for QED, 60 for QCD")
    parser.add_argument("--batch-size", type=int, help="default 16 for QED, 4 for QCD")
    parser.add_argument("--threads", type=int, help="CPU threads for torch")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = Config()
    cfg.data.theory = args.theory
    cfg.model = dataclasses.replace(cfg.model, **VARIANTS[args.variant])
    cfg.train.batch_size = args.batch_size or (16 if args.theory == "QED" else 4)
    # A QCD epoch costs about four QED epochs: its amplitudes are far longer.
    cfg.train.epochs = args.epochs or (120 if args.theory == "QED" else 60)
    torch.manual_seed(cfg.train.seed)

    data = build(cfg)
    cfg.data.segment_amp = data.segment_amp
    model = AmplitudeModel(cfg.model, data.graph_vocab, data.amp_vocab, data.target_vocab,
                           data.table_sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"{args.theory}_{args.variant}"
    print(f"{name}: {model.n_parameters():,} parameters on {device}", flush=True)

    start = time.time()
    trained = train_model(model, data, cfg, device, log=lambda m: print(m, flush=True))
    results = evaluate(model, data, cfg, device)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    save_checkpoint(f"checkpoints/{name}.pt", model, cfg, data)
    with open(f"results/{name}.json", "w", encoding="utf-8") as fh:
        json.dump({"theory": args.theory, "variant": args.variant, "config": cfg.to_dict(),
                   "parameters": model.n_parameters(), "data": data.stats,
                   "minutes": round((time.time() - start) / 60, 1), **trained, **results},
                  fh, indent=1)

    t = results["test"]
    print(f"\n{name} test ({t['n']} records): token acc {100 * t['token_accuracy']:.1f}% | "
          f"sequence acc {100 * t['sequence_accuracy']:.1f}% | "
          f"symbolic acc {100 * t['symbolic_accuracy']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
