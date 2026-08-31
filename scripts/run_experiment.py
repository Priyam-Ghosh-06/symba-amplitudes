"""Run the arm grid and write results incrementally.

    python scripts/run_experiment.py --theory QED --seeds 0 1 2
    python scripts/run_experiment.py --theory QCD --arms full_vanilla_dense

Every arm is trained from fresh weights, selected on free-running symbolic exact
match on the validation split, and finally scored once on the held-out test
split. Results are flushed to JSON after every run, so a partial grid is still
readable.
"""

import argparse
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symba.config import Config
from symba.data.pipeline import build
from symba.eval.baselines import run_all as run_baselines
from symba.eval.decode import ConstraintMask
from symba.eval.metrics import format_report
from symba.model.model import AmplitudeModel
from symba.train.loop import evaluate_split, train_model

# name -> config overrides. Every arm differs from the control in one factor,
# so a difference is attributable (01 P4).
ARMS = {
    "full_vanilla_dense":   {},                                    # control
    "full_xsa_proj_dense":  {"model.attention": "xsa_proj"},
    "full_xsa_mask_dense":  {"model.attention": "xsa_mask"},
    "full_vanilla_moe":     {"model.ffn": "moe"},
    "graph_only":           {"model.use_math": False},
    "math_only":            {"model.use_graph": False},
    "no_type_embedding":    {"model.use_type_embedding": False},
    "role_filler":          {"model.embedding": "role_filler"},
    "tpr_binding":          {"model.embedding": "tpr"},
    "capacity_256":         {"model.d_model": 256,
                             "model.dim_feedforward": 1024},
    "capacity_64":          {"model.d_model": 64,
                             "model.dim_feedforward": 256},
    "unconstrained_decode": {"train.constrained_decoding": False},
    "raw_target":           {"data.target": "raw"},
}

DEFAULT_ARMS = ["full_vanilla_dense", "full_xsa_proj_dense",
                "full_xsa_mask_dense", "full_vanilla_moe",
                "graph_only", "math_only", "capacity_256",
                "unconstrained_decode"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--theory", default="QED", choices=["QED", "QCD"])
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                   choices=sorted(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--protocol", default="record",
                   choices=["record", "template"])
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1,
                   help="under protocol template this is a fraction of the "
                        "template classes, so raise it or the test split is a "
                        "handful of records from one or two classes")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--data", default="data/Symba")
    p.add_argument("--out", default=None)
    p.add_argument("--skip-baselines", action="store_true")
    return p.parse_args()


def base_config(args, seed):
    batch = args.batch_size or (16 if args.theory == "QED" else 4)
    return Config().with_overrides(**{
        "name": f"{args.theory}_{args.protocol}",
        "data.root": args.data,
        "data.theory": args.theory,
        "data.split_protocol": args.protocol,
        "data.val_frac": args.val_frac,
        "data.test_frac": args.test_frac,
        "train.num_epochs": args.epochs,
        "train.batch_size": batch,
        "train.patience": args.patience,
        "train.eval_every": args.eval_every,
        "train.seed": seed,
    })


def main():
    args = parse_args()
    out_path = args.out or f"results/{args.theory}_{args.protocol}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"theory": args.theory, "protocol": args.protocol,
               "device": str(device), "arms": {}, "baselines": {},
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def flush():
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

    # Bundles are per seed: the split depends on the seed, so the baselines do
    # too and must be recomputed alongside the models they are compared against.
    bundles = {}
    for seed in args.seeds:
        cfg = base_config(args, seed)
        bundles[seed] = (cfg, build(cfg))
        results["data"] = bundles[seed][1].stats

    if not args.skip_baselines:
        print("\nBaselines (test split):")
        for seed in args.seeds:
            _cfg, bundle = bundles[seed]
            scored = run_baselines(bundle.train, bundle.test)
            results["baselines"][str(seed)] = scored
            for name, metrics in scored.items():
                print(format_report(f"seed {seed} / {name}", metrics))
        flush()

    total = len(args.arms) * len(args.seeds)
    done = 0
    for arm in args.arms:
        results["arms"].setdefault(arm, {})
        for seed in args.seeds:
            done += 1
            cfg, bundle = bundles[seed]
            run_cfg = cfg.with_overrides(**ARMS[arm])
            run_name = f"{args.theory}/{arm}/seed{seed}"
            print(f"\n[{done}/{total}] {run_name}")

            t0 = time.time()
            try:
                if run_cfg.data.target != cfg.data.target:
                    # This arm changes the data, so it needs its own bundle.
                    run_bundle = build(run_cfg, verbose=False)
                else:
                    run_bundle = bundle

                model = AmplitudeModel(run_cfg.model, run_bundle.graph_vocab,
                                       run_bundle.amp_vocab,
                                       run_bundle.target_vocab,
                                       run_bundle.lengths)
                trained = train_model(model, run_bundle, run_cfg, device,
                                      run_name=run_name)

                max_len = run_bundle.lengths[2] + 4
                constraint = ConstraintMask(run_bundle.target_vocab)
                test_metrics, predictions = evaluate_split(
                    model, run_bundle.loaders["test"],
                    run_bundle.datasets["test"], run_bundle.target_vocab,
                    run_cfg.train, device, max_len, constraint)

                entry = {
                    "config": run_cfg.to_dict(),
                    "n_parameters": model.n_parameters(),
                    "train": trained,
                    "test": test_metrics,
                    "moe_stats": model.moe_stats()[:4],
                    "minutes": round((time.time() - t0) / 60, 2),
                    "example_predictions": [" ".join(p) for p in predictions[:3]],
                }
                print(format_report(f"{run_name} TEST", test_metrics))
            except Exception as exc:
                traceback.print_exc()
                entry = {"error": f"{type(exc).__name__}: {exc}",
                         "minutes": round((time.time() - t0) / 60, 2)}

            results["arms"][arm][str(seed)] = entry
            flush()

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flush()
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
