"""Run a batch of experiments sequentially, resumably, in one process.

    python scripts/run_queue.py --batch core
    python scripts/run_queue.py --list
    python scripts/run_queue.py --batch qcd_long --dry-run

Why a queue rather than parallel jobs: this box has 15.7 GB and a QCD run's
resident set reaches ~3.8 GB, so running three at once put it into swap and
every job appeared to hang. One at a time finishes sooner.

Every (arm, seed) already present in the output file is skipped, so an
interrupted batch resumes where it stopped and costs at most one run.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symba.eval.metrics import format_report
from symba.experiment import ARMS, BundleCache, Job, compute_baselines, run_arm

ALL_ARMS = ("full_vanilla_dense", "math_only", "graph_only",
            "full_xsa_proj_dense", "full_xsa_mask_dense", "full_vanilla_moe",
            "unconstrained_decode", "capacity_256", "capacity_64",
            "no_type_embedding", "role_filler", "tpr_binding")

# Batches are ordered so the most informative result lands first. Training
# length is 120 epochs everywhere: validation symbolic EM was still climbing at
# 30 (8.3% -> 36.1% -> 63.9% -> 75.0%), so a shorter run measures the compute
# budget rather than the architecture.
BATCHES = {
    # The two headline numbers, and the QCD rerun that the 30-epoch results need.
    "core": [
        Job("QCD_record_long", theory="QCD",
            arms=("full_vanilla_dense", "math_only", "graph_only")),
        Job("QED_record_arms", theory="QED", baselines=False,
            arms=("full_xsa_proj_dense", "full_xsa_mask_dense",
                  "full_vanilla_moe", "unconstrained_decode")),
    ],
    # Confidence intervals. Three seeds is the minimum for an arm comparison.
    "seeds": [
        Job("QED_record_seeds", theory="QED", seeds=(1, 2),
            arms=("full_vanilla_dense", "math_only", "graph_only")),
        Job("QCD_record_seeds", theory="QCD", seeds=(1, 2),
            arms=("full_vanilla_dense", "math_only", "graph_only")),
    ],
    # Protocol B at a usable held-out size: the fraction applies to template
    # classes, not records, so 10% would hold out two classes and tell us nothing.
    "template": [
        Job("QED_templateB_long", theory="QED", protocol="template",
            val_frac=0.15, test_frac=0.30,
            arms=("full_vanilla_dense", "math_only", "graph_only")),
        Job("QCD_templateB_long", theory="QCD", protocol="template",
            val_frac=0.18, test_frac=0.30,
            arms=("full_vanilla_dense", "math_only", "graph_only")),
    ],
    "capacity": [
        Job("QED_capacity", theory="QED", baselines=False,
            arms=("capacity_64", "capacity_256", "no_type_embedding",
                  "role_filler", "tpr_binding")),
    ],
}


def load(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", nargs="+", default=["core"],
                        help=f"'all', or one or more of: {', '.join(BATCHES)}")
    parser.add_argument("--list", action="store_true",
                        help="show the batches and what is already done")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override every job's epoch budget")
    args = parser.parse_args()

    if args.list:
        for name, jobs in BATCHES.items():
            print(f"\n{name}")
            for job in jobs:
                done = load(job.out_path) or {"arms": {}}
                have = {f"{a}/{s}" for a, seeds in done["arms"].items()
                        for s in seeds}
                want = [f"{a}/{s}" for a in job.arms for s in job.seeds]
                todo = [w for w in want if w not in have]
                print(f"  {job.name:<22} {len(want) - len(todo):>2}/{len(want)} done"
                      + (f"  next: {todo[0]}" if todo else "  complete"))
        return

    names = list(BATCHES) if "all" in args.batch else args.batch
    unknown = [n for n in names if n not in BATCHES]
    if unknown:
        parser.error(f"unknown batch(es) {unknown}; choose from "
                     f"'all' or {list(BATCHES)}")
    jobs = [j for name in names for j in BATCHES[name]]
    if args.epochs:
        for job in jobs:
            job.epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundles = BundleCache()
    print(f"device={device}  jobs={len(jobs)}  "
          f"threads={torch.get_num_threads()}")

    for job in jobs:
        os.makedirs("results", exist_ok=True)
        results = load(job.out_path) or {
            "theory": job.theory, "protocol": job.protocol,
            "device": str(device), "arms": {}, "baselines": {},
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        def flush():
            with open(job.out_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, default=str)

        pending = [(a, s) for a in job.arms for s in job.seeds
                   if str(s) not in results["arms"].get(a, {})]
        if not pending:
            print(f"\n== {job.name}: already complete, skipping")
            continue

        print(f"\n== {job.name}  ({job.theory}/{job.protocol}, "
              f"{len(pending)} run(s) pending, {job.epochs} epochs)")
        if args.dry_run:
            for arm, seed in pending:
                print(f"   would run {arm} seed{seed}")
            continue

        for seed in job.seeds:
            cfg = job.config(seed)
            bundle = bundles.get(cfg)
            results["data"] = bundle.stats
            if job.baselines and str(seed) not in results["baselines"]:
                scored = compute_baselines(job, seed, bundles)
                results["baselines"][str(seed)] = scored
                for name, metrics in scored.items():
                    print(format_report(f"baseline seed{seed} / {name}", metrics))
                flush()

        for index, (arm, seed) in enumerate(pending, 1):
            print(f"\n[{index}/{len(pending)}] {job.name} :: {arm} seed{seed}")
            entry = run_arm(arm, seed, job, bundles, device)
            results["arms"].setdefault(arm, {})[str(seed)] = entry
            flush()
            if "test" in entry:
                print(format_report(f"{arm} seed{seed} TEST", entry["test"]))
            else:
                print(f"   FAILED: {entry.get('error')}")

        print(f"== {job.name} written to {job.out_path}", flush=True)

    print()
    print("=" * 60)
    print("QUEUE FINISHED")
    for job in jobs:
        done = load(job.out_path) or {"arms": {}}
        have = {f"{a}/{s}" for a, seeds in done["arms"].items() for s in seeds}
        failed = [f"{a}/{s}" for a, seeds in done["arms"].items()
                  for s, entry in seeds.items() if "error" in entry]
        want = [f"{a}/{s}" for a in job.arms for s in job.seeds]
        missing = [w for w in want if w not in have]
        status = "complete" if not missing and not failed else (
            f"{len(want) - len(missing)}/{len(want)} done"
            + (f", {len(failed)} failed" if failed else ""))
        print(f"  {job.name:<22} {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
