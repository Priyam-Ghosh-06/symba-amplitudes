"""One experiment run, and the pieces both CLIs share.

Split out of ``scripts/run_experiment.py`` so a queue can execute many jobs in
a single process. That matters for two reasons measured on this machine:

* concurrent jobs thrash. 15.7 GB total, and a QCD job resident set reaches
  ~3.8 GB, so three at once put the box into swap and every one of them
  appeared to hang. Sequential is faster than parallel here.
* building a bundle costs ~20 s of sympy (canonicalisation plus template
  classing). Jobs that share a (theory, protocol, seed) can share the bundle
  instead of paying for it again.
"""

import time
import traceback
from dataclasses import dataclass, field

import torch

from .config import Config
from .data.pipeline import build
from .eval.baselines import run_all as run_baselines
from .eval.decode import ConstraintMask
from .eval.metrics import clear_caches
from .model.model import AmplitudeModel
from .train.loop import evaluate_split, train_model

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


@dataclass
class Job:
    """One output file's worth of work."""
    name: str
    theory: str = "QED"
    protocol: str = "record"
    arms: tuple = ("full_vanilla_dense",)
    seeds: tuple = (0,)
    epochs: int = 120
    patience: int = 25
    eval_every: int = 10
    batch_size: int = None
    val_frac: float = 0.1
    test_frac: float = 0.1
    baselines: bool = True
    data_root: str = "data/Symba"

    @property
    def out_path(self) -> str:
        return f"results/{self.name}.json"

    def config(self, seed: int) -> Config:
        batch = self.batch_size or (16 if self.theory == "QED" else 4)
        return Config().with_overrides(**{
            "name": self.name,
            "data.root": self.data_root,
            "data.theory": self.theory,
            "data.split_protocol": self.protocol,
            "data.val_frac": self.val_frac,
            "data.test_frac": self.test_frac,
            "train.num_epochs": self.epochs,
            "train.batch_size": batch,
            "train.patience": self.patience,
            "train.eval_every": self.eval_every,
            "train.seed": seed,
        })


class BundleCache:
    """Memoises built bundles, keyed by everything the build depends on."""

    def __init__(self, limit: int = 3):
        self.limit = limit
        self._store = {}

    def get(self, cfg: Config, verbose: bool = True):
        key = (cfg.data.theory, cfg.data.split_protocol, cfg.data.val_frac,
               cfg.data.test_frac, cfg.data.target, cfg.data.amp_representation,
               cfg.train.seed, cfg.train.batch_size)
        if key not in self._store:
            if len(self._store) >= self.limit:
                self._store.pop(next(iter(self._store)))
            self._store[key] = build(cfg, verbose=verbose)
        return self._store[key]


def run_arm(arm: str, seed: int, job: Job, bundles: BundleCache,
            device, log=print) -> dict:
    """Train and test one arm. Returns the result entry, never raises."""
    cfg = job.config(seed)
    run_cfg = cfg.with_overrides(**ARMS[arm])
    run_name = f"{job.theory}/{arm}/seed{seed}"
    t0 = time.time()

    try:
        bundle = bundles.get(run_cfg, verbose=False)

        model = AmplitudeModel(run_cfg.model, bundle.graph_vocab,
                               bundle.amp_vocab, bundle.target_vocab,
                               bundle.lengths)
        trained = train_model(model, bundle, run_cfg, device,
                              run_name=run_name, log=log)

        max_len = bundle.lengths[2] + 4
        test_metrics, predictions = evaluate_split(
            model, bundle.loaders["test"], bundle.datasets["test"],
            bundle.target_vocab, run_cfg.train, device, max_len,
            ConstraintMask(bundle.target_vocab))

        entry = {
            "config": run_cfg.to_dict(),
            "n_parameters": model.n_parameters(),
            "train": trained,
            "test": test_metrics,
            "moe_stats": model.moe_stats()[:4],
            "minutes": round((time.time() - t0) / 60, 2),
            "example_predictions": [" ".join(p) for p in predictions[:3]],
        }
    except Exception as exc:
        traceback.print_exc()
        entry = {"error": f"{type(exc).__name__}: {exc}",
                 "minutes": round((time.time() - t0) / 60, 2)}
    finally:
        # Each run scores hundreds of expressions; sympy keeps every
        # intermediate unless told otherwise.
        clear_caches()

    return entry


def compute_baselines(job: Job, seed: int, bundles: BundleCache) -> dict:
    bundle = bundles.get(job.config(seed), verbose=False)
    scored = run_baselines(bundle.train, bundle.test)
    clear_caches()
    return scored
