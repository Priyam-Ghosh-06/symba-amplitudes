"""Figures for the write-up, built from results/*.json.

    python scripts/plots.py

Writes PNGs to results/figures/. Every panel carries n and a 95% Wilson interval,
because with test splits of 14-102 records a bare bar chart would imply a
precision the data does not have.
"""

import glob
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "results/figures"
BLUE, RED, GREY, GREEN = "#2E86AB", "#E84855", "#9AA0A6", "#3BB273"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.autolayout": False,
})


def load_all() -> dict:
    out = {}
    for path in sorted(glob.glob("results/*.json")):
        with open(path, encoding="utf-8") as fh:
            out[os.path.basename(path)[:-5]] = json.load(fh)
    return out


def _em(entry):
    """(value, low, high, n) in percent, or None."""
    m = (entry or {}).get("test", {}).get("symbolic_exact_match")
    if not m:
        return None
    return (m["value"] * 100, m["ci_low"] * 100, m["ci_high"] * 100, m["n"])


def collect_arm(data, files, arm):
    """Every (value, lo, hi, n) for one arm across the given result files."""
    out = []
    for name in files:
        for seed, entry in sorted(data.get(name, {}).get("arms", {})
                                  .get(arm, {}).items()):
            got = _em(entry)
            if got:
                out.append(got)
    return out


def baseline(data, files, name):
    vals = []
    for f in files:
        for _seed, block in sorted(data.get(f, {}).get("baselines", {}).items()):
            m = block.get(name, {}).get("symbolic_exact_match")
            if m:
                vals.append(m["value"] * 100)
    return statistics.mean(vals) if vals else None


# --- Figure 1: the headline, model against the baselines it must beat -------

def fig_headline(data):
    theories = [
        ("QED", ["QED_record_seeds"]),
        ("QCD", ["QCD_record_seeds"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, (theory, files) in zip(axes, theories):
        rows = [
            ("most frequent", baseline(data, files, "most_frequent"), GREY),
            ("1-NN retrieval", baseline(data, files, "nearest_neighbour"), GREY),
            ("template oracle", baseline(data, files, "template_oracle"), RED),
        ]
        runs = collect_arm(data, files, "full_vanilla_dense")
        model = statistics.mean(r[0] for r in runs) if runs else 0.0

        labels = [r[0] for r in rows] + [f"model\n({len(runs)} seeds)"]
        values = [(r[1] or 0.0) for r in rows] + [model]
        colours = [r[2] for r in rows] + [BLUE]

        bars = ax.bar(labels, values, color=colours, width=0.62)
        # Seed spread on the model bar; the baselines are deterministic per split.
        if len(runs) > 1:
            ax.errorbar(len(labels) - 1, model,
                        yerr=statistics.pstdev([r[0] for r in runs]),
                        fmt="none", ecolor="black", capsize=4, lw=1.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2,
                    f"{value:.0f}", ha="center", fontsize=8)

        ax.set_ylim(0, 112)
        ax.set_ylabel("symbolic exact match (%)")
        ax.set_title(f"{theory} — protocol A (record split)")
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("The model must clear retrieval, not just score highly",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/01_headline_vs_baselines.png", bbox_inches="tight")
    plt.close(fig)


# --- Figure 2: the honest one -----------------------------------------------

def fig_protocols(data):
    fig, ax = plt.subplots(figsize=(7.2, 4))
    groups = [
        ("QED", ["QED_record_seeds"],
         ["QED_templateB_long"]),
        ("QCD", ["QCD_record_seeds"],
         ["QCD_templateB_long"]),
    ]
    x = np.arange(len(groups))
    width = 0.35

    a_vals, b_vals = [], []
    for _t, a_files, b_files in groups:
        a = collect_arm(data, a_files, "full_vanilla_dense")
        b = collect_arm(data, b_files, "full_vanilla_dense")
        a_vals.append(statistics.mean(r[0] for r in a) if a else 0)
        b_vals.append(statistics.mean(r[0] for r in b) if b else 0)

    ax.bar(x - width / 2, a_vals, width, label="protocol A — record split\n(formula seen in training)", color=BLUE)
    ax.bar(x + width / 2, b_vals, width, label="protocol B — template split\n(formula never seen)", color=RED)

    for xi, (a, b) in enumerate(zip(a_vals, b_vals)):
        ax.text(xi - width / 2, a + 2, f"{a:.0f}", ha="center", fontsize=9)
        ax.text(xi + width / 2, b + 2, f"{b:.0f}", ha="center", fontsize=9)

    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylim(0, 115)
    ax.set_ylabel("symbolic exact match (%)")
    ax.set_title("Interpolation is solved. Generalisation is not.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
              fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_protocol_A_vs_B.png", bbox_inches="tight")
    plt.close(fig)


# --- Figure 3: modality ablation, the project's Claim 1 ----------------------

def fig_modality(data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    setups = [
        ("QED", ["QED_record_seeds"]),
        ("QCD", ["QCD_record_seeds"]),
    ]
    arms = [("graph_only", "graph only"), ("math_only", "AST only"),
            ("full_vanilla_dense", "graph + AST")]

    for ax, (theory, files) in zip(axes, setups):
        labels, means, spreads, ns = [], [], [], []
        for arm, label in arms:
            runs = collect_arm(data, files, arm)
            if not runs:
                continue
            values = [r[0] for r in runs]
            labels.append(f"{label}\n({len(values)} seeds)")
            means.append(statistics.mean(values))
            spreads.append(statistics.pstdev(values) if len(values) > 1 else 0)
            ns.append(sum(r[3] for r in runs))

        colours = [GREY, GREY, BLUE][-len(means):]
        ax.bar(labels, means, yerr=spreads, color=colours, width=0.6,
               capsize=4, error_kw={"lw": 1.2})
        for i, (m, s) in enumerate(zip(means, spreads)):
            ax.text(i, m + s + 2, f"{m:.1f}", ha="center", fontsize=8)

        ax.set_ylim(0, 115)
        ax.set_ylabel("symbolic exact match (%)")
        ax.set_title(f"{theory} — do both pathways help?")
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Claim 1: modality ablation, mean over seeds, bars = seed spread",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/03_modality_ablation.png", bbox_inches="tight")
    plt.close(fig)


# --- Figure 4: learning curves ----------------------------------------------

def fig_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    sources = [("QED", "QED_record_seeds"), ("QCD", "QCD_record_seeds")]

    for ax, (theory, name) in zip(axes, sources):
        entry = data.get(name, {}).get("arms", {}).get(
            "full_vanilla_dense", {}).get("0")
        if not entry:
            continue
        history = entry["train"]["history"]

        epochs = [h["epoch"] for h in history]
        ax.plot(epochs, [h["train_loss"] for h in history],
                color=GREY, lw=1, label="train loss")
        ax.plot(epochs, [h["val_loss"] for h in history],
                color=RED, lw=1, label="val loss")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy (log)")

        twin = ax.twinx()
        pts = [(h["epoch"], h["val_symbolic_em"] * 100) for h in history
               if "val_symbolic_em" in h]
        twin.plot([p[0] for p in pts], [p[1] for p in pts],
                  color=BLUE, marker="o", ms=3, lw=1.6,
                  label="val symbolic EM")
        twin.set_ylabel("val symbolic exact match (%)", color=BLUE)
        twin.tick_params(axis="y", labelcolor=BLUE)
        twin.set_ylim(0, 105)
        twin.spines["right"].set_visible(True)

        ax.set_title(f"{theory} — loss saturates long before accuracy does")
        lines = ax.get_lines() + twin.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=7,
                  loc="center right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/04_learning_curves.png", bbox_inches="tight")
    plt.close(fig)


# --- Figure 5: where the errors are -----------------------------------------

def fig_decomposition(data):
    metrics = [("parse_validity", "parses"),
               ("mass_dimension_validity", "mass-dim 4"),
               ("channel_accuracy", "right channel"),
               ("monomial_f1", "monomial F1"),
               ("symbolic_exact_match", "fully correct")]
    setups = [("QED protocol A", "QED_record_seeds", "full_vanilla_dense", "0"),
              ("QCD protocol A", "QCD_record_seeds", "full_vanilla_dense", "0"),
              ("QED protocol B", "QED_templateB_long", "full_vanilla_dense", "0"),
              ("QCD protocol B", "QCD_templateB_long", "full_vanilla_dense", "0")]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(metrics))
    width = 0.2

    for i, (label, name, arm, seed) in enumerate(setups):
        entry = data.get(name, {}).get("arms", {}).get(arm, {}).get(seed)
        if not entry or "test" not in entry:
            continue
        test = entry["test"]
        values = [test.get(k, {}).get("value", 0) * 100 for k, _l in metrics]
        colour = BLUE if "protocol A" in label else RED
        ax.bar(x + (i - 1.5) * width, values, width, label=label,
               color=colour, alpha=1.0 if i % 2 == 0 else 0.6)

    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylabel("%")
    ax.set_ylim(0, 112)
    ax.set_title("Error decomposition: on unseen formulas the output stays\n"
                 "well-formed but stops being dimensionally possible")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/05_error_decomposition.png", bbox_inches="tight")
    plt.close(fig)


# --- Figure 6: seed spread vs arm spread ------------------------------------

def fig_seed_noise(data):
    files = ["QED_record_seeds", "QED_record_arms"]
    arms = ["full_vanilla_dense", "full_xsa_proj_dense", "full_xsa_mask_dense",
            "full_vanilla_moe"]

    fig, ax = plt.subplots(figsize=(7.6, 4))
    seeds = collect_arm(data, files, "full_vanilla_dense")
    seed_values = [r[0] for r in seeds]

    arm_values, arm_labels = [], []
    for arm in arms:
        runs = collect_arm(data, files, arm)
        if runs:
            arm_values.append(runs[0][0])
            arm_labels.append(arm.replace("full_", "").replace("_dense", ""))

    ax.scatter([0] * len(seed_values), seed_values, s=60, color=BLUE, zorder=3,
               label=f"same arm, different seeds (n={len(seed_values)})")
    ax.scatter([1] * len(arm_values), arm_values, s=60, color=RED, zorder=3,
               label=f"different arms, seed 0 (n={len(arm_values)})")
    # Arms that scored identically would print their labels on top of each
    # other, so nudge each subsequent one down a little.
    order = sorted(range(len(arm_values)), key=lambda i: -arm_values[i])
    last = None
    for rank, i in enumerate(order):
        y = arm_values[i]
        if last is not None and abs(y - last) < 0.45:
            y = last - 0.45
        ax.annotate(arm_labels[i], (1.05, y), fontsize=7, va="center")
        last = y

    for xi, values in ((0, seed_values), (1, arm_values)):
        if len(values) > 1:
            ax.plot([xi - 0.08, xi + 0.08],
                    [statistics.mean(values)] * 2, color="black", lw=2)

    ax.set_xticks([0, 1], ["seed variance", "architecture variance"])
    ax.set_xlim(-0.4, 1.6)
    ax.set_ylabel("QED symbolic exact match (%)")
    ax.set_title("Changing the seed moves the number more than\n"
                 "changing the architecture does")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/06_seed_vs_architecture.png", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load_all()
    for fn in (fig_headline, fig_protocols, fig_modality, fig_curves,
               fig_decomposition, fig_seed_noise):
        try:
            fn(data)
            print(f"  ok   {fn.__name__}")
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\nwrote {len(glob.glob(OUT + '/*.png'))} figures to {OUT}/")


if __name__ == "__main__":
    main()
