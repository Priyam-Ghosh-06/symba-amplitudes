"""Figures and README tables from results/*.json.

    python scripts/plot_results.py

Writes results/figures/*.png and prints the tables in markdown.
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
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from symba.data.serialize import from_prefix, token_type

THEORIES = ("QED", "QCD")
VARIANTS = {"full": "full model", "graph_only": "graph encoder only",
            "ast_only": "amplitude encoder only", "no_role_filler": "plain embedding",
            "no_moe": "dense FFN", "no_xsa": "vanilla attention"}
COLOR = {"QED": "#2E86AB", "QCD": "#E84855"}
FIGURES = "results/figures"

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def load_runs():
    runs = {}
    for path in sorted(glob.glob("results/*.json")):
        with open(path, encoding="utf-8") as fh:
            run = json.load(fh)
        runs[run["theory"], run["variant"]] = run
    return runs


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/{name}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES}/{name}")


def training_curves(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, theory in zip(axes, THEORIES):
        run = runs[theory, "full"]
        history = run["history"]
        epochs = [r["epoch"] for r in history]
        ax.plot(epochs, [r["train_loss"] for r in history], color="#9AA0A6", label="train loss")
        ax.plot(epochs, [r["val_loss"] for r in history], color=COLOR[theory], label="val loss")
        ax.set(yscale="log", xlabel="epoch", ylabel="cross-entropy", title=theory)
        acc = ax.twinx()
        points = [(r["epoch"], 100 * r["val_sequence_accuracy"])
                  for r in history if "val_sequence_accuracy" in r]
        acc.plot(*zip(*points), "o-", ms=3, color="black", label="val sequence accuracy")
        acc.set(ylim=(0, 105), ylabel="val sequence accuracy (%)")
        acc.spines["right"].set_visible(True)
        ax.axvline(run["best_epoch"], ls=":", color="grey")
        lines = ax.get_lines()[:2] + acc.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=7)
    save(fig, "training_curves.png")


def ablations(runs):
    names = [v for v in VARIANTS if any((t, v) in runs for t in THEORIES)]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.2))
    for k, theory in enumerate(THEORIES):
        present = [(i, 100 * runs[theory, v]["test"]["sequence_accuracy"])
                   for i, v in enumerate(names) if (theory, v) in runs]
        bars = ax.barh([i + (k - 0.5) * 0.38 for i, _ in present], [v for _, v in present],
                       height=0.38, color=COLOR[theory],
                       label=f"{theory} ({runs[theory, 'full']['test']['n']} test records)")
        for bar, (_, value) in zip(bars, present):
            ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}",
                    va="center", fontsize=7)
    ax.set_yticks(range(len(names)), [VARIANTS[v] for v in names])
    ax.invert_yaxis()
    ax.set(xlim=(0, 112), xlabel="test sequence accuracy (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    save(fig, "ablations.png")


def accuracy_vs_length(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, theory in zip(axes, THEORIES):
        preds = runs[theory, "full"]["predictions"]
        lengths = [len(p["reference"].split()) for p in preds]
        right = [n for n, p in zip(lengths, preds) if p["correct"]]
        wrong = [n for n, p in zip(lengths, preds) if not p["correct"]]
        bins = np.linspace(min(lengths), max(lengths) + 1, 9)
        ax.hist([right, wrong], bins=bins, stacked=True, color=[COLOR[theory], "#BBBBBB"],
                label=[f"correct ({len(right)})", f"wrong ({len(wrong)})"])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlabel="target length (tokens)", ylabel="test records", title=theory)
        ax.legend(fontsize=7)
    save(fig, "accuracy_vs_length.png")


def cross_attention(runs):
    """Where the decoder looks in the Feynman graph while writing each physical
    symbol of the answer (operator and digit rows are left out)."""
    symbols = {"MASS", "MANDELSTAM", "COUPLING", "REGPROP"}
    panels = []
    for theory in THEORIES:
        ex = runs[theory, "full"]["attention_example"]
        rows = [i for i, t in enumerate(ex["output_tokens"]) if token_type(t) in symbols]
        panels.append((theory, ex, rows))
    height = 0.22 * max(len(rows) for _, _, rows in panels) + 2
    fig, axes = plt.subplots(1, 2, figsize=(11, height))
    for ax, (theory, ex, rows) in zip(axes, panels):
        image = ax.imshow(np.array(ex["weights"])[rows], cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(ex["graph_tokens"])), ex["graph_tokens"], rotation=90, fontsize=6)
        ax.set_yticks(range(len(rows)), [ex["output_tokens"][i] for i in rows], fontsize=7)
        ax.set(xlabel="graph token", ylabel="symbol being written",
               title=f"{theory}: {100 * ex['graph_share']:.0f}% of attention on the graph")
        fig.colorbar(image, ax=ax, fraction=0.04, label="share of graph attention")
    save(fig, "cross_attention.png")


def expert_routing(runs):
    routed = [(t, runs[t, "full"]["expert_routing"]) for t in THEORIES]
    n_layers = len(routed[0][1]["counts"])
    fig, axes = plt.subplots(len(routed), n_layers, figsize=(3.2 * n_layers, 2.6 * len(routed)))
    for row, (theory, routing) in zip(np.atleast_2d(axes), routed):
        for layer, ax in enumerate(row):
            counts = np.array(routing["counts"][layer], dtype=float)
            used = counts.sum(1) > 0
            share = counts[used] / counts[used].sum(1, keepdims=True)
            ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for (i, j), value in np.ndenumerate(share):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if value > 0.6 else "black")
            ax.set_xticks(range(share.shape[1]), [f"E{j}" for j in range(share.shape[1])])
            ax.set_yticks(range(share.shape[0]),
                          [t for t, u in zip(routing["token_types"], used) if u])
            ax.set_title(f"{theory}, decoder layer {layer + 1}", fontsize=8)
    save(fig, "expert_routing.png")


def pct(x):
    return f"{100 * x:.1f}"


def tables(runs):
    full = {t: runs[t, "full"] for t in THEORIES}
    rows = [
        ("test records", lambda r: str(r["test"]["n"])),
        ("token accuracy (%)", lambda r: pct(r["test"]["token_accuracy"])),
        ("sequence accuracy (%)", lambda r: pct(r["test"]["sequence_accuracy"])),
        ("symbolic accuracy (%)", lambda r: pct(r["test"]["symbolic_accuracy"])),
        ("sequence accuracy, greedy (%)", lambda r: pct(r["test"]["greedy_sequence_accuracy"])),
        ("valid expressions (%)", lambda r: pct(r["test"]["valid_expressions"])),
        ("parameters", lambda r: f"{r['parameters']:,}"),
        ("selected epoch", lambda r: str(r["best_epoch"])),
        ("seconds per epoch (CPU, median)",
         lambda r: f"{statistics.median(h['seconds'] for h in r['history']):.0f}"),
    ]
    print("\n### Results\n\n| | QED | QCD |\n|---|---|---|")
    for label, get in rows:
        print(f"| {label} | {get(full['QED'])} | {get(full['QCD'])} |")

    print("\n### Ablations (test sequence accuracy, %)\n\n| model | QED | QCD |\n|---|---|---|")
    for variant, label in VARIANTS.items():
        cells = [pct(runs[t, variant]["test"]["sequence_accuracy"]) if (t, variant) in runs
                 else "-" for t in THEORIES]
        print(f"| {label} | {cells[0]} | {cells[1]} |")

    print("\n### Dataset\n\n| | QED | QCD |\n|---|---|---|")
    stats = {t: full[t]["data"] for t in THEORIES}
    for label, get in [("records", lambda s: s["records"]),
                       ("train / val / test", lambda s: f"{s['train']} / {s['val']} / {s['test']}"),
                       ("longest raw sq_amp (characters)", lambda s: s["longest"]["sq_amp_characters"]),
                       ("longest canonical target (tokens)", lambda s: s["longest"]["target_tokens"]),
                       ("longest amplitude (tokens)", lambda s: s["longest"]["amp_tokens"]),
                       ("longest single diagram (tokens)", lambda s: s["longest"]["diagram_tokens"]),
                       ("target vocabulary", lambda s: s["vocab"]["target"]),
                       ("per-diagram encoding", lambda s: "yes" if s["segment_amp"] else "no")]:
        print(f"| {label} | {get(stats['QED'])} | {get(stats['QCD'])} |")

    print("\n### Example test predictions\n")
    for theory in THEORIES:
        for p in full[theory]["predictions"]:
            if not p["correct"]:
                print(f"{theory} wrong: {p['file']}:{p['line']}")
                print(f"  predicted: {from_prefix(p['prediction'].split())}")
                print(f"  truth:     {from_prefix(p['reference'].split())}")
        p = min((p for p in full[theory]["predictions"] if p["correct"]),
                key=lambda p: len(p["reference"]))
        print(f"{theory} correct: {p['file']}:{p['line']}")
        print(f"  {from_prefix(p['reference'].split())}")


def main():
    os.chdir(ROOT)
    os.makedirs(FIGURES, exist_ok=True)
    runs = load_runs()
    for figure in (training_curves, ablations, accuracy_vs_length, cross_attention,
                   expert_routing):
        figure(runs)
    tables(runs)


if __name__ == "__main__":
    main()
