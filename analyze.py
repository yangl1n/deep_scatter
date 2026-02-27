"""Analyze experiment results from train.py JSON logs.

Reads all ``results_*.json`` files from a directory, builds a summary
table, and produces the three key plots from PROJECT_GUIDE.md:

1. Training set size vs. accuracy
2. Training set size vs. fine-tune gain
3. Filter delta vs. fine-tune gain

Usage
-----
# Print summary table and save plots:
    python analyze.py --results-dir runs

# Show plots interactively:
    python analyze.py --results-dir runs --interactive

# Also plot per-experiment training curves:
    python analyze.py --results-dir runs --curves
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Load experiments
# ---------------------------------------------------------------------------

def load_experiments(results_dir):
    """Read all results_*.json files into a DataFrame.

    Each row is one experiment. Columns come from merging the ``config``
    and ``results`` dicts.  The raw ``curves`` dict is kept in a hidden
    ``_curves`` column for optional per-experiment plotting.
    """
    pattern = os.path.join(results_dir, "results_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No results_*.json files found in {results_dir}")
        sys.exit(1)

    rows = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        row = {}
        row.update(data.get("config", {}))
        row.update(data.get("results", {}))
        row["_path"] = path
        row["_curves"] = data.get("curves", {})
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} experiment(s) from {results_dir}\n")
    return df


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

SUMMARY_COLS = [
    "dataset", "n_blocks", "L", "modulus_type", "mixing_horizon",
    "actual_train_samples", "head_acc", "baseline_acc",
    "finetune_acc", "finetune_gain",
    "filter_delta_l2", "filter_delta_relative",
    "n_params_extractor", "n_params_classifier",
]


def print_summary(df):
    cols = [c for c in SUMMARY_COLS if c in df.columns]
    view = df[cols].copy()
    if "actual_train_samples" in view.columns:
        view = view.sort_values("actual_train_samples")
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(view.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Plot 1: train size vs. accuracy
# ---------------------------------------------------------------------------

def plot_size_vs_accuracy(df, save_dir):
    if "actual_train_samples" not in df.columns or "head_acc" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = df.sort_values("actual_train_samples")

    ax.plot(grouped["actual_train_samples"], grouped["head_acc"],
            "o-", label="Head only (Phase A)")

    if "finetune_acc" in df.columns and df["finetune_acc"].notna().any():
        ax.plot(grouped["actual_train_samples"], grouped["finetune_acc"],
                "s-", label="Fine-tuned (Phase B)")

    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Training Set Size vs. Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "plot_size_vs_accuracy.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 2: train size vs. fine-tune gain
# ---------------------------------------------------------------------------

def plot_size_vs_gain(df, save_dir):
    sub = df.dropna(subset=["finetune_gain"])
    if sub.empty:
        return None

    sub = sub.sort_values("actual_train_samples")
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = sub["actual_train_samples"].astype(str)
    gains = sub["finetune_gain"]
    colors = ["tab:green" if g >= 0 else "tab:red" for g in gains]

    ax.bar(labels, gains, color=colors)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Fine-tune gain (%)")
    ax.set_title("Fine-Tuning Gain by Training Set Size")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "plot_size_vs_gain.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 3: filter delta vs. fine-tune gain
# ---------------------------------------------------------------------------

def plot_delta_vs_gain(df, save_dir):
    sub = df.dropna(subset=["filter_delta_relative", "finetune_gain"])
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sub["filter_delta_relative"], sub["finetune_gain"],
               s=60, edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Relative filter delta (L2)")
    ax.set_ylabel("Fine-tune gain (%)")
    ax.set_title("Filter Change vs. Accuracy Improvement")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)

    if len(sub) >= 3:
        r = sub[["filter_delta_relative", "finetune_gain"]].corr().iloc[0, 1]
        ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.95),
                    xycoords="axes fraction", fontsize=11)

    fig.tight_layout()
    path = os.path.join(save_dir, "plot_delta_vs_gain.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Per-experiment training curves
# ---------------------------------------------------------------------------

def plot_training_curves(df, save_dir):
    curves_dir = os.path.join(save_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)

    for i, row in df.iterrows():
        curves = row.get("_curves", {})
        if not curves:
            continue

        phases = [p for p in ["phase_a", "phase_b"] if curves.get(p)]
        if not phases:
            continue

        fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 4))
        if len(phases) == 1:
            axes = [axes]

        for phase, ax in zip(phases, axes):
            entries = curves[phase]
            epochs = [e["epoch"] for e in entries]
            ax.plot(epochs, [e["val_acc"] for e in entries],
                    label="val_acc", color="tab:blue")
            ax2 = ax.twinx()
            ax2.plot(epochs, [e["train_loss"] for e in entries],
                     label="train_loss", color="tab:orange", alpha=0.7)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val Acc (%)", color="tab:blue")
            ax2.set_ylabel("Train Loss", color="tab:orange")
            ax.set_title(phase.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

        samples = row.get("actual_train_samples", "?")
        fig.suptitle(f"Experiment {i} — {samples} samples", fontsize=11)
        fig.tight_layout()

        path = os.path.join(curves_dir, f"curves_exp{i}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)

    print(f"Saved training curves -> {curves_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Analyze experiment results")
    p.add_argument("--results-dir", default="runs",
                   help="directory containing results_*.json files")
    p.add_argument("--interactive", action="store_true",
                   help="show plots interactively (plt.show)")
    p.add_argument("--curves", action="store_true",
                   help="also plot per-experiment training curves")
    args = p.parse_args()

    df = load_experiments(args.results_dir)
    print_summary(df)

    plot_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_size_vs_accuracy(df, plot_dir)
    plot_size_vs_gain(df, plot_dir)
    plot_delta_vs_gain(df, plot_dir)

    if args.curves:
        plot_training_curves(df, plot_dir)

    if args.interactive:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
