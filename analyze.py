"""Analyze experiment results from train.py JSON logs.

Reads all ``results_*.json`` files from a directory, builds a summary
table, and produces grouped comparison plots:

1. Architecture comparison (grouped bar, full-data only)
2. Box plot — accuracy by modulus type (scattering init only)
3. Box plot — accuracy by depth (scattering init only)
4. Data efficiency curves per architecture family
5. Box plot — fine-tune gain by data fraction
6. Filter delta vs. fine-tune gain (labeled scatter)
7. Paired scattering vs random init bar chart
8. Three-line data efficiency (random < scat-frozen < scat-finetuned)
9. Scattering advantage over random vs data size
10. Per-experiment training curves

Usage
-----
# Print summary table and save plots:
    python analyze.py --results-dir cifar_all_results

# Recursively search subdirectories for results:
    python analyze.py --results-dir cifar_runs --recursive

# Show plots interactively:
    python analyze.py --results-dir cifar_all_results --interactive

# Also plot per-experiment training curves:
    python analyze.py --results-dir cifar_all_results --curves
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_label(row):
    """Derive a short human-readable label from save_dir."""
    sd = row.get("save_dir", "")
    if sd:
        return sd.rstrip("/").split("/")[-1]
    parts = []
    parts.append(f"{row.get('n_blocks', '?')}b")
    mt = row.get("modulus_type", "")
    parts.append("prelu" if mt == "phase_relu" else "cmod")
    return "_".join(parts)


def _data_fraction_label(row):
    """Map train_size to a readable fraction string."""
    ts = row.get("train_size")
    if ts is None or not isinstance(ts, (int, float)) or np.isnan(ts) or ts >= 1.0:
        return "100%"
    return f"{int(ts * 100)}%"


def _is_random(row):
    """Check if an experiment used random initialization."""
    val = row.get("random_init", False)
    if pd.isna(val):
        return False
    return bool(val)


def _arch_key(row):
    """Derive a canonical architecture key ignoring init type and data fraction.

    E.g. both ``3b_prelu`` and ``3b_prelu_rand`` map to
    ``(3, 'phase_relu', 8, 7, None, False, False)``.

    NaN values are replaced with None so that tuple equality works
    (``NaN != NaN`` but ``None == None``).
    """
    def _nan_to_none(val):
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return val

    return (
        _nan_to_none(row.get("n_blocks")),
        row.get("modulus_type"),
        _nan_to_none(row.get("L")),
        _nan_to_none(row.get("kernel_size")),
        _nan_to_none(row.get("mixing_horizon")),
        row.get("global_avg_pool", False),
        row.get("lowpass_last", False),
    )


def _find_matched_pairs(df):
    """Find (scattering, random) experiment pairs with the same architecture.

    Returns a list of dicts with keys: arch_key, scat_row, rand_row,
    data_frac, actual_train_samples.
    """
    pairs = []
    scat = df[~df["_is_random"]]
    rand = df[df["_is_random"]]

    for _, r_row in rand.iterrows():
        key = _arch_key(r_row)
        frac = r_row["data_frac"]
        candidates = scat[(scat["_arch_key"] == key) & (scat["data_frac"] == frac)]
        if candidates.empty:
            continue
        s_row = candidates.iloc[0]
        pairs.append({
            "arch_key": key,
            "label": s_row["label"],
            "scat": s_row,
            "rand": r_row,
            "data_frac": frac,
            "actual_train_samples": r_row.get("actual_train_samples"),
        })
    return pairs


def _boxplot_with_strip(ax, groups, positions, colors, width=0.6):
    """Draw box plots with overlaid individual points.

    Parameters
    ----------
    groups : list of array-like
        Data arrays, one per box.
    positions : array-like
        X positions for each box.
    colors : list of str
        Face colors for each box.
    """
    bp = ax.boxplot(groups, positions=positions, widths=width,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    rng = np.random.default_rng(42)
    for data, pos in zip(groups, positions):
        if len(data) == 0:
            continue
        jitter = rng.uniform(-width * 0.3, width * 0.3, size=len(data))
        ax.scatter(pos + jitter, data, s=30, zorder=5,
                   edgecolors="black", linewidth=0.5, alpha=0.85)


# ---------------------------------------------------------------------------
# Load experiments
# ---------------------------------------------------------------------------

def load_experiments(results_dir, recursive=False):
    """Read all results_*.json files into a DataFrame.

    Each row is one experiment.  Columns come from merging the ``config``
    and ``results`` dicts.  The raw ``curves`` dict is kept in a hidden
    ``_curves`` column for optional per-experiment plotting.
    """
    if recursive:
        pattern = os.path.join(results_dir, "**", "results_*.json")
        paths = sorted(glob.glob(pattern, recursive=True))
    else:
        pattern = os.path.join(results_dir, "results_*.json")
        paths = sorted(glob.glob(pattern))

    if not paths:
        print(f"No results_*.json files found in {results_dir}"
              f"{' (recursive)' if recursive else ''}")
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
    df["label"] = df.apply(_make_label, axis=1)
    df["data_frac"] = df.apply(_data_fraction_label, axis=1)
    df["_is_random"] = df.apply(_is_random, axis=1)
    df["_arch_key"] = df.apply(lambda r: _arch_key(r), axis=1)
    if "joint_acc" in df.columns and "finetune_acc" in df.columns:
        df["best_acc"] = df["joint_acc"].fillna(df["finetune_acc"])
    elif "joint_acc" in df.columns:
        df["best_acc"] = df["joint_acc"]
    elif "finetune_acc" in df.columns:
        df["best_acc"] = df["finetune_acc"]
    n_scat = (~df["_is_random"]).sum()
    n_rand = df["_is_random"].sum()
    print(f"Loaded {len(df)} experiment(s) from {results_dir}"
          f" ({n_scat} scattering, {n_rand} random)\n")
    return df


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

SUMMARY_COLS = [
    "label", "random_init", "joint", "n_blocks", "L", "modulus_type",
    "kernel_size", "mixing_horizon", "global_avg_pool", "lowpass_last",
    "actual_train_samples",
    "head_acc", "finetune_acc", "finetune_gain", "joint_acc",
    "filter_delta_relative",
    "n_params_extractor", "n_params_classifier",
]


def print_summary(df, save_dir=None):
    cols = [c for c in SUMMARY_COLS if c in df.columns]
    view = df[cols].copy()
    sort_keys = [c for c in ["n_blocks", "modulus_type", "actual_train_samples"]
                 if c in view.columns]
    if sort_keys:
        view = view.sort_values(sort_keys)
    print("=" * 120)
    print("EXPERIMENT SUMMARY")
    print("=" * 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(view.to_string(index=False))
    print()
    if save_dir:
        path = os.path.join(save_dir, "summary.csv")
        view.to_csv(path, index=False)
        print(f"Saved {path}\n")


# ---------------------------------------------------------------------------
# Plot 1: Architecture comparison — grouped bar (full-data only)
# ---------------------------------------------------------------------------

def plot_arch_comparison(df, save_dir):
    full = df[df["data_frac"] == "100%"].copy()
    if full.empty:
        return None

    has_head = "head_acc" in full.columns and full["head_acc"].notna().any()
    has_ft = "finetune_acc" in full.columns and full["finetune_acc"].notna().any()
    has_joint = "joint_acc" in full.columns and full["joint_acc"].notna().any()
    if not (has_head or has_joint):
        return None

    full = full.sort_values(["n_blocks", "modulus_type"])
    labels = full["label"].values
    x = np.arange(len(labels))

    n_bars = sum([has_head, has_ft, has_joint])
    width = 0.8 / max(n_bars, 1)
    offsets = np.linspace(-(n_bars - 1) * width / 2,
                          (n_bars - 1) * width / 2, n_bars)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 6))
    bar_idx = 0

    if has_head:
        ax.bar(x + offsets[bar_idx], full["head_acc"].fillna(0), width,
               label="Head only (Phase A)", color="tab:blue", alpha=0.7)
        bar_idx += 1

    if has_ft:
        bars_b = ax.bar(x + offsets[bar_idx], full["finetune_acc"].fillna(0),
                        width, label="Fine-tuned (Phase B)",
                        color="tab:orange", alpha=0.7)
        if "finetune_gain" in full.columns:
            for bar, gain in zip(bars_b, full["finetune_gain"]):
                if pd.notna(gain):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f"+{gain:.1f}" if gain >= 0 else f"{gain:.1f}",
                            ha="center", va="bottom", fontsize=7)
        bar_idx += 1

    if has_joint:
        ax.bar(x + offsets[bar_idx], full["joint_acc"].fillna(0), width,
               label="Joint training", color="tab:green", alpha=0.7)
        bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Architecture Comparison (full training data)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "plot_arch_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Box plot — accuracy by modulus type
# ---------------------------------------------------------------------------

def plot_box_modulus(df, save_dir):
    scat = df[~df["_is_random"]]
    if "modulus_type" not in scat.columns or "head_acc" not in scat.columns:
        return None

    types = sorted(scat["modulus_type"].dropna().unique())
    if len(types) < 2:
        return None

    has_ft = "finetune_acc" in scat.columns and scat["finetune_acc"].notna().any()
    fig, ax = plt.subplots(figsize=(8, 5))

    groups_head = [scat.loc[scat["modulus_type"] == t, "head_acc"].dropna().values
                   for t in types]

    if has_ft:
        groups_ft = [scat.loc[scat["modulus_type"] == t, "finetune_acc"].dropna().values
                     for t in types]
        pos_h = np.arange(len(types)) * 2.5
        pos_f = pos_h + 0.8
        _boxplot_with_strip(ax, groups_head, pos_h,
                            ["tab:blue"] * len(types))
        _boxplot_with_strip(ax, groups_ft, pos_f,
                            ["tab:orange"] * len(types))
        ax.set_xticks(pos_h + 0.4)
        ax.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.5),
            plt.Rectangle((0, 0), 1, 1, fc="tab:orange", alpha=0.5),
        ], labels=["Head (A)", "Fine-tuned (B)"])
    else:
        pos_h = np.arange(len(types))
        _boxplot_with_strip(ax, groups_head, pos_h,
                            ["tab:blue"] * len(types))
        ax.set_xticks(pos_h)

    ax.set_xticklabels(types)
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Accuracy by Modulus Type (scattering init only)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "plot_box_modulus.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Box plot — accuracy by depth
# ---------------------------------------------------------------------------

def plot_box_depth(df, save_dir):
    scat = df[~df["_is_random"]]
    if "n_blocks" not in scat.columns or "head_acc" not in scat.columns:
        return None

    depths = sorted(scat["n_blocks"].dropna().unique())
    if len(depths) < 2:
        return None

    has_ft = "finetune_acc" in scat.columns and scat["finetune_acc"].notna().any()
    fig, ax = plt.subplots(figsize=(8, 5))

    groups_head = [scat.loc[scat["n_blocks"] == d, "head_acc"].dropna().values
                   for d in depths]

    if has_ft:
        groups_ft = [scat.loc[scat["n_blocks"] == d, "finetune_acc"].dropna().values
                     for d in depths]
        pos_h = np.arange(len(depths)) * 2.5
        pos_f = pos_h + 0.8
        _boxplot_with_strip(ax, groups_head, pos_h,
                            ["tab:blue"] * len(depths))
        _boxplot_with_strip(ax, groups_ft, pos_f,
                            ["tab:orange"] * len(depths))
        ax.set_xticks(pos_h + 0.4)
        ax.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.5),
            plt.Rectangle((0, 0), 1, 1, fc="tab:orange", alpha=0.5),
        ], labels=["Head (A)", "Fine-tuned (B)"])
    else:
        pos_h = np.arange(len(depths))
        _boxplot_with_strip(ax, groups_head, pos_h,
                            ["tab:blue"] * len(depths))
        ax.set_xticks(pos_h)

    ax.set_xticklabels([str(int(d)) for d in depths])
    ax.set_xlabel("Number of blocks")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Accuracy by Network Depth (scattering init only)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "plot_box_depth.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Data efficiency curves per architecture family
# ---------------------------------------------------------------------------

def _identify_families(df):
    """Find architecture families that were tested at multiple data sizes.

    Returns a dict mapping family_label -> sub-DataFrame sorted by sample count.
    """
    if "actual_train_samples" not in df.columns:
        return {}

    families = {}
    for label, grp in df.groupby("label"):
        if len(grp) > 1:
            families[label] = grp.sort_values("actual_train_samples")
            continue
        row = grp.iloc[0]
        ts = row.get("train_size")
        if ts is not None and ts < 1.0:
            base = label.rsplit("_s", 1)[0]
            families.setdefault(base, pd.DataFrame())
            families[base] = pd.concat([families[base], grp])

    full = df[df["data_frac"] == "100%"]
    for label, grp in families.items():
        if "100%" not in grp["data_frac"].values:
            match = full[full["label"] == label]
            if not match.empty:
                families[label] = pd.concat([grp, match])

    return {k: v.sort_values("actual_train_samples").drop_duplicates("actual_train_samples")
            for k, v in families.items() if len(v) >= 2}


def plot_data_efficiency(df, save_dir):
    families = _identify_families(df)
    if not families:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")

    for i, (name, grp) in enumerate(sorted(families.items())):
        color = cmap(i % 10)
        samples = grp["actual_train_samples"].values

        ax.plot(samples, grp["head_acc"].values, "o--",
                color=color, alpha=0.6, label=f"{name} head")
        if "finetune_acc" in grp.columns and grp["finetune_acc"].notna().any():
            ax.plot(samples, grp["finetune_acc"].values, "s-",
                    color=color, label=f"{name} fine-tuned")

    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Data Efficiency by Architecture Family")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "plot_data_efficiency.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 5: Box plot — fine-tune gain by data fraction
# ---------------------------------------------------------------------------

def plot_box_gain(df, save_dir):
    if "finetune_gain" not in df.columns:
        return None

    sub = df.dropna(subset=["finetune_gain"])
    if sub.empty:
        return None

    frac_order = ["10%", "30%", "50%", "100%"]
    fracs = [f for f in frac_order if f in sub["data_frac"].values]
    if not fracs:
        return None

    groups = [sub.loc[sub["data_frac"] == f, "finetune_gain"].values
              for f in fracs]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(fracs))
    _boxplot_with_strip(ax, groups, positions, ["tab:green"] * len(fracs))

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(fracs)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Fine-tune gain (%)")
    ax.set_title("Fine-Tuning Gain by Data Fraction")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "plot_box_gain.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 6: Filter delta vs. fine-tune gain (labeled scatter)
# ---------------------------------------------------------------------------

def plot_delta_vs_gain(df, save_dir):
    sub = df.dropna(subset=["filter_delta_relative", "finetune_gain"])
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sub["filter_delta_relative"], sub["finetune_gain"],
               s=60, edgecolors="black", linewidth=0.5, zorder=5)

    for _, row in sub.iterrows():
        ax.annotate(row["label"],
                    (row["filter_delta_relative"], row["finetune_gain"]),
                    fontsize=6, alpha=0.75,
                    xytext=(4, 4), textcoords="offset points")

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
# Plot 7: Paired scattering vs random bar chart
# ---------------------------------------------------------------------------

def plot_init_comparison(df, save_dir):
    """Grouped bar: for each matched architecture, show scattering head_acc,
    scattering finetune_acc, and random joint_acc side-by-side."""
    pairs = _find_matched_pairs(df)
    full_pairs = [p for p in pairs if p["data_frac"] == "100%"]
    if not full_pairs:
        return None

    labels = [p["label"] for p in full_pairs]
    head_vals = [p["scat"].get("head_acc", 0) or 0 for p in full_pairs]
    ft_vals = [p["scat"].get("finetune_acc", 0) or 0 for p in full_pairs]
    rand_vals = [p["rand"].get("joint_acc", 0) or 0 for p in full_pairs]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.2), 6))
    ax.bar(x - width, head_vals, width,
           label="Scattering (head only)", color="tab:blue", alpha=0.7)
    ax.bar(x, ft_vals, width,
           label="Scattering (fine-tuned)", color="tab:orange", alpha=0.7)
    ax.bar(x + width, rand_vals, width,
           label="Random init (joint)", color="tab:red", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Scattering Init vs Random Init (full training data)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "plot_init_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 8: Three-line data efficiency (random < scat-frozen < scat-finetuned)
# ---------------------------------------------------------------------------

def _base_arch_label(label):
    """Strip data-fraction suffix (_s01, _s03, etc.) to get the base name."""
    return re.sub(r"_s\d+$", "", label)


def _build_init_families(df):
    """Build scattering and random data-efficiency families for matching archs.

    Returns a dict mapping base_label -> {
        "scat_samples": [...], "scat_head": [...], "scat_ft": [...],
        "rand_samples": [...], "rand_joint": [...]
    }
    """
    pairs = _find_matched_pairs(df)
    if not pairs:
        return {}

    families = {}
    for p in pairs:
        base = _base_arch_label(p["label"])
        families.setdefault(base, {
            "scat_samples": [], "scat_head": [], "scat_ft": [],
            "rand_samples": [], "rand_joint": [],
        })
        f = families[base]
        s = p["actual_train_samples"]
        f["scat_samples"].append(s)
        f["scat_head"].append(p["scat"].get("head_acc"))
        f["scat_ft"].append(p["scat"].get("finetune_acc"))
        f["rand_samples"].append(s)
        f["rand_joint"].append(p["rand"].get("joint_acc"))

    result = {}
    for base, f in families.items():
        if len(f["scat_samples"]) < 2:
            continue
        order = np.argsort(f["scat_samples"])
        for key in f:
            f[key] = [f[key][i] for i in order]
        result[base] = f
    return result


def plot_init_data_efficiency(df, save_dir):
    """Three-line plot per architecture family: random joint < scat head < scat fine-tuned."""
    families = _build_init_families(df)
    if not families:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")

    for i, (name, f) in enumerate(sorted(families.items())):
        color = cmap(i % 10)
        ax.plot(f["scat_samples"], f["scat_head"], "^--",
                color=color, alpha=0.6, label=f"{name} scat head")
        ax.plot(f["scat_samples"], f["scat_ft"], "s-",
                color=color, label=f"{name} scat fine-tuned")
        ax.plot(f["rand_samples"], f["rand_joint"], "x:",
                color=color, alpha=0.6, markersize=8,
                label=f"{name} random joint")

    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Data Efficiency: Scattering Init vs Random Init")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "plot_init_data_efficiency.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 9: Scattering advantage over random vs data size
# ---------------------------------------------------------------------------

def plot_init_advantage(df, save_dir):
    """Line chart: accuracy advantage (scattering fine-tuned minus random joint)
    as a function of training set size, per architecture family."""
    families = _build_init_families(df)
    if not families:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")

    for i, (name, f) in enumerate(sorted(families.items())):
        color = cmap(i % 10)
        samples = f["scat_samples"]
        advantage = [
            (ft or 0) - (rj or 0)
            for ft, rj in zip(f["scat_ft"], f["rand_joint"])
        ]
        ax.plot(samples, advantage, "o-", color=color, label=name)
        for s, adv in zip(samples, advantage):
            ax.annotate(f"{adv:+.1f}", (s, adv), fontsize=7,
                        textcoords="offset points", xytext=(0, 6),
                        ha="center")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Accuracy advantage (%)")
    ax.set_title("Scattering Init Advantage over Random Init")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "plot_init_advantage.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 10: Per-experiment training curves
# ---------------------------------------------------------------------------

def plot_training_curves(df, save_dir):
    curves_dir = os.path.join(save_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    count = 0

    for _, row in df.iterrows():
        curves = row.get("_curves", {})
        if not curves:
            continue

        phases = [p for p in ["phase_a", "phase_b", "joint"] if curves.get(p)]
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

        label = row.get("label", "unknown")
        samples = row.get("actual_train_samples", "?")
        fig.suptitle(f"{label} ({samples} samples)", fontsize=11)
        fig.tight_layout()

        path = os.path.join(curves_dir, f"curves_{label}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        count += 1

    print(f"Saved {count} training curve(s) -> {curves_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Analyze experiment results")
    p.add_argument("--results-dir", default="runs",
                   help="directory containing results_*.json files")
    p.add_argument("--recursive", action="store_true",
                   help="search subdirectories for results_*.json")
    p.add_argument("--interactive", action="store_true",
                   help="show plots interactively (plt.show)")
    p.add_argument("--curves", action="store_true",
                   help="also plot per-experiment training curves")
    args = p.parse_args()

    df = load_experiments(args.results_dir, recursive=args.recursive)
    print_summary(df, save_dir=args.results_dir)

    plot_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_arch_comparison(df, plot_dir)
    plot_box_modulus(df, plot_dir)
    plot_box_depth(df, plot_dir)
    plot_data_efficiency(df, plot_dir)
    plot_box_gain(df, plot_dir)
    plot_delta_vs_gain(df, plot_dir)
    plot_init_comparison(df, plot_dir)
    plot_init_data_efficiency(df, plot_dir)
    plot_init_advantage(df, plot_dir)

    if args.curves:
        plot_training_curves(df, plot_dir)

    if args.interactive:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
