"""Analyze experiment results from train.py JSON logs.

Reads all ``results_*.json`` files from ``cifar_exps/`` (or another directory),
builds a summary table, and produces comparison plots:

 1. Architecture comparison (grouped bar, full-data)
 2. Box plot -- accuracy by modulus type
 3. Box plot -- accuracy by depth
 4. Box plot -- fine-tune gain by data fraction
 5. Filter delta vs fine-tune gain (scatter)
 6. Data efficiency (three-line: random < scat-frozen < scat-tuned)
 7. Scattering advantage over random vs data size
 8. Heatmap (architecture x data fraction)
 9. L2 penalty effect
10. Paired difference (scat - random) horizontal bars
11. Mixing horizon comparison (4-block only)
12. Best architecture per data fraction
13. Per-experiment training curves (optional)

Usage
-----
    python analyze.py --results-dir cifar_exps --recursive
    python analyze.py --results-dir cifar_exps --recursive --curves --interactive
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FRAC_ORDER = ["10%", "30%", "50%", "100%"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_label(row):
    """Last component of save_dir, e.g. '3b_prelu_4L_s01'."""
    return row.get("save_dir", "").rstrip("/").split("/")[-1]


def _data_fraction_label(row):
    ts = row.get("train_size")
    if ts is None or (isinstance(ts, float) and np.isnan(ts)) or ts >= 1.0:
        return "100%"
    return f"{int(ts * 100)}%"


def _is_random(row):
    return bool(row.get("random_init", False))


def _arch_key(row):
    """Canonical architecture tuple (ignores init type, data fraction, l2)."""
    mh = row.get("mixing_horizon")
    if isinstance(mh, float) and np.isnan(mh):
        mh = None
    return (int(row["n_blocks"]), row["modulus_type"],
            int(row["L"]), mh)


def _arch_short_label(row):
    """Short label like '3b_prelu_4L' or '4b_cmod_8L_h27'."""
    nb = int(row["n_blocks"])
    mod = "prelu" if row["modulus_type"] == "phase_relu" else "cmod"
    lv = int(row["L"])
    mh = row.get("mixing_horizon")
    if isinstance(mh, float) and np.isnan(mh):
        mh = None
    if mh is not None:
        return f"{nb}b_{mod}_{lv}L_h{int(mh)}"
    return f"{nb}b_{mod}_{lv}L"


def _find_matched_pairs(df):
    """Match each random experiment to the best scattering experiment
    (highest finetune_acc across l2 settings) with the same arch + data_frac.

    Returns list of dicts: arch_key, label, scat, rand, data_frac,
    actual_train_samples.
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
        valid = candidates[candidates["finetune_acc"].notna()]
        s_row = (valid.loc[valid["finetune_acc"].idxmax()]
                 if not valid.empty else candidates.iloc[0])
        pairs.append({
            "arch_key": key,
            "label": s_row["_arch_label"],
            "scat": s_row,
            "rand": r_row,
            "data_frac": frac,
            "actual_train_samples": r_row.get("actual_train_samples"),
        })
    return pairs


def _boxplot_with_strip(ax, groups, positions, colors, width=0.6):
    """Box plots with overlaid jittered points."""
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


def _save(fig, save_dir, name):
    path = os.path.join(save_dir, name)
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Load & summarize
# ---------------------------------------------------------------------------

def load_experiments(results_dir, recursive=False):
    """Read all results_*.json into a DataFrame."""
    pattern = os.path.join(results_dir, "**" if recursive else "",
                           "results_*.json")
    paths = sorted(glob.glob(pattern, recursive=recursive))

    if not paths:
        print(f"No results_*.json found in {results_dir}"
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
    df["_arch_key"] = df.apply(_arch_key, axis=1)
    df["_arch_label"] = df.apply(_arch_short_label, axis=1)

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


SUMMARY_COLS = [
    "label", "random_init", "joint", "n_blocks", "L", "modulus_type",
    "mixing_horizon", "l2_penalty", "actual_train_samples",
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
# Plot 1: Architecture comparison (grouped bar, full-data)
# ---------------------------------------------------------------------------

def plot_arch_comparison(df, save_dir):
    full = df[df["data_frac"] == "100%"].copy()
    if full.empty:
        return None

    scat_all = full[~full["_is_random"]]
    # Pick the best l2 variant per architecture
    best_idx = scat_all.groupby("_arch_label")["finetune_acc"].idxmax()
    scat = scat_all.loc[best_idx].sort_values(["n_blocks", "modulus_type"])
    rand = full[full["_is_random"]].sort_values(["n_blocks", "modulus_type"])

    labels = scat["_arch_label"].values
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.0), 6))

    if "head_acc" in scat.columns:
        ax.bar(x - width, scat["head_acc"].fillna(0), width,
               label="Scat head (Phase A)", color="tab:blue", alpha=0.7)
    if "finetune_acc" in scat.columns:
        bars = ax.bar(x, scat["finetune_acc"].fillna(0), width,
                      label="Scat fine-tuned (Phase B)", color="tab:orange", alpha=0.7)
        if "finetune_gain" in scat.columns:
            for bar, gain in zip(bars, scat["finetune_gain"]):
                if pd.notna(gain):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f"+{gain:.1f}" if gain >= 0 else f"{gain:.1f}",
                            ha="center", va="bottom", fontsize=7)

    rand_by_arch = rand.set_index("_arch_label")
    rand_vals = [rand_by_arch.loc[al, "joint_acc"]
                 if al in rand_by_arch.index else 0
                 for al in labels]
    ax.bar(x + width, rand_vals, width,
           label="Random (joint)", color="tab:red", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Architecture Comparison (full training data)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_arch_comparison.png")


# ---------------------------------------------------------------------------
# Plot 2: Box plot -- accuracy by modulus type
# ---------------------------------------------------------------------------

def plot_box_modulus(df, save_dir):
    scat = df[~df["_is_random"]]
    types = sorted(scat["modulus_type"].dropna().unique())
    if len(types) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    groups_head = [scat.loc[scat["modulus_type"] == t, "head_acc"].dropna().values
                   for t in types]
    groups_ft = [scat.loc[scat["modulus_type"] == t, "finetune_acc"].dropna().values
                 for t in types]

    pos_h = np.arange(len(types)) * 2.5
    pos_f = pos_h + 0.8
    _boxplot_with_strip(ax, groups_head, pos_h, ["tab:blue"] * len(types))
    _boxplot_with_strip(ax, groups_ft, pos_f, ["tab:orange"] * len(types))
    ax.set_xticks(pos_h + 0.4)
    ax.set_xticklabels(types)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, fc="tab:orange", alpha=0.5),
    ], labels=["Head (A)", "Fine-tuned (B)"])
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Accuracy by Modulus Type (scattering init)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_box_modulus.png")


# ---------------------------------------------------------------------------
# Plot 3: Box plot -- accuracy by depth
# ---------------------------------------------------------------------------

def plot_box_depth(df, save_dir):
    scat = df[~df["_is_random"]]
    depths = sorted(scat["n_blocks"].dropna().unique())
    if len(depths) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    groups_head = [scat.loc[scat["n_blocks"] == d, "head_acc"].dropna().values
                   for d in depths]
    groups_ft = [scat.loc[scat["n_blocks"] == d, "finetune_acc"].dropna().values
                 for d in depths]

    pos_h = np.arange(len(depths)) * 2.5
    pos_f = pos_h + 0.8
    _boxplot_with_strip(ax, groups_head, pos_h, ["tab:blue"] * len(depths))
    _boxplot_with_strip(ax, groups_ft, pos_f, ["tab:orange"] * len(depths))
    ax.set_xticks(pos_h + 0.4)
    ax.set_xticklabels([str(int(d)) for d in depths])
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, fc="tab:orange", alpha=0.5),
    ], labels=["Head (A)", "Fine-tuned (B)"])
    ax.set_xlabel("Number of blocks")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Accuracy by Network Depth (scattering init)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_box_depth.png")


# ---------------------------------------------------------------------------
# Plot 4: Box plot -- fine-tune gain by data fraction
# ---------------------------------------------------------------------------

def plot_box_gain(df, save_dir):
    sub = df.dropna(subset=["finetune_gain"])
    if sub.empty:
        return None

    fracs = [f for f in FRAC_ORDER if f in sub["data_frac"].values]
    if not fracs:
        return None

    groups = [sub.loc[sub["data_frac"] == f, "finetune_gain"].values for f in fracs]
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
    return _save(fig, save_dir, "plot_box_gain.png")


# ---------------------------------------------------------------------------
# Plot 5: Filter delta vs fine-tune gain (scatter)
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
    return _save(fig, save_dir, "plot_delta_vs_gain.png")


# ---------------------------------------------------------------------------
# Plot 6: Data efficiency (three-line per architecture family)
# ---------------------------------------------------------------------------

def _build_init_families(df):
    """Group matched (scat, random) pairs by architecture into families.

    Returns dict: arch_label -> {scat_samples, scat_head, scat_ft,
    rand_samples, rand_joint}, sorted by sample count.
    """
    pairs = _find_matched_pairs(df)
    if not pairs:
        return {}

    families = {}
    for p in pairs:
        arch = p["label"]
        families.setdefault(arch, {
            "scat_samples": [], "scat_head": [], "scat_ft": [],
            "rand_samples": [], "rand_joint": [],
        })
        f = families[arch]
        s = p["actual_train_samples"]
        f["scat_samples"].append(s)
        f["scat_head"].append(p["scat"].get("head_acc"))
        f["scat_ft"].append(p["scat"].get("finetune_acc"))
        f["rand_samples"].append(s)
        f["rand_joint"].append(p["rand"].get("joint_acc"))

    result = {}
    for name, f in families.items():
        if len(f["scat_samples"]) < 2:
            continue
        order = np.argsort(f["scat_samples"])
        for key in f:
            f[key] = [f[key][i] for i in order]
        result[name] = f
    return result


def plot_data_efficiency(df, save_dir):
    """Three-line per family: random joint, scat head, scat fine-tuned."""
    families = _build_init_families(df)
    if not families:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
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
    ax.legend(fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_data_efficiency.png")


# ---------------------------------------------------------------------------
# Plot 7: Scattering advantage over random vs data size
# ---------------------------------------------------------------------------

def plot_init_advantage(df, save_dir):
    """Line chart: (scat fine-tuned - random joint) vs training samples."""
    families = _build_init_families(df)
    if not families:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for i, (name, f) in enumerate(sorted(families.items())):
        color = cmap(i % 10)
        advantage = [(ft or 0) - (rj or 0)
                     for ft, rj in zip(f["scat_ft"], f["rand_joint"])]
        ax.plot(f["scat_samples"], advantage, "o-", color=color, label=name)
        for s, adv in zip(f["scat_samples"], advantage):
            ax.annotate(f"{adv:+.1f}", (s, adv), fontsize=7,
                        textcoords="offset points", xytext=(0, 6), ha="center")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Accuracy advantage (%)")
    ax.set_title("Scattering Init Advantage over Random Init")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_init_advantage.png")


# ---------------------------------------------------------------------------
# Plot 8: Heatmap -- architecture x data fraction
# ---------------------------------------------------------------------------

def plot_heatmap(df, save_dir):
    """Side-by-side heatmaps: scattering best finetune_acc and random joint_acc."""
    fracs = [f for f in FRAC_ORDER if f in df["data_frac"].values]
    if not fracs:
        return None

    scat = df[~df["_is_random"]]
    rand = df[df["_is_random"]]

    arch_labels = sorted(scat["_arch_label"].unique(),
                         key=lambda x: (int(x[0]), x))
    if not arch_labels:
        return None

    scat_best = scat.groupby(["_arch_label", "data_frac"])["finetune_acc"].max()
    rand_best = rand.groupby(["_arch_label", "data_frac"])["joint_acc"].max()

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                    figsize=(14, max(5, len(arch_labels) * 0.4)))

    for ax, title, data in [
        (ax1, "Scattering (best fine-tuned)", scat_best),
        (ax2, "Random (joint)", rand_best),
    ]:
        mat = np.full((len(arch_labels), len(fracs)), np.nan)
        for i, al in enumerate(arch_labels):
            for j, fr in enumerate(fracs):
                try:
                    mat[i, j] = data.loc[(al, fr)]
                except KeyError:
                    pass

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels(fracs)
        ax.set_yticks(range(len(arch_labels)))
        ax.set_yticklabels(arch_labels, fontsize=8)
        ax.set_xlabel("Data fraction")
        ax.set_title(title)

        vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 1
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=7,
                            color="black" if v < vmax * 0.8 else "white")

        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Accuracy Heatmap: Architecture x Data Fraction", fontsize=13)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 9: L2 penalty effect (scattering init only)
# ---------------------------------------------------------------------------

def plot_l2_effect(df, save_dir):
    """Box plot: accuracy delta (l2=0.005 minus l2=0) grouped by data fraction."""
    scat = df[~df["_is_random"]]
    if "l2_penalty" not in scat.columns:
        return None

    fracs = [f for f in FRAC_ORDER if f in scat["data_frac"].values]
    if not fracs:
        return None

    deltas_by_frac = {}
    for frac in fracs:
        sub = scat[scat["data_frac"] == frac]
        deltas = []
        for al in sub["_arch_label"].unique():
            arch_sub = sub[sub["_arch_label"] == al]
            l2_0 = arch_sub[arch_sub["l2_penalty"] == 0]["finetune_acc"]
            l2_on = arch_sub[arch_sub["l2_penalty"] > 0]["finetune_acc"]
            if not l2_0.empty and not l2_on.empty:
                deltas.append(l2_on.max() - l2_0.max())
        deltas_by_frac[frac] = np.array(deltas) if deltas else np.array([])

    non_empty = [f for f in fracs if len(deltas_by_frac[f]) > 0]
    if not non_empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [deltas_by_frac[f] for f in non_empty]
    positions = np.arange(len(non_empty))
    _boxplot_with_strip(ax, groups, positions, ["tab:purple"] * len(non_empty))

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(non_empty)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Accuracy delta (l2=0.005 minus l2=0)")
    ax.set_title("L2 Anchor Penalty Effect on Scattering Init")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_l2_effect.png")


# ---------------------------------------------------------------------------
# Plot 10: Paired difference (scattering - random)
# ---------------------------------------------------------------------------

def plot_paired_diff(df, save_dir):
    """Horizontal bar: scat_best - random for every arch x fraction pair."""
    pairs = _find_matched_pairs(df)
    if not pairs:
        return None

    entries = []
    for p in pairs:
        scat_acc = p["scat"].get("finetune_acc", 0) or 0
        rand_acc = p["rand"].get("joint_acc", 0) or 0
        entries.append({
            "label": f"{p['label']} ({p['data_frac']})",
            "diff": scat_acc - rand_acc,
        })

    entries.sort(key=lambda e: e["diff"])
    labels = [e["label"] for e in entries]
    diffs = [e["diff"] for e in entries]
    colors = ["tab:blue" if d >= 0 else "tab:red" for d in diffs]

    fig, ax = plt.subplots(figsize=(10, max(5, len(entries) * 0.3)))
    y = np.arange(len(entries))
    ax.barh(y, diffs, color=colors, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Accuracy difference (scattering - random)")
    ax.set_title("Scattering vs Random Init: Paired Differences")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_paired_diff.png")


# ---------------------------------------------------------------------------
# Plot 11: Mixing horizon comparison (4-block only)
# ---------------------------------------------------------------------------

def plot_mixing_horizon(df, save_dir):
    """Grouped bar: mixing_horizon=27 vs 243 for 4-block models."""
    sub = df[df["n_blocks"] == 4].copy()
    if sub.empty or "mixing_horizon" not in sub.columns:
        return None

    horizons = sorted(sub["mixing_horizon"].dropna().unique())
    if len(horizons) < 2:
        return None

    fracs = [f for f in FRAC_ORDER if f in sub["data_frac"].values]
    if not fracs:
        return None

    scat = sub[~sub["_is_random"]]
    rand = sub[sub["_is_random"]]

    combos = []
    for mod in sorted(scat["modulus_type"].dropna().unique()):
        mod_short = "prelu" if mod == "phase_relu" else "cmod"
        for lv in sorted(scat["L"].dropna().unique()):
            for frac in fracs:
                label = f"4b_{mod_short}_{int(lv)}L ({frac})"
                vals = {}
                for h in horizons:
                    s_sub = scat[(scat["modulus_type"] == mod) &
                                 (scat["L"] == lv) &
                                 (scat["mixing_horizon"] == h) &
                                 (scat["data_frac"] == frac)]
                    r_sub = rand[(rand["modulus_type"] == mod) &
                                 (rand["L"] == lv) &
                                 (rand["mixing_horizon"] == h) &
                                 (rand["data_frac"] == frac)]
                    vals[h] = {
                        "scat": s_sub["finetune_acc"].max() if not s_sub.empty else np.nan,
                        "rand": r_sub["joint_acc"].max() if not r_sub.empty else np.nan,
                    }
                combos.append({"label": label, "vals": vals})

    if not combos:
        return None

    x = np.arange(len(combos))
    h0, h1 = horizons[0], horizons[1]
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(12, len(combos) * 0.8), 6))

    ax.bar(x - 1.5 * width,
           [c["vals"][h0]["scat"] for c in combos], width,
           label=f"Scat h={int(h0)}", color="tab:blue", alpha=0.7)
    ax.bar(x - 0.5 * width,
           [c["vals"][h1]["scat"] for c in combos], width,
           label=f"Scat h={int(h1)}", color="tab:cyan", alpha=0.7)
    ax.bar(x + 0.5 * width,
           [c["vals"][h0]["rand"] for c in combos], width,
           label=f"Rand h={int(h0)}", color="tab:red", alpha=0.7)
    ax.bar(x + 1.5 * width,
           [c["vals"][h1]["rand"] for c in combos], width,
           label=f"Rand h={int(h1)}", color="tab:orange", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([c["label"] for c in combos], rotation=45, ha="right",
                       fontsize=7)
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Mixing Horizon Comparison (4-block models)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_mixing_horizon.png")


# ---------------------------------------------------------------------------
# Plot 12: Best architecture per data fraction
# ---------------------------------------------------------------------------

def plot_best_summary(df, save_dir):
    """Bar chart: best scattering accuracy per data fraction, labelled with arch."""
    scat = df[~df["_is_random"]]

    fracs = [f for f in FRAC_ORDER if f in scat["data_frac"].values]
    if not fracs:
        return None

    best_accs, best_labels = [], []
    for frac in fracs:
        sub = scat[scat["data_frac"] == frac]
        if sub.empty:
            best_accs.append(0)
            best_labels.append("n/a")
            continue
        idx = sub["finetune_acc"].idxmax()
        best_accs.append(sub.loc[idx, "finetune_acc"])
        best_labels.append(sub.loc[idx, "_arch_label"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fracs))
    bars = ax.bar(x, best_accs, color="tab:green", alpha=0.7,
                  edgecolor="black", linewidth=0.5)

    for bar, lbl in zip(bars, best_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                lbl, ha="center", va="bottom", fontsize=8, rotation=20)

    ax.set_xticks(x)
    ax.set_xticklabels(fracs)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Best val accuracy (%)")
    ax.set_title("Best Scattering Architecture per Data Fraction")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_best_summary.png")


# ---------------------------------------------------------------------------
# Plot 13: Per-experiment training curves (optional)
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

    print(f"  Saved {count} training curve(s) -> {curves_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Analyze experiment results")
    p.add_argument("--results-dir", default="cifar_exps",
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

    print("Generating plots...")
    plot_arch_comparison(df, plot_dir)
    plot_box_modulus(df, plot_dir)
    plot_box_depth(df, plot_dir)
    plot_box_gain(df, plot_dir)
    plot_delta_vs_gain(df, plot_dir)
    plot_data_efficiency(df, plot_dir)
    plot_init_advantage(df, plot_dir)
    plot_heatmap(df, plot_dir)
    plot_l2_effect(df, plot_dir)
    plot_paired_diff(df, plot_dir)
    plot_mixing_horizon(df, plot_dir)
    plot_best_summary(df, plot_dir)

    if args.curves:
        plot_training_curves(df, plot_dir)

    if args.interactive:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
