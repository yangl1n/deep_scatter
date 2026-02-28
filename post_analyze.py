"""Post-experiment analysis: multi-seed statistical analysis for report-ready plots.

Designed for cifar_post_* directories with 5 seeds per (arch, fraction, init_type).
Produces data efficiency curves with error bands, scattering advantage with 95% CI,
filter change vs fine-tune gain plots, and statistical significance tables.

Usage
-----
    python post_analyze.py --results-dir cifar_post_cifar10 --recursive
    python post_analyze.py --results-dir cifar_post_cifar100 --recursive
    python post_analyze.py --results-dir cifar_post_cifar10 --recursive --curves

Plots produced
--------------
- post_data_efficiency.png: Data efficiency (scat head, scat fine-tuned, random)
- post_advantage_ci.png: Scattering advantage with 95% CI
- post_publication_summary.png: Scat vs random (mean ± std)
- post_delta_vs_gain.png: Filter change vs fine-tune gain (scatter + linear fit)
- post_delta_gain_by_fraction.png: Gain and filter delta vs data fraction (dual axis)
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
from scipy import stats

FRAC_ORDER = ["5%", "10%", "15%", "20%", "30%", "50%", "100%"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_label(row):
    return row.get("save_dir", "").rstrip("/").split("/")[-1]


def _extract_seed(row):
    """Extract seed from save_dir, e.g. '3b_cmod_4L_s1_seed42' -> 42."""
    label = _make_label(row)
    m = re.search(r"seed(\d+)", label, re.I)
    return int(m.group(1)) if m else None


def _data_fraction_label(row):
    ts = row.get("train_size")
    if ts is None or (isinstance(ts, float) and np.isnan(ts)) or ts >= 1.0:
        return "100%"
    return f"{int(ts * 100)}%"


def _is_random(row):
    return bool(row.get("random_init", False))


def _arch_short_label(row):
    nb = int(row["n_blocks"])
    mod = "prelu" if row["modulus_type"] == "phase_relu" else "cmod"
    lv = int(row["L"])
    mh = row.get("mixing_horizon")
    if isinstance(mh, float) and np.isnan(mh):
        mh = None
    if mh is not None:
        return f"{nb}b_{mod}_{lv}L_h{int(mh)}"
    return f"{nb}b_{mod}_{lv}L"


def _save(fig, save_dir, name):
    path = os.path.join(save_dir, name)
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_experiments(results_dir, recursive=False):
    pattern = os.path.join(results_dir, "**" if recursive else "", "results_*.json")
    paths = sorted(glob.glob(pattern, recursive=recursive))

    if not paths:
        print(f"No results_*.json found in {results_dir}")
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
    df["_arch_label"] = df.apply(_arch_short_label, axis=1)
    df["_seed"] = df.apply(_extract_seed, axis=1)
    if "dataset" not in df.columns:
        df["dataset"] = "cifar10"

    if "joint_acc" in df.columns and "finetune_acc" in df.columns:
        df["best_acc"] = df["joint_acc"].fillna(df["finetune_acc"])
    elif "joint_acc" in df.columns:
        df["best_acc"] = df["joint_acc"]
    elif "finetune_acc" in df.columns:
        df["best_acc"] = df["finetune_acc"]

    n_with_seed = df["_seed"].notna().sum()
    print(f"Loaded {len(df)} experiments ({n_with_seed} with seed)\n")
    return df


# ---------------------------------------------------------------------------
# Aggregate by (arch, fraction, init_type) -> mean, std, sem
# ---------------------------------------------------------------------------

def _build_aggregates(df):
    """Group by arch, data_frac, init_type; compute mean, std, sem across seeds."""
    frac_order = [f for f in FRAC_ORDER if f in df["data_frac"].values]
    scat = df[~df["_is_random"]]
    rand = df[df["_is_random"]]

    agg = []
    for arch in sorted(df["_arch_label"].unique()):
        for frac in frac_order:
            s_sub = scat[(scat["_arch_label"] == arch) & (scat["data_frac"] == frac)]
            r_sub = rand[(rand["_arch_label"] == arch) & (rand["data_frac"] == frac)]

            def _stats(sub, acc_col):
                vals = sub[acc_col].dropna()
                if len(vals) == 0:
                    return np.nan, np.nan, np.nan, []
                return vals.mean(), vals.std(), vals.sem(), vals.tolist()

            head_mean, head_std, head_sem, _ = _stats(s_sub, "head_acc")
            ft_mean, ft_std, ft_sem, ft_vals = _stats(s_sub, "finetune_acc")
            j_mean, j_std, j_sem, j_vals = _stats(r_sub, "joint_acc")
            gain_mean, gain_std, _, gain_vals = _stats(s_sub, "finetune_gain")
            delta_mean, delta_std, _, delta_vals = _stats(s_sub, "filter_delta_relative")

            samples = r_sub["actual_train_samples"].dropna()
            n_samples = int(samples.iloc[0]) if not samples.empty else None

            # Correlation between delta and gain across seeds (for this arch, frac)
            corr_dg = np.nan
            if len(gain_vals) >= 3 and len(delta_vals) >= 3:
                r, _ = stats.pearsonr(delta_vals, gain_vals)
                corr_dg = r

            agg.append({
                "arch": arch,
                "data_frac": frac,
                "samples": n_samples,
                "scat_head_mean": head_mean, "scat_head_std": head_std,
                "scat_ft_mean": ft_mean, "scat_ft_std": ft_std, "scat_ft_vals": ft_vals,
                "rand_mean": j_mean, "rand_std": j_std, "rand_vals": j_vals,
                "gain_mean": gain_mean, "gain_std": gain_std, "gain_vals": gain_vals,
                "delta_mean": delta_mean, "delta_std": delta_std, "delta_vals": delta_vals,
                "corr_delta_gain": corr_dg,
            })
    return pd.DataFrame(agg), frac_order


def _build_paired_stats(df):
    """For each (arch, frac): paired t-test scat vs rand (matched by seed)."""
    scat = df[~df["_is_random"]]
    rand = df[df["_is_random"]]

    rows = []
    for arch in sorted(df["_arch_label"].unique()):
        for frac in sorted(df["data_frac"].unique()):
            s_sub = scat[(scat["_arch_label"] == arch) & (scat["data_frac"] == frac)]
            r_sub = rand[(rand["_arch_label"] == arch) & (rand["data_frac"] == frac)]

            if s_sub.empty or r_sub.empty:
                continue

            # Match by seed
            merged = s_sub[["_seed", "finetune_acc"]].merge(
                r_sub[["_seed", "joint_acc"]], on="_seed", how="inner"
            )
            if len(merged) < 2:
                continue

            scat_acc = merged["finetune_acc"].values
            rand_acc = merged["joint_acc"].values
            diff = scat_acc - rand_acc

            t_stat, p_val = stats.ttest_rel(scat_acc, rand_acc)
            n = len(diff)
            sem = diff.std() / np.sqrt(n) if n > 1 else 0
            t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0
            ci95 = t_crit * sem

            rows.append({
                "arch": arch,
                "data_frac": frac,
                "mean_scat": scat_acc.mean(),
                "mean_rand": rand_acc.mean(),
                "delta": diff.mean(),
                "std_delta": diff.std(),
                "p_value": p_val,
                "ci95_lo": diff.mean() - ci95,
                "ci95_hi": diff.mean() + ci95,
                "n_seeds": n,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_data_efficiency(agg_df, frac_order, save_dir):
    """Dual-panel: one per arch, 3 lines with mean ± std shaded bands."""
    archs = agg_df["arch"].unique()
    if len(archs) == 0:
        return

    fig, axes = plt.subplots(1, len(archs), figsize=(6 * len(archs), 5))
    if len(archs) == 1:
        axes = [axes]

    samples = agg_df.groupby("data_frac")["samples"].first()
    x_vals = [samples.get(f) for f in frac_order if f in samples.index]
    x_labels = frac_order[: len(x_vals)]

    for ax, arch in zip(axes, archs):
        sub = agg_df[agg_df["arch"] == arch].set_index("data_frac")
        sub_sorted = sub.reindex([f for f in frac_order if f in sub.index]).dropna(how="all")
        x = sub_sorted["samples"].values
        if len(x) == 0:
            continue

        # Scat head (Phase A)
        h_mean = sub_sorted["scat_head_mean"].values
        h_std = sub_sorted["scat_head_std"].values
        ax.plot(x, h_mean, "^--", color="tab:blue", alpha=0.8, label="Scat head")
        ax.fill_between(x, h_mean - h_std, h_mean + h_std, color="tab:blue", alpha=0.2)

        # Scat fine-tuned
        ft_mean = sub_sorted["scat_ft_mean"].values
        ft_std = sub_sorted["scat_ft_std"].values
        ax.plot(x, ft_mean, "s-", color="tab:orange", label="Scat fine-tuned")
        ax.fill_between(x, ft_mean - ft_std, ft_mean + ft_std, color="tab:orange", alpha=0.2)

        # Random
        r_mean = sub_sorted["rand_mean"].values
        r_std = sub_sorted["rand_std"].values
        ax.plot(x, r_mean, "x:", color="tab:red", markersize=8, label="Random joint")
        ax.fill_between(x, r_mean - r_std, r_mean + r_std, color="tab:red", alpha=0.2)

        ax.set_xscale("log")
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Val accuracy (%)")
        ax.set_title(arch)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Data Efficiency (mean ± std across 5 seeds)", fontsize=12)
    fig.tight_layout()
    _save(fig, save_dir, "post_data_efficiency.png")


def plot_advantage_ci(paired_df, save_dir):
    """Line plot: scattering advantage vs data fraction with 95% CI error bars."""
    if paired_df.empty:
        return

    frac_order = [f for f in FRAC_ORDER if f in paired_df["data_frac"].values]
    archs = sorted(paired_df["arch"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")

    for i, arch in enumerate(archs):
        sub = paired_df[paired_df["arch"] == arch].set_index("data_frac")
        sub = sub.reindex([f for f in frac_order if f in sub.index]).dropna(how="all")
        if sub.empty:
            continue

        x = np.arange(len(sub))
        delta = sub["delta"].values
        err_lo = (sub["delta"] - sub["ci95_lo"]).values
        err_hi = (sub["ci95_hi"] - sub["delta"]).values
        yerr = np.array([err_lo, err_hi])

        color = cmap(i % 10)
        ax.errorbar(x, delta, yerr=yerr, fmt="o-", color=color, label=arch, capsize=4)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(frac_order)))
    ax.set_xticklabels(frac_order)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Accuracy advantage (%)")
    ax.set_title("Scattering Init Advantage (mean ± 95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "post_advantage_ci.png")


def plot_convergence_check(df, save_dir):
    """Overlay training curves for sample conditions (e.g. 10%, 100%) across seeds."""
    curves_dir = os.path.join(save_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)

    # Pick 2 sample conditions: 10% and 100% for first arch
    archs = df["_arch_label"].unique()[:1]
    for arch in archs:
        for frac in ["10%", "100%"]:
            sub = df[(df["_arch_label"] == arch) & (df["data_frac"] == frac) & (~df["_is_random"])]
            if sub.empty or len(sub) < 2:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            for _, row in sub.iterrows():
                curves = row.get("_curves", {})
                phase_b = curves.get("phase_b", [])
                if not phase_b:
                    continue
                epochs = [e["epoch"] for e in phase_b]
                val_acc = [e["val_acc"] for e in phase_b]
                seed = row.get("_seed", "?")
                ax.plot(epochs, val_acc, alpha=0.7, label=f"seed {seed}")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val accuracy (%)")
            ax.set_title(f"{arch} {frac} (scattering, Phase B)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            safe = arch.replace(" ", "_") + "_" + frac.replace("%", "pct")
            _save(fig, curves_dir, f"convergence_{safe}.png")


def plot_publication_summary(agg_df, paired_df, frac_order, save_dir):
    """Clean two-panel figure for reports."""
    archs = sorted(agg_df["arch"].unique())
    if len(archs) == 0:
        return

    fig, axes = plt.subplots(1, len(archs), figsize=(6 * len(archs), 5))
    if len(archs) == 1:
        axes = [axes]

    for ax, arch in zip(axes, archs):
        sub = agg_df[agg_df["arch"] == arch].set_index("data_frac")
        sub = sub.reindex([f for f in frac_order if f in sub.index]).dropna(how="all")
        if sub.empty:
            continue

        x = np.arange(len(sub))
        ax.plot(x, sub["scat_ft_mean"], "s-", color="tab:orange", linewidth=2,
                label="Scattering (fine-tuned)")
        ax.fill_between(x,
                        sub["scat_ft_mean"] - sub["scat_ft_std"],
                        sub["scat_ft_mean"] + sub["scat_ft_std"],
                        color="tab:orange", alpha=0.25)

        ax.plot(x, sub["rand_mean"], "o--", color="tab:red", linewidth=2,
                label="Random init")
        ax.fill_between(x,
                        sub["rand_mean"] - sub["rand_std"],
                        sub["rand_mean"] + sub["rand_std"],
                        color="tab:red", alpha=0.25)

        ax.set_xticks(x)
        ax.set_xticklabels(sub.index.tolist())
        ax.set_xlabel("Training data fraction")
        ax.set_ylabel("Val accuracy (%)")
        ax.set_title(arch)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Scattering vs Random Init (mean ± std, 5 seeds)", fontsize=12)
    fig.tight_layout()
    _save(fig, save_dir, "post_publication_summary.png")


def plot_delta_vs_gain(agg_df, save_dir):
    """Scatter: filter change vs fine-tune gain per (arch, fraction), with error bars and linear fit."""
    sub = agg_df.dropna(subset=["delta_mean", "gain_mean"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    archs = sorted(sub["arch"].unique())

    for i, arch in enumerate(archs):
        a_sub = sub[sub["arch"] == arch]
        x = a_sub["delta_mean"].values
        y = a_sub["gain_mean"].values
        xerr = a_sub["delta_std"].values
        yerr = a_sub["gain_std"].values
        color = cmap(i % 10)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color=color,
                    label=arch, capsize=3, capthick=1)

        for _, row in a_sub.iterrows():
            ax.annotate(row["data_frac"], (row["delta_mean"], row["gain_mean"]),
                        fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points")

    # Overall linear fit
    x_all = sub["delta_mean"].values
    y_all = sub["gain_mean"].values
    if len(x_all) >= 3:
        slope, intercept, r, p, se = stats.linregress(x_all, y_all)
        x_line = np.linspace(x_all.min(), x_all.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.7,
                label=f"Fit: r={r:.3f}, R²={r**2:.3f}")

    ax.set_xlabel("Relative filter change (L2)")
    ax.set_ylabel("Fine-tune gain (%)")
    ax.set_title("Filter Change vs Fine-Tuning Gain (mean ± std across seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "post_delta_vs_gain.png")


def plot_delta_gain_by_fraction(agg_df, frac_order, save_dir):
    """Dual-panel: gain and filter delta vs data fraction, one panel per architecture."""
    archs = sorted(agg_df["arch"].unique())
    if len(archs) == 0:
        return

    fig, axes = plt.subplots(1, len(archs), figsize=(6 * len(archs), 5))
    if len(archs) == 1:
        axes = [axes]

    for ax, arch in zip(axes, archs):
        sub = agg_df[agg_df["arch"] == arch].set_index("data_frac")
        sub = sub.reindex([f for f in frac_order if f in sub.index]).dropna(how="all")
        if sub.empty:
            continue

        x = np.arange(len(sub))
        ax2 = ax.twinx()

        # Gain (left y-axis)
        ax.plot(x, sub["gain_mean"], "s-", color="tab:green", linewidth=2,
                label="Fine-tune gain")
        ax.fill_between(x,
                        sub["gain_mean"] - sub["gain_std"],
                        sub["gain_mean"] + sub["gain_std"],
                        color="tab:green", alpha=0.2)
        ax.set_ylabel("Fine-tune gain (%)", color="tab:green")
        ax.tick_params(axis="y", labelcolor="tab:green")

        # Filter delta (right y-axis)
        ax2.plot(x, sub["delta_mean"], "o--", color="tab:purple", linewidth=2,
                 label="Filter change")
        ax2.fill_between(x,
                         sub["delta_mean"] - sub["delta_std"],
                         sub["delta_mean"] + sub["delta_std"],
                         color="tab:purple", alpha=0.2)
        ax2.set_ylabel("Relative filter change (L2)", color="tab:purple")
        ax2.tick_params(axis="y", labelcolor="tab:purple")

        ax.set_xticks(x)
        ax.set_xticklabels(sub.index.tolist())
        ax.set_xlabel("Training data fraction")
        ax.set_title(arch)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)

    fig.suptitle("Fine-Tune Gain and Filter Change vs Data Fraction", fontsize=12)
    fig.tight_layout()
    _save(fig, save_dir, "post_delta_gain_by_fraction.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Post-experiment multi-seed analysis")
    p.add_argument("--results-dir", default="cifar_post_cifar10",
                   help="directory containing results (e.g. cifar_post_cifar10)")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--curves", action="store_true", help="plot convergence check")
    args = p.parse_args()

    df = load_experiments(args.results_dir, recursive=args.recursive)

    # Require seeds for post analysis
    with_seed = df[df["_seed"].notna()]
    if len(with_seed) < 10:
        print("Warning: few experiments with seed. Post analysis expects 5 seeds per condition.")

    agg_df, frac_order = _build_aggregates(df)
    paired_df = _build_paired_stats(df)

    plot_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("Generating post-analysis plots...")
    plot_data_efficiency(agg_df, frac_order, plot_dir)
    plot_advantage_ci(paired_df, plot_dir)
    plot_publication_summary(agg_df, paired_df, frac_order, plot_dir)
    plot_delta_vs_gain(agg_df, plot_dir)
    plot_delta_gain_by_fraction(agg_df, frac_order, plot_dir)

    if args.curves:
        plot_convergence_check(df, plot_dir)

    # Save significance table
    if not paired_df.empty:
        sig_path = os.path.join(args.results_dir, "significance_table.csv")
        paired_df.to_csv(sig_path, index=False)
        print(f"  Saved {sig_path}")

    # Save delta-gain table (filter change vs fine-tune gain)
    dg_cols = ["arch", "data_frac", "delta_mean", "delta_std", "gain_mean", "gain_std", "corr_delta_gain"]
    if all(c in agg_df.columns for c in dg_cols):
        dg_df = agg_df[dg_cols].dropna(subset=["delta_mean", "gain_mean"])
        if not dg_df.empty:
            dg_path = os.path.join(args.results_dir, "delta_gain_table.csv")
            dg_df.to_csv(dg_path, index=False)
            print(f"  Saved {dg_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
