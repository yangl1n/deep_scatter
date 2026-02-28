# CIFAR Post-Experiments Plan (Report-Ready)

Goal: Run focused, multi-seed experiments on the best architectures from Round 1,
with fine-grained data fractions and statistical analysis, for CIFAR-10 and CIFAR-100.

## Datasets

- **CIFAR-10**: 10 classes, 50k train / 10k test
- **CIFAR-100**: 100 classes, 50k train / 10k test (harder, scattering advantage expected to be larger)

## Architecture Grid (2 configs)

Selected from Round 1 winners (strongest scattering advantage in low-data regime):

| Architecture | Params | Notes |
|--------------|--------|-------|
| `3b_cmod_4L` | ~1.3M | 3-block, complex_modulus, L=4, no mixing_horizon |
| `4b_cmod_4L_h27` | ~2.0M | 4-block, complex_modulus, L=4, mixing_horizon=27 |

Common: `--kernel-size 5`, `--global-avg-pool` off, `--lowpass-last` off for 3-block, on for 4-block.

## Training Variables

### Scattering init (two-phase)
- Base: `--lr-epochs 20 --epochs 150 --global-batch-size 128 --print-freq 200 --workers 10`
- train-size: [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
- l2-penalty: 0 only
- seeds: [42, 43, 44, 45, 46]
- 2 archs x 7 fractions x 5 seeds = **70 experiments per dataset**

### Random init (joint, single-phase)
- Base: `--random-init --joint --epochs 180 --global-batch-size 128 --print-freq 200 --workers 10`
- train-size: [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
- seeds: [42, 43, 44, 45, 46]
- 2 archs x 7 fractions x 5 seeds = **70 experiments per dataset**

### Total per dataset: 140 experiments
### Total (CIFAR-10 + CIFAR-100): 280 experiments

## Save Directory Convention

Results go to `cifar_post_{dataset}/` (e.g. `cifar_post_cifar10/`, `cifar_post_cifar100/`).

Naming:
- Scattering: `{arch}_s{frac}_seed{seed}` (e.g. `3b_cmod_4L_s1_seed42`)
- Random: `{arch}_rand_s{frac}_seed{seed}` (e.g. `3b_cmod_4L_rand_s1_seed42`)

Where frac = `05|1|15|2|3|5|10` (5%, 10%, 15%, 20%, 30%, 50%, 100%).

## Running

```bash
cd new

# CIFAR-10 (140 experiments), 6 GPUs:
./run_cifar_post.sh 6 cifar10

# CIFAR-100 (140 experiments), 6 GPUs:
./run_cifar_post.sh 6 cifar100

# Random init (run separately or in parallel):
./run_cifar_post_rand.sh 6 cifar10
./run_cifar_post_rand.sh 6 cifar100
```

Both scripts accept: `<num_gpus> <dataset>` where dataset is `cifar10` or `cifar100`.

## Analysis

Run post_analyze.py on each results directory (CIFAR-10 and CIFAR-100 are analyzed separately):

```bash
# CIFAR-10 (generates plots with error bars + significance table):
python post_analyze.py --results-dir cifar_post_cifar10 --recursive

# CIFAR-100:
python post_analyze.py --results-dir cifar_post_cifar100 --recursive

# Optional: convergence check (overlay training curves across seeds)
python post_analyze.py --results-dir cifar_post_cifar10 --recursive --curves
```

### Plots produced by post_analyze.py

1. **Data efficiency with error bands** — Dual-panel (one per arch), 3 lines (scat-head, scat-finetuned, random) with mean ± std shaded bands across 5 seeds.
2. **Scattering advantage with 95% CI** — Line plot of (scat - random) vs data fraction, with error bars.
3. **Publication summary figure** — Clean two-panel figure per dataset, suitable for reports.
4. **Filter change vs fine-tune gain** — Scatter of mean filter delta vs mean gain per (arch, fraction), with error bars and linear fit (r, R²).
5. **Gain and filter delta by fraction** — Dual-axis: fine-tune gain and relative filter change vs data fraction, one panel per architecture.
6. **Convergence check** (optional, with `--curves`) — Overlay of all 5 seeds' training curves for sample conditions.

### Tables produced

- **significance_table.csv** — mean_scat, mean_rand, delta, std, p-value (paired t-test) per (arch, fraction).
- **delta_gain_table.csv** — delta_mean, delta_std, gain_mean, gain_std, corr_delta_gain per (arch, fraction).
