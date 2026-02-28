# CIFAR-10 Experiment Plan

Goal: demonstrate that scattering-initialized models outperform randomly-initialized
models, especially in low-data regimes, and that fine-tuning improves over frozen features.

## Architecture Grid (18 configurations)

Common: `--kernel-size 5`, `--global-avg-pool` off (always False)

### 3-block (6 configs)
- `--lowpass-last` off, `--mixing-horizon` None (dense)
- `--n-blocks 3`
- L: [4, 6, 8]
- modulus-type: [phase_relu, complex_modulus]

### 4-block (12 configs)
- `--lowpass-last` on, `--kernel-size 5`
- `--n-blocks 4`
- L: [4, 6, 8]
- modulus-type: [phase_relu, complex_modulus]
- mixing-horizon: [27, 243]

## Training Variables

### Scattering init (two-phase)
- Base: `--lr-epochs 20 --epochs 100 --global-batch-size 128 --print-freq 200 --workers 10`
- train-size: [0.1, 0.3, 0.5, 1.0]
- l2-penalty: [0, 0.005]
- 18 archs x 4 fractions x 2 l2 = **144 experiments**

### Random init (joint, single-phase)
- Base: `--random-init --joint --epochs 120 --global-batch-size 128 --print-freq 200 --workers 10`
- train-size: [0.1, 0.3, 0.5, 1.0]
- l2-penalty: 0 only (ignored for random init)
- 18 archs x 4 fractions x 1 = **72 experiments**

### Total: 216 experiments

## Running

Two bash scripts, each accepts a GPU count argument:
```bash
# Scattering init (144 experiments), using 6 GPUs:
cd new
./run_cifar.sh 6

# Random init (72 experiments), using 6 GPUs:
cd new
./run_cifar_rand.sh 6

# Both can run on separate nodes simultaneously.
```
Experiments run in batches of N GPUs. After each batch, checkpoint .pth files are
deleted (only results_*.json kept) to save disk space. All results go to `cifar_exps/`.

### Example single experiment commands

Scattering init (two-phase), 3-block, phase_relu, L=8, 10% data, l2=0.005:
```bash
python train.py --lr-epochs 20 --epochs 100 --global-batch-size 128 --print-freq 200 \
    --workers 10 --kernel-size 5 --n-blocks 3 --modulus-type phase_relu --L 8 \
    --train-size 0.1 --l2-penalty 0.005 --save-dir cifar_exps/3b_prelu_8L_s01_l2
```

Scattering init, 4-block, complex_modulus, L=6, horizon=27, full data, l2=0:
```bash
python train.py --lr-epochs 20 --epochs 100 --global-batch-size 128 --print-freq 200 \
    --workers 10 --kernel-size 5 --n-blocks 4 --modulus-type complex_modulus --L 6 \
    --lowpass-last --mixing-horizon 27 --save-dir cifar_exps/4b_cmod_6L_h27_s10
```

Random init (joint, single-phase), 3-block, phase_relu, L=8, 30% data:
```bash
python train.py --random-init --joint --epochs 120 --global-batch-size 128 \
    --print-freq 200 --workers 10 --kernel-size 5 --n-blocks 3 \
    --modulus-type phase_relu --L 8 --train-size 0.3 \
    --save-dir cifar_exps/3b_prelu_8L_rand_s03
```

## Save-dir Naming Convention

- 3-block scat: `cifar_exps/3b_{mod}_{L}L_s{frac}[_l2]`
- 4-block scat: `cifar_exps/4b_{mod}_{L}L_h{horizon}_s{frac}[_l2]`
- 3-block rand: `cifar_exps/3b_{mod}_{L}L_rand_s{frac}`
- 4-block rand: `cifar_exps/4b_{mod}_{L}L_h{horizon}_rand_s{frac}`

Where: mod=prelu|cmod, L=4|6|8, horizon=27|243, frac=1|3|5|10, _l2 suffix when l2=0.005.

## Analysis

```bash
# Generate summary table + all plots:
python analyze.py --results-dir cifar_exps --recursive

# Also generate per-experiment training curves:
python analyze.py --results-dir cifar_exps --recursive --curves

# Show plots interactively:
python analyze.py --results-dir cifar_exps --recursive --interactive
```

### Plots produced:
1. Architecture comparison (grouped bar, full-data)
2. Box plot — accuracy by modulus type
3. Box plot — accuracy by depth
4. Box plot — fine-tune gain by data fraction
5. Filter delta vs fine-tune gain (scatter)
6. Data efficiency (three-line: random < scat-frozen < scat-tuned)
7. Scattering advantage over random vs data size
8. Heatmap: architecture x data fraction (scat vs random side-by-side)
9. L2 penalty effect (delta accuracy with vs without l2)
10. Paired difference (scat - random) horizontal bars
11. Mixing horizon comparison (4-block h27 vs h243)
12. Best architecture summary per data fraction
13. Per-experiment training curves (optional, with --curves)
