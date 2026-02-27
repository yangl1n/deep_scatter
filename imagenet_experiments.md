# ImageNet Experiments

All experiments use DDP via `torchrun`.

## Experiment table

| ID | Blocks | L | Modulus | lowpass-last | mixing-horizon | GAP | train-size | Group |
|----|--------|---|--------|-------------|----------------|-----|------------|-------|
| A  | 5      | 4 | complex_modulus | yes | — | no | full | Depth |
| B  | 5      | 4 | complex_modulus | yes | 27 | no | full | Depth |
| C  | 4      | 4 | complex_modulus | yes | — | no | full | Depth (baseline) |
| D  | 5      | 8 | complex_modulus | yes | 27 | no | full | Orientations |
| E  | 4      | 8 | complex_modulus | yes | 27 | no | full | Orientations |
| F  | 4      | 4 | phase_relu | yes | — | no | full | Modulus type |
| G  | 5      | 8 | phase_relu | yes | 27 | no | full | Modulus type |
| H  | 4      | 4 | complex_modulus | yes | — | no | 0.1 | Data efficiency |
| I  | 4      | 4 | complex_modulus | yes | — | no | 0.3 | Data efficiency |

**Exp C** is already running. All others are new.

## Channel counts and classifier size

| ID | Final channels | Final spatial | Classifier params (no GAP) | Classifier params (GAP) |
|----|---------------|--------------|---------------------------|------------------------|
| A  | 1,875         | 7×7          | 92M                       | 1.9M                   |
| B  | 1,875         | 7×7          | 92M                       | 1.9M                   |
| C  | 375           | 14×14        | 73.5M                     | 375K                   |
| D  | 19,683        | 7×7          | 964M                      | 19.7M                  |
| E  | 2,187         | 14×14        | 429M                      | 2.2M                   |
| F  | 375           | 14×14        | 73.5M                     | 375K                   |
| G  | 19,683        | 7×7          | 964M                      | 19.7M                  |
| H  | 375           | 14×14        | 73.5M                     | 375K                   |
| I  | 375           | 14×14        | 73.5M                     | 375K                   |

> **Note:** Exp D, G (5 blocks, L=8) produce 19,683 channels — `--global-avg-pool` is
> strongly recommended. Exp A, B, E also have large classifiers without GAP.
> If OOM occurs, add `--global-avg-pool` or reduce `--global-batch-size` to 512, or use `--mixing-horizon 27` to reduce the 3D filters.

## Commands

Common prefix:

Suppose 8 gpus, each one run 128 batch size, so totally 1024.
And the imagenet folder contains train/ and val/, each contain class subfolders

```bash
BASE="torchrun --nproc_per_node=8 train.py --dataset imagenet --global-batch-size 1024 --data-dir /datasets/ImageNet --lr-epochs 20 --print-freq 200 "
```

### Group 1: Depth comparison (L=4, lowpass-last)

```bash
# Exp A: 5 blocks, no mixing-horizon
$BASE --n-blocks 5 --L 4 --lowpass-last --save-dir imagenet_runs/5b_L4_lp

# Exp B: 5 blocks, mixing-horizon 27
$BASE --n-blocks 5 --L 4 --lowpass-last --mixing-horizon 27 --save-dir imagenet_runs/5b_L4_lp_h27

# Exp C: 4 blocks (already running)
$BASE --n-blocks 4 --L 4 --lowpass-last --save-dir imagenet_runs/4b_L4_lp
```

### Group 2: Orientations (L=8, mixing-horizon 27)

```bash
# Exp D: 5 blocks, L=8 (very heavy — likely needs --global-avg-pool)
$BASE --n-blocks 5 --L 8 --lowpass-last --mixing-horizon 27 --save-dir imagenet_runs/5b_L8_lp_h27

# Exp E: 4 blocks, L=8
$BASE --n-blocks 4 --L 8 --lowpass-last --mixing-horizon 27 --save-dir imagenet_runs/4b_L8_lp_h27
```

### Group 3: Modulus type (phase_relu)

```bash
# Exp F: 4 blocks, phase_relu, L=4
$BASE --n-blocks 4 --L 4 --lowpass-last --modulus-type phase_relu --save-dir imagenet_runs/4b_L4_lp_prelu

# Exp G: 5 blocks, phase_relu, L=8 (very heavy — likely needs --global-avg-pool)
$BASE --n-blocks 5 --L 8 --lowpass-last --modulus-type phase_relu --mixing-horizon 27 --save-dir imagenet_runs/5b_L8_lp_prelu_h27
```

### Group 4: Data efficiency

```bash
# Exp H: 10% of ImageNet
$BASE --n-blocks 4 --L 4 --lowpass-last --train-size 0.1 --save-dir imagenet_runs/4b_L4_lp_s01

# Exp I: 30% of ImageNet
$BASE --n-blocks 4 --L 4 --lowpass-last --train-size 0.3 --save-dir imagenet_runs/4b_L4_lp_s03
```

## Priority

1. **Exp A, F** — 5-block depth scaling and phase_relu comparison (context for existing 4-block run)
2. **Exp E** — does L=8 help on ImageNet at 4 blocks?
3. **Exp H, I** — data efficiency (core research question)
4. **Exp B, D, G** — mixing-horizon and deeper L=8 variants (may need GAP)

## What to look for

| Comparison | Experiments | Key metric |
|------------|------------|------------|
| Does 5th block help over 4? | A vs C | `head_acc`, `finetune_gain` |
| Does mixing-horizon help at 5 blocks? | A vs B | `head_acc` (similar?) + speed |
| L=4 vs L=8 at 4 blocks | C vs E | `head_acc`, `finetune_gain` |
| L=4 vs L=8 at 5 blocks | A vs D | `head_acc`, `finetune_gain` |
| complex_modulus vs phase_relu | C vs F, D vs G | `head_acc`, `finetune_gain` |
| Data efficiency of scattering prior | C vs H vs I | `finetune_gain` vs `train_size` |
