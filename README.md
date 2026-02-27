# Spatial Scattering — Learnable Wavelet Conv2d

A lightweight, kymatio-free reimplementation of the **wavelet scattering transform** using standard `nn.Conv2d` layers initialized with Morlet / Gabor wavelets. All filter parameters are learnable by default, with cross-channel interactions enabled via 3D (channel + spatial) convolution kernels.

## Files

| File | Description |
|---|---|
| `wavelet_kernels.py` | Pure-NumPy Gabor / Morlet 2D kernel generation |
| `spatial_scattering.py` | `ScatteringBlock` — single-scale (J=1) scattering building block |
| `scattering_net.py` | `ScatteringNet` — multi-order scattering model (stacks J blocks + classifier) |
| `data.py` | Dataset construction (CIFAR-10 / ImageNet), stratified subset sampling |
| `engine.py` | Training / eval loop, early stopping, parameter-delta metrics |
| `utils.py` | Checkpoint I/O, experiment JSON logger, distributed helpers |
| `train.py` | Two-phase experiment pipeline (Phase A: train head, Phase B: fine-tune) |
| `quick_check.py` | Lightweight single-phase training for fast sanity checks |
| `analyze.py` | Load experiment JSONs, print summary table, generate plots |

## Motivation

The original implementation (cascade block `phase_scattering2d_torch.py` with J=1) performs scattering
in the Fourier domain via kymatio, operating on full-image-sized filters. This
project replaces that with **spatial-domain convolutions**:

| | Fourier domain (original) | Spatial Conv2d (this project) |
|---|---|---|
| Filter size | Full image (M x N) | Small kernel (e.g. 7 x 7) |
| Channel mixing | None (per-channel) | Cross-channel via 3D kernels |
| Learnable | No | Yes (filters + downsampling) |
| Dependencies | kymatio + scipy | Pure PyTorch + NumPy |
| Nonlinearity | Phase ReLU only | Phase ReLU or complex modulus |
| Downsampling | Fourier subsampling | Learnable 2x2 Conv2d (avg-pool init) |

## ScatteringBlock Architecture

Each block performs a single-scale (J=1) scattering with 2x spatial downsampling.

For input `[B, C_in, M, N]` with `L` wavelet orientations:

```
Input [B, C_in, M, N]
  |
  |-- ORDER 0 (low-pass) -------------------------------------------------
  |     phi_conv: Conv2d(C_in, C_in, K)     block-diagonal Gabor init
  |       |
  |     AvgPoolConv(2x2, stride=2)          learnable downsample
  |       |
  |     S_0: [B, C_in, M/2, N/2]
  |
  |-- ORDER 1 (bandpass) --------------------------------------------------
  |     psi_real: Conv2d(C_in, C_in*L, K)   block-diagonal Morlet init
  |     psi_imag: Conv2d(C_in, C_in*L, K)   block-diagonal Morlet init
  |       |
  |     AvgPoolConv(2x2, stride=2)          learnable downsample (shared)
  |       |
  |     ┌─ modulus_type='phase_relu' ─────────────────────────────────┐
  |     |  stack [Re, Im] -> PhaseReLU -> [B, 4*C_in*L, M/2, N/2]   |
  |     |    |                                                        |
  |     |  phase_collapse: grouped 1x1 Conv2d (init: 1/√2)           |
  |     |    -> [B, C_in*L, M/2, N/2]                                |
  |     |                                                             |
  |     |  (|a|+|b|)/√2 ≈ √(a²+b²) = |z|                            |
  |     ├─ modulus_type='complex_modulus' ────────────────────────────┤
  |     |  sqrt(Re² + Im²)             -> [B, C_in*L, M/2, N/2]     |
  |     └────────────────────────────────────────────────────────────┘
  |       = S_1
  |
  cat(S_0, S_1) -> [B, C_out, M/2, N/2]
```

Output channels: `C_out = C_in * (1 + L)` for both modes.

### Phase Collapse

In `phase_relu` mode, the 4 rectified phase channels per (channel, orientation)
group are collapsed into 1 channel via a **grouped 1x1 convolution**:

```
Conv2d(C_in*4*L, C_in*L, kernel_size=1, groups=C_in*L)
```

- Weight shape: `(C_in*L, 4, 1, 1)` — 4 learnable weights per group
- Initialized to `1/√2`, so the initial output is `(a⁺+b⁺+a⁻+b⁻)/√2 = (|a|+|b|)/√2 ≈ |z|`
- During training the 4 weights can deviate from `1/√2` to learn a better combination

This makes `phase_relu` output the same dimension as `complex_modulus`
(`C_in*(1+L)`), eliminating channel explosion while providing a differentiable,
learnable alternative to the hard-coded complex modulus.

## Cross-Channel 3D Kernels

Wavelet Conv2d weights have shape `(C_out, C_in, K, K)`. They are initialized
with a **block-diagonal** structure — each output group only depends on its
corresponding input channel, equivalent to per-channel (groups=C) convolution.
The off-diagonal planes are initialized to zero.

During training, the off-diagonal entries can learn non-zero values, enabling
cross-channel feature mixing while starting from a mathematically principled
wavelet initialization.

## ScatteringNet

Stacks J `ScatteringBlock` modules to compute scattering coefficients,
followed by a linear classifier.

```
Input (B, 3, 32, 32)
  → Block 0: (B, 3, 32, 32)    → (B, C1, 16, 16)
  → Block 1: (B, C1, 16, 16)   → (B, C2, 8, 8)
  → Block 2: (B, C2, 8, 8)     → (B, C3, 4, 4)
  → Flatten → Linear → (B, nb_classes)
```

### Connection to Fast Wavelet Transform and Scattering

The cascaded block structure mirrors the **fast wavelet transform (FWT)**
algorithm: each block applies a single-scale wavelet decomposition (bandpass
filters + low-pass) followed by 2x downsampling, operating on the output of
the previous scale. In FWT this is known as the *lifting* or *filter-bank*
iteration — instead of computing all scales at once with large filters, it
decomposes one octave at a time on successively coarser signals.

The scattering transform extends FWT by inserting a **pointwise nonlinearity**
(complex modulus or approximately average of phase ReLU) between scales and cascade wavelet transform. This cascade of
*wavelet → nonlinearity → wavelet → nonlinearity → ...* produces the
scattering coefficients of increasing order:

```
         x
         │
    Block 0: compute scale j=0 nodes in scatter tree
    ┌────┴────┐
    L₁        U₁        U₁ = σ(x ∗ ψ₀)
         │
    Block 1: scale j=1
    ┌────┴────┐
    L₂        U₂        
         │
    Block 2: scale j=2
    ┌────┴────┐
    L₃        U₃        
         │
            ...
```

At each block, the low-pass path (L) captures the local average at that
scale, while the bandpass path (U) propagates high-frequency detail to the
next block. The cascade of low-pass outputs across blocks produces
increasingly smoothed versions of higher-order scattering paths — the
concatenation of all block outputs approximates the full scattering
representation.

By implementing scattering as a cascade of identical single-scale blocks
rather than a monolithic multi-scale operator, this design inherits the
computational efficiency of FWT (linear in signal size) while enabling
learnable filters and cross-channel interactions at each scale.

## Usage (Python API)

```python
from scattering_net import ScatteringNet

# Traditional scattering (fixed wavelets + complex modulus)
model = ScatteringNet(
    n_blocks=3, in_channels=3, in_spatial=32,
    nb_classes=10, L=8,
    learnable=False, modulus_type='complex_modulus',
)

# Learnable scattering (trainable wavelets + phase ReLU with collapse)
model = ScatteringNet(
    n_blocks=3, in_channels=3, in_spatial=32,
    nb_classes=10, L=8,
    learnable=True, modulus_type='phase_relu',
)
```

## Quick Check (`quick_check.py`)

A lightweight single-phase training script for fast sanity checks on
CIFAR-10 or ImageNet. Trains everything end-to-end with AdamW.

```bash
cd new/

# Default: CIFAR-10, 2 blocks, frozen wavelets, complex modulus
python quick_check.py

# Learnable wavelets with phase_relu
python quick_check.py --learnable --modulus-type phase_relu

# Smaller model
python quick_check.py --n-blocks 2 --L 4 --kernel-size 5

# ImageNet (requires local dataset)
python quick_check.py --dataset imagenet --data-dir /path/to/imagenet
```

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `cifar10` or `imagenet` |
| `--data-dir` | `./data` | Root directory for dataset download |
| `--n-blocks` | `2` | Number of stacked scattering blocks |
| `--L` | `8` | Wavelet orientations per block |
| `--kernel-size` | `7` | Wavelet kernel size (must be odd) |
| `--modulus-type` | `complex_modulus` | `complex_modulus` or `phase_relu` |
| `--mixing-horizon` | `None` | Max input channels per wavelet filter (see below) |
| `--global-avg-pool` | off | Adaptive avg pool to 1x1 before classifier |
| `--lowpass-last` | off | Final block outputs only low-pass path |
| `--learnable` | off | Allow wavelet weight updates |
| `--epochs` | `150` | Total training epochs |
| `--batch-size` | `128` | Mini-batch size |
| `--lr` | `1e-3` | Initial learning rate (AdamW) |
| `--save-dir` | `runs` | Checkpoint directory |

Saves the best model to `<save-dir>/best.pth`.

---

## Experiment Pipeline (`train.py`)

The main training script implements the two-phase pipeline described in
`PROJECT_GUIDE.md`. Supports CIFAR-10 and ImageNet, stratified subset
sampling, multi-GPU (DataParallel / DDP), and JSON experiment logging.

### Quick start

```bash
cd new/

# CIFAR-10, full dataset
python train.py

# CIFAR-10, 10% subset
python train.py --train-size 0.1

# ImageNet (requires local dataset)
python train.py --dataset imagenet --data-dir /path/to/imagenet

# ImageNet, 8-GPU DDP (auto-detected via torchrun), 6 blocks L=6
torchrun --nproc_per_node=8 train.py --dataset imagenet --data-dir /datasets/ImageNet --n-blocks 6 --L 6 --epochs 90 --lr-head 0.01 --lr-extractor 0.0001 --global-batch-size 1024 --save-dir runs/imagenet_full_test
```

### Two-phase pipeline

**Phase A -- Train classifier head** (feature extractor frozen):
- Only BN + classifier parameters are updated
- CosineAnnealingLR schedule + early stopping
- Records `head_acc` at completion

**Phase B -- Joint fine-tune**:
- Unfreezes all scattering blocks
- Differential learning rates: `--lr-extractor` (default 0.001) for blocks,
  `--lr-head` (default 0.01) for BN + classifier
- Records `finetune_acc`, `finetune_gain`, `filter_delta_l2`,
  `filter_delta_relative`

### CLI arguments

#### Dataset

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `cifar10` or `imagenet` |
| `--data-dir` | `./data` | Root dir (ImageNet: parent of `train/` and `val/`) |
| `--train-size` | full | Subset size: float for fraction (0.1), int for count (5000) |

#### Model

| Argument | Default | Description |
|---|---|---|
| `--n-blocks` | `2` | Number of stacked scattering blocks |
| `--L` | `8` | Wavelet orientations per block |
| `--kernel-size` | `7` | Wavelet kernel size (must be odd) |
| `--modulus-type` | `complex_modulus` | `complex_modulus` or `phase_relu` |
| `--mixing-horizon` | `None` | Max input channels per wavelet filter (see below) |
| `--global-avg-pool` | off | Adaptive avg pool to 1x1 before classifier |
| `--lowpass-last` | off | Final block outputs only low-pass path |

#### Training phases

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `100` | Max epochs per phase |
| `--lr-head` | `1e-3` | BN + classifier learning rate (both phases) |
| `--lr-extractor` | `1e-4` | Feature extractor learning rate (Phase B) |

#### Training

| Argument | Default | Description |
|---|---|---|
| `--global-batch-size` | `128` | Global batch size (divided across GPUs in DDP) |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--patience` | `20` | Early-stopping patience (epochs) |
| `--workers` | `4` | DataLoader worker processes |
| `--seed` | none | Random seed for reproducibility |
| `--save-dir` | `runs` | Directory for checkpoints and JSON logs |
| `--print-freq` | `50` | Print stats every N batches |

#### Multi-GPU

DDP is auto-detected: when launched with `torchrun`, the `RANK` and
`WORLD_SIZE` environment variables are present and DDP is initialized
automatically. No flag is needed.

`--global-batch-size` specifies the total batch size across all GPUs. In
DDP mode it is divided evenly (must be divisible by world size). Learning
rates are **not** scaled — the same LR applies regardless of GPU count
because the global batch size stays constant.

### Data augmentation

| Dataset | Train | Validation |
|---|---|---|
| CIFAR-10 | RandomCrop(32, pad=4), HFlip, Normalize | Normalize only |
| ImageNet | RandomResizedCrop(224), HFlip, Normalize | Resize(256), CenterCrop(224), Normalize |

### Channel growth and memory

Each block multiplies channel count by `(1 + L)`:

| Blocks | L | Channels (CIFAR-10) | Classifier input | Notes |
|---|---|---|---|---|
| 2 | 8 | 3 → 27 → 243 | 243 × 8 × 8 = 15,552 | Practical on any GPU |
| 2 | 4 | 3 → 15 → 75 | 75 × 8 × 8 = 4,800 | Lightweight |
| 3 | 8 | 3 → 27 → 243 → 2,187 | 2,187 × 4 × 4 = 34,992 | High memory usage |
| 3 | 4 | 3 → 15 → 75 → 375 | 375 × 4 × 4 = 6,000 | Moderate |

### Mixing horizon (`--mixing-horizon`)

By default, each wavelet convolution is **dense**: every output filter sees
all `C_in` input channels, giving `O(C_in^2 * K^2)` parameters per Conv2d.
With exponential channel growth, this becomes infeasible at 4+ blocks (e.g.
block 3 alone needs ~15 GB for weights when `C_in=2187`).

`--mixing-horizon N` caps the number of input channels each wavelet filter
can see, using PyTorch grouped convolutions internally.  This reduces
parameter scaling from `O(C_in^2)` to `O(C_in * N)` per block:

| Block | C_in | Dense params (K=7, L=8) | Horizon=27 params |
|---|---|---|---|
| 0 | 3 | 7.5 K | 7.5 K (dense) |
| 1 | 27 | 607 K | 607 K (dense) |
| 2 | 243 | 49 M | 5.5 M |
| 3 | 2,187 | **3.98 B (~15 GB)** | 49 M (~188 MB) |
| 4 | 19,683 | **322.7 B (~1.2 TB)** | 443 M (~1.7 GB) |

When a block's `C_in` is already `<= mixing_horizon`, the convolution stays
fully dense (no grouping is applied), shown as "(dense)" in the startup log.

The scattering-transform initialization is preserved regardless of the
`mixing_horizon` setting -- at init time, each channel is processed
independently by its own wavelet, so the grouped structure makes no
difference. During training, off-diagonal entries within each group can
learn cross-channel interactions.

Recommended values:
- `None` (default): dense, for `<= 3` blocks
- `27`: for 4-5 blocks with `L=8` (each filter sees one block's worth of channels)
- `1`: fully grouped, pure per-channel scattering with no cross-channel learning

Example:

```bash
python train.py --n-blocks 5 --L 8 --mixing-horizon 27
```

### Classifier size reduction (`--global-avg-pool`, `--lowpass-last`)

With many blocks, the classifier (`nn.Linear`) can dominate total parameter
count because its input is `C_out * H * W` where both channel count and
spatial area contribute.  Two flags address this:

**`--global-avg-pool`**: Applies `AdaptiveAvgPool2d(1)` before the
classifier, collapsing spatial dimensions to 1x1.  The classifier input
becomes just `C_out` instead of `C_out * H * W`.

**`--lowpass-last`**: The final scattering block outputs only its low-pass
(phi) path, reducing output channels from `C_in * (1+L)` to `C_in`.  This
is theoretically motivated: in the standard scattering transform, band-pass
paths at the last order are "orphaned" (they would need another block to be
scattered further).

These flags can be combined.  Example with 5 blocks, L=6, ImageNet (224x224):

| Configuration | Classifier channels | Spatial | Classifier params |
|---|---|---|---|
| Default | 50,421 | 7x7 | 2,470M |
| `--global-avg-pool` | 50,421 | 1x1 | 50.4M |
| `--lowpass-last` | 7,203 | 7x7 | 353M |
| **Both flags** | 7,203 | 1x1 | **7.2M** |

Recommended for ImageNet or any setup with `>= 4` blocks:

```bash
python train.py --n-blocks 5 --L 6 --mixing-horizon 21 \
    --global-avg-pool --lowpass-last
```

### Experiment output

Each run produces a timestamped JSON file in `<save-dir>/`:

```json
{
  "config": { ... },
  "results": {
    "head_acc": 72.5,
    "head_err": 27.5,
    "baseline_acc": 72.5,
    "finetune_acc": 75.3,
    "finetune_gain": 2.8,
    "filter_delta_l2": 0.4521,
    "filter_delta_relative": 0.0312,
    "n_params_total": 770531,
    "n_params_extractor": 615001,
    "n_params_classifier": 155530,
    "n_params_trainable": 155530,
    "actual_train_samples": 50000,
    "test_samples": 10000
  },
  "curves": {
    "phase_a": [ {"epoch": 1, "train_loss": ..., "val_acc": ...}, ... ],
    "phase_b": [ ... ]
  }
}
```

### Checkpointing

Each phase saves checkpoints independently:
- `<save-dir>/phase_a/best.pth` -- best Phase A model
- `<save-dir>/phase_b/best.pth` -- best Phase B model

To load a checkpoint:

```python
import torch
from scattering_net import ScatteringNet

model = ScatteringNet(...)
ckpt = torch.load("runs/phase_b/best.pth", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
print(f"Best val acc: {ckpt['best_val_acc']:.2f}% (epoch {ckpt['epoch']})")
```

---

## Analysis (`analyze.py`)

After running experiments, use `analyze.py` to aggregate results and
generate plots. It reads all `results_*.json` files from a directory.

```bash
# Print summary table and save plots to runs/plots/
python analyze.py --results-dir runs

# Also show plots interactively
python analyze.py --results-dir runs --interactive

# Include per-experiment training curves
python analyze.py --results-dir runs --curves
```

### Generated plots

| Plot | File | Description |
|---|---|---|
| Train size vs. accuracy | `plots/plot_size_vs_accuracy.png` | Head-only and fine-tuned accuracy at each training set size |
| Train size vs. gain | `plots/plot_size_vs_gain.png` | Fine-tune gain (green = positive, red = negative) per subset |
| Filter delta vs. gain | `plots/plot_delta_vs_gain.png` | Scatter plot with Pearson r correlation |
| Training curves | `plots/curves/curves_exp*.png` | Per-experiment val_acc and train_loss over epochs |

### Typical workflow

```bash
# 1. Run experiments at different training set sizes
python train.py --train-size 0.05 --save-dir runs/s005
python train.py --train-size 0.1  --save-dir runs/s010
python train.py --train-size 0.3  --save-dir runs/s030
python train.py --train-size 1.0  --save-dir runs/s100

# 2. Collect all JSONs into one directory
#    (each run already produces results_*.json in its save-dir)

# 3. Analyze
python analyze.py --results-dir runs/s005
#    or point at a parent dir containing multiple results files
```

Requires `pandas` and `matplotlib`.
