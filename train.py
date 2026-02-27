"""Two-phase training pipeline for ScatteringNet.

Phase A: freeze feature extractor, train classifier head only.
Phase B: joint fine-tune — unfreeze extractor with differential learning
         rates (lower for extractor, higher for head).

Usage examples
--------------
# CIFAR-10, full dataset (single GPU):
    python train.py --dataset cifar10

# CIFAR-10, 10 % subset:
    python train.py --dataset cifar10 --train-size 0.1

# ImageNet, 1024 global batch size DDP (auto-detected via torchrun):
# 8 gpus, each with 128 batch size
    torchrun --nproc_per_node=8 train.py --dataset imagenet --global-batch-size 1024 --data-dir /datasets/ImageNet --n-blocks 4 --L 4 --lowpass-last --lr-epochs 20 --save-dir runs_test_2
"""

import argparse
import os
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from scattering_net import ScatteringNet
from data import build_dataloaders
from engine import (run_epoch, EarlyStopping,
                    snapshot_extractor_params, compute_filter_delta)
from utils import (get_raw_model, save_checkpoint, load_checkpoint,
                   ExperimentLogger, init_distributed, cleanup_distributed,
                   get_world_size, is_main_process)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-phase ScatteringNet training pipeline")

    # Dataset
    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset", default="cifar10",
                   choices=["cifar10", "imagenet"])
    g.add_argument("--data-dir", default="./data",
                   help="root directory for dataset (ImageNet: parent of "
                        "train/ and val/)")
    g.add_argument("--train-size", default=None, type=float,
                   help="training subset size: float for fraction (e.g. 0.1),"
                        " int for absolute count (e.g. 5000). "
                        "Default: use full training set")

    # Model
    g = p.add_argument_group("Model")
    g.add_argument("--n-blocks", default=2, type=int)
    g.add_argument("--L", default=8, type=int,
                   help="wavelet orientations per block")
    g.add_argument("--kernel-size", default=7, type=int,
                   help="wavelet kernel size (must be odd)")
    g.add_argument("--modulus-type", default="complex_modulus",
                   choices=["complex_modulus", "phase_relu"])
    g.add_argument("--mixing-horizon", default=None, type=int,
                   help="max input channels each wavelet filter can see; "
                        "None=dense. Recommended 27 for >=4 blocks")
    g.add_argument("--global-avg-pool", action="store_true",
                   help="apply adaptive avg pool to 1x1 before classifier")
    g.add_argument("--lowpass-last", action="store_true",
                   help="final block outputs only low-pass (phi) path")
    g.add_argument("--random-init", action="store_true",
                   help="skip wavelet initialization, use Kaiming random init")
    g.add_argument("--joint", action="store_true",
                   help="single-phase joint training (skip Phase A/B split)")

    # Training phases
    g = p.add_argument_group("Training phases")
    g.add_argument("--epochs", default=100, type=int,
                   help="max epochs per phase (default for both phases)")
    g.add_argument("--lr-epochs", default=None, type=int,
                   help="max epochs for Phase A (head-only); "
                        "defaults to --epochs if not given")
    g.add_argument("--lr-head", default=1e-3, type=float,
                   help="classifier head learning rate (both phases)")
    g.add_argument("--lr-extractor", default=1e-4, type=float,
                   help="feature extractor learning rate (Phase B)")

    # Common training
    g = p.add_argument_group("Training")
    g.add_argument("--global-batch-size", default=128, type=int, dest="batch_size",
                   help="global batch size (divided across GPUs in DDP)")
    g.add_argument("--weight-decay", default=1e-4, type=float)
    g.add_argument("--patience", default=20, type=int,
                   help="early-stopping patience (epochs)")
    g.add_argument("--workers", default=4, type=int)
    g.add_argument("--seed", default=None, type=int)
    g.add_argument("--save-dir", default="runs", type=str)
    g.add_argument("--print-freq", default=50, type=int)

    args = p.parse_args()

    # Coerce train_size: if it looks like an int, cast it
    if args.train_size is not None and args.train_size == int(args.train_size) and args.train_size > 1:
        args.train_size = int(args.train_size)

    if args.lr_epochs is None:
        args.lr_epochs = args.epochs

    return args


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _run_phase(tag, model, train_loader, val_loader, optimizer, scheduler,
               criterion, device, epochs, patience, print_freq, save_dir,
               logger):
    """Generic epoch loop with early stopping and checkpoint saving.

    Returns best_val_acc (top-1).
    """
    stopper = EarlyStopping(patience=patience)
    best_acc = 0.0
    verbose = is_main_process()

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        if verbose:
            print(f"[{tag}] Epoch {epoch}/{epochs}  lr={lr:.6f}")

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc1, _ = run_epoch(
            train_loader, model, criterion, optimizer, device,
            is_training=True, print_freq=print_freq, verbose=verbose)

        val_loss, val_acc1, val_acc5 = run_epoch(
            val_loader, model, criterion, None, device,
            is_training=False, print_freq=0, verbose=False)

        scheduler.step()

        is_best = val_acc1 > best_acc
        best_acc = max(val_acc1, best_acc)

        if verbose:
            marker = " *" if is_best else ""
            print(f"  Train  loss={train_loss:.4f}  acc@1={train_acc1:.2f}%")
            print(f"  Val    loss={val_loss:.4f}  acc@1={val_acc1:.2f}%  "
                  f"acc@5={val_acc5:.2f}%{marker}")

            phase_key = {"Phase-A": "phase_a", "Phase-B": "phase_b"}.get(tag, "joint")
            logger.log_epoch(phase_key, epoch, train_loss, val_loss,
                             val_acc1, val_acc5)

            save_checkpoint({
                "epoch": epoch,
                "state_dict": get_raw_model(model).state_dict(),
                "best_val_acc": best_acc,
            }, is_best, save_dir=save_dir)

        # Ensure rank 0 finishes writing before any rank proceeds
        if dist.is_initialized():
            dist.barrier()

        if stopper.step(val_acc1):
            if verbose:
                print(f"  Early stopping triggered after {epoch} epochs.")
            break

    # Wait for rank 0 to finish any final writes before loading
    if dist.is_initialized():
        dist.barrier()

    # Reload best checkpoint on all ranks so model state stays in sync
    best_path = os.path.join(save_dir, "best.pth")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model)

    return best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    distributed = init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = get_world_size()
    if world_size > 1:
        assert args.batch_size % world_size == 0, (
            f"global-batch-size ({args.batch_size}) must be divisible "
            f"by world_size ({world_size})"
        )
        args.batch_size = args.batch_size // world_size
        if is_main_process():
            print(f"Global batch {args.batch_size * world_size} / "
                  f"{world_size} GPUs = {args.batch_size} per GPU")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        if is_main_process():
            warnings.warn("Seeded run: cudnn.deterministic=True, may slow down training.")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        print(f"Device: {device}")

    # ---- Data ----
    train_loader, val_loader, info = build_dataloaders(args)
    if is_main_process():
        print(f"Dataset: {args.dataset}  |  "
              f"train_size={args.train_size}  "
              f"train {info['actual_train_samples']}  "
              f"val {info['test_samples']}")

    # ---- Model ----
    model = ScatteringNet(
        n_blocks=args.n_blocks,
        in_channels=info["in_channels"],
        in_spatial=info["in_spatial"],
        nb_classes=info["nb_classes"],
        L=args.L,
        kernel_size=args.kernel_size,
        learnable=args.joint,
        modulus_type=args.modulus_type,
        mixing_horizon=args.mixing_horizon,
        global_avg_pool=args.global_avg_pool,
        lowpass_last=args.lowpass_last,
        random_init=args.random_init,
    ).to(device)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])

    raw = get_raw_model(model)
    ps = raw.param_summary()
    bs = raw.block_summary()
    if is_main_process():
        init_tag = "random" if args.random_init else "wavelet"
        mode_tag = "joint" if args.joint else "two-phase"
        print(f"Model: {args.n_blocks} blocks, L={args.L}, "
              f"modulus={args.modulus_type}, "
              f"mixing_horizon={args.mixing_horizon}, "
              f"gap={args.global_avg_pool}, lowpass_last={args.lowpass_last}")
        print(f"  init={init_tag}, training={mode_tag}")
        print(f"Parameters: {ps['total']:,} total, {ps['trainable']:,} trainable")
        print(f"  Extractor (blocks): {ps['extractor']:,}")
        print(f"  Classifier (head):  {ps['classifier']:,}")
        print(f"  {'Block':<6} {'In_ch':>8} {'Out_ch':>8} {'Horizon':>8}")
        for row in bs:
            tags = []
            if row["mixing_horizon"] == row["in_channels"]:
                tags.append("dense")
            req = row.get("requested_horizon")
            if req is not None and req != row["mixing_horizon"]:
                tags.append(f"requested {req}")
            if row.get("lowpass_only"):
                tags.append("lowpass only")
            suffix = f"  ({', '.join(tags)})" if tags else ""
            print(f"  {row['block']:<6} {row['in_channels']:>8} "
                  f"{row['out_channels']:>8} {row['mixing_horizon']:>8}{suffix}")

    criterion = nn.CrossEntropyLoss().to(device)
    logger = ExperimentLogger(args.save_dir, args)
    logger.set_result("train_size", args.train_size)
    logger.set_result("actual_train_samples", info["actual_train_samples"])
    logger.set_result("test_samples", info["test_samples"])
    logger.set_result("world_size", world_size)
    logger.set_result("lr_head", args.lr_head)
    logger.set_result("lr_extractor", args.lr_extractor)
    logger.set_result("block_summary", bs)
    for k, v in ps.items():
        logger.set_result(f"n_params_{k}", v)

    cudnn.benchmark = True
    head_params = list(raw.bn.parameters()) + list(raw.classifier.parameters())

    if args.joint:
        _run_joint(args, model, raw, head_params, train_loader, val_loader,
                   criterion, device, distributed, local_rank, logger)
    else:
        _run_two_phase(args, model, raw, head_params, train_loader, val_loader,
                       criterion, device, distributed, local_rank, logger)

    if is_main_process():
        logger.save()

    cleanup_distributed()


def _run_joint(args, model, raw, head_params, train_loader, val_loader,
               criterion, device, distributed, local_rank, logger):
    """Single-phase joint training — all parameters trainable from the start."""
    save_dir = os.path.join(args.save_dir, "joint")

    if is_main_process():
        print("\n" + "=" * 60)
        print("Joint training (all parameters)")
        print("=" * 60 + "\n")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr_head, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_acc = _run_phase(
        "Joint", model, train_loader, val_loader,
        optimizer, scheduler, criterion, device,
        epochs=args.epochs, patience=args.patience,
        print_freq=args.print_freq, save_dir=save_dir,
        logger=logger)

    logger.set_result("joint_acc", best_acc)
    logger.set_result("joint_err", round(100.0 - best_acc, 4))
    if is_main_process():
        print(f"\nJoint training complete. acc = {best_acc:.2f}%")


def _run_two_phase(args, model, raw, head_params, train_loader, val_loader,
                   criterion, device, distributed, local_rank, logger):
    """Standard two-phase pipeline: freeze extractor, then fine-tune."""
    phase_a_dir = os.path.join(args.save_dir, "phase_a")
    phase_b_dir = os.path.join(args.save_dir, "phase_b")

    # ================================================================
    # Phase A: train classifier head (feature extractor frozen)
    # ================================================================
    if is_main_process():
        print("\n" + "=" * 60)
        print("Phase A: Train classifier head (extractor frozen)")
        print("=" * 60 + "\n")

    for block in raw.blocks:
        block.freeze()

    optimizer_a = optim.AdamW(
        head_params,
        lr=args.lr_head, weight_decay=args.weight_decay)
    scheduler_a = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a, T_max=args.lr_epochs)

    head_acc = _run_phase(
        "Phase-A", model, train_loader, val_loader,
        optimizer_a, scheduler_a, criterion, device,
        epochs=args.lr_epochs, patience=args.patience,
        print_freq=args.print_freq, save_dir=phase_a_dir,
        logger=logger)

    logger.set_result("head_acc", head_acc)
    logger.set_result("head_err", round(100.0 - head_acc, 4))
    if is_main_process():
        print(f"\nPhase A complete. head_acc = {head_acc:.2f}%\n")

    # ================================================================
    # Phase B: joint fine-tune (differential LR)
    # ================================================================
    if is_main_process():
        print("=" * 60)
        print("Phase B: Joint fine-tune")
        print("=" * 60 + "\n")

    _, baseline_acc, _ = run_epoch(
        val_loader, model, criterion, None, device,
        is_training=False, print_freq=0, verbose=False)
    logger.set_result("baseline_acc", baseline_acc)
    logger.set_result("baseline_err", round(100.0 - baseline_acc, 4))
    if is_main_process():
        print(f"Baseline acc (before fine-tune): {baseline_acc:.2f}%")

    param_snapshot = snapshot_extractor_params(raw)

    for block in raw.blocks:
        block.unfreeze()

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            raw, device_ids=[local_rank])

    optimizer_b = optim.AdamW(
        [{"params": list(raw.blocks.parameters()),
          "lr": args.lr_extractor},
         {"params": head_params,
          "lr": args.lr_head}],
        weight_decay=args.weight_decay)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, T_max=args.epochs)

    finetune_acc = _run_phase(
        "Phase-B", model, train_loader, val_loader,
        optimizer_b, scheduler_b, criterion, device,
        epochs=args.epochs, patience=args.patience,
        print_freq=args.print_freq, save_dir=phase_b_dir,
        logger=logger)

    finetune_gain = finetune_acc - baseline_acc
    delta_l2, delta_rel = compute_filter_delta(param_snapshot, raw)

    logger.set_result("finetune_acc", finetune_acc)
    logger.set_result("finetune_err", round(100.0 - finetune_acc, 4))
    logger.set_result("finetune_gain", finetune_gain)
    logger.set_result("filter_delta_l2", delta_l2)
    logger.set_result("filter_delta_relative", delta_rel)

    if is_main_process():
        print(f"\nPhase B complete.")
        print(f"  finetune_acc   = {finetune_acc:.2f}%")
        print(f"  finetune_gain  = {finetune_gain:+.2f}%")
        print(f"  filter_delta   = {delta_l2:.4f} "
              f"(relative: {delta_rel:.4f})")


if __name__ == "__main__":
    main()
