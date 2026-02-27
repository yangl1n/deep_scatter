"""Quick single-phase training of ScatteringNet on CIFAR-10 or ImageNet.

A lightweight script for fast sanity checks. For the full two-phase
experiment pipeline (Phase A head training + Phase B fine-tuning),
use ``train.py`` instead.

Usage examples
--------------
# Default CIFAR-10, 2 blocks, complex modulus, frozen wavelets:
    python quick_check.py

# Learnable wavelets:
    # accuracy: 85.41%
    python quick_check.py --learnable --kernel-size 5
    # accuracy: 85.35%
    python quick_check.py --learnable
    # accuracy: 84.80%
    python quick_check.py --learnable --modulus-type phase_relu
    # accuracy: 86.36%
    python quick_check.py --learnable --modulus-type phase_relu --n-blocks 3
    # accuracy: 82.5%
    python quick_check.py --learnable --modulus-type phase_relu --n-blocks 3 --L 4
    # accuracy: 84.32%
    python quick_check.py --learnable --kernel-size 5 --n-blocks 4 --lowpass-last --mixing-horizon 27

# ImageNet:
    python quick_check.py --dataset imagenet --data-dir /datasets/ImageNet --n-blocks 4 --L 4 
    python quick_check.py --dataset imagenet --data-dir /datasets/ImageNet --n-blocks 4 --L 4 --lowpass-last
"""

import argparse
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from scattering_net import ScatteringNet


def parse_args():
    p = argparse.ArgumentParser(description="Train ScatteringNet")

    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "imagenet"])
    p.add_argument("--data-dir", default="./data", help="root dir for dataset download")

    p.add_argument("--n-blocks", default=2, type=int)
    p.add_argument("--L", default=8, type=int, help="wavelet orientations per block")
    p.add_argument("--kernel-size", default=7, type=int, help="wavelet kernel size (odd)")
    p.add_argument("--modulus-type", default="complex_modulus",
                   choices=["complex_modulus", "phase_relu"])
    p.add_argument("--mixing-horizon", default=None, type=int,
                   help="max input channels each wavelet filter can see; "
                        "None=dense. Recommended 27 for >=4 blocks")
    p.add_argument("--global-avg-pool", action="store_true",
                   help="apply adaptive avg pool to 1x1 before classifier")
    p.add_argument("--lowpass-last", action="store_true",
                   help="final block outputs only low-pass (phi) path")
    p.add_argument("--random-init", action="store_true",
                   help="skip wavelet initialization, use Kaiming random init")
    p.add_argument("--learnable", action="store_true",
                   help="allow wavelet weights to be updated during training")

    p.add_argument("--epochs", default=150, type=int)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    p.add_argument("--weight-decay", default=1e-4, type=float)

    p.add_argument("--workers", default=4, type=int, help="dataloader workers")
    p.add_argument("--seed", default=None, type=int, help="random seed")
    p.add_argument("--save-dir", default="runs", help="directory for checkpoints")
    p.add_argument("--print-freq", default=50, type=int, help="batch print frequency")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloaders(args):
    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_ds = datasets.CIFAR10(args.data_dir, train=True,
                                    download=True, transform=train_tf)
        val_ds = datasets.CIFAR10(args.data_dir, train=False,
                                  download=True, transform=val_tf)
        in_channels, in_spatial, nb_classes = 3, 32, 10

    else:  # imagenet
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        traindir = os.path.join(args.data_dir, "train")
        valdir = os.path.join(args.data_dir, "val")
        train_ds = datasets.ImageFolder(traindir, transform=train_tf)
        val_ds = datasets.ImageFolder(valdir, transform=val_tf)
        in_channels, in_spatial, nb_classes = 3, 224, 1000

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, in_channels, in_spatial, nb_classes


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def run_epoch(loader, model, criterion, optimizer, device, is_training, print_freq):
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    ctx = torch.enable_grad() if is_training else torch.no_grad()
    with ctx:
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if print_freq > 0 and (i + 1) % print_freq == 0:
                elapsed = time.time() - start
                print(f"  [{i+1}/{len(loader)}]  "
                      f"loss {total_loss/total:.4f}  "
                      f"acc {100.*correct/total:.2f}%  "
                      f"({elapsed:.1f}s)")

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("Seeded run: cudnn.deterministic=True, may be slower.")

    if args.n_blocks >= 3 and args.L >= 8:
        ch = 3
        final_ch = ch * (1 + args.L) ** args.n_blocks
        warnings.warn(
            f"Channel count will grow to {final_ch} after {args.n_blocks} blocks "
            f"with L={args.L}. This may require significant GPU memory."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, in_channels, in_spatial, nb_classes = \
        build_dataloaders(args)
    print(f"Dataset: {args.dataset}  |  "
          f"train {len(train_loader.dataset)}  "
          f"val {len(val_loader.dataset)}")

    # Model
    model = ScatteringNet(
        n_blocks=args.n_blocks,
        in_channels=in_channels,
        in_spatial=in_spatial,
        nb_classes=nb_classes,
        L=args.L,
        kernel_size=args.kernel_size,
        learnable=args.learnable,
        modulus_type=args.modulus_type,
        mixing_horizon=args.mixing_horizon,
        global_avg_pool=args.global_avg_pool,
        lowpass_last=args.lowpass_last,
        random_init=args.random_init,
    ).to(device)

    ps = model.param_summary()
    bs = model.block_summary()
    print(f"Model: {args.n_blocks} blocks, L={args.L}, "
          f"modulus={args.modulus_type}, learnable={args.learnable}, "
          f"mixing_horizon={args.mixing_horizon}, "
          f"gap={args.global_avg_pool}, lowpass_last={args.lowpass_last}")
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

    # Optimizer & scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)

    best_val_acc = 0.0
    print(f"\nTraining for {args.epochs} epochs\n")

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs}  lr={lr:.6f}")

        train_loss, train_acc = run_epoch(
            train_loader, model, criterion, optimizer, device,
            is_training=True, print_freq=args.print_freq)

        val_loss, val_acc = run_epoch(
            val_loader, model, criterion, None, device,
            is_training=False, print_freq=0)

        scheduler.step()

        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)
        marker = " *" if is_best else ""

        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.2f}%{marker}")

        if is_best:
            ckpt = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "args": vars(args),
            }
            path = os.path.join(args.save_dir, "best.pth")
            torch.save(ckpt, path)
            print(f"  Saved best checkpoint -> {path}")

    print(f"\nDone. Best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
