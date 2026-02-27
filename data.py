"""Dataset construction and stratified subset sampling.

Supports CIFAR-10 and ImageNet with standard augmentation pipelines.
"""

import os

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# ---------------------------------------------------------------------------
# Per-dataset constants
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "cifar10": {
        "in_channels": 3,
        "in_spatial": 32,
        "nb_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "imagenet": {
        "in_channels": 3,
        "in_spatial": 224,
        "nb_classes": 1000,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}


# ---------------------------------------------------------------------------
# Stratified subset sampling
# ---------------------------------------------------------------------------

def _get_targets(dataset):
    """Extract integer target labels from a dataset."""
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    raise AttributeError(
        f"{type(dataset).__name__} has no .targets or .labels attribute"
    )


def stratified_subset(dataset, train_size, seed=42):
    """Return a class-balanced Subset of *dataset*.

    Parameters
    ----------
    dataset : torchvision dataset
        Must expose a ``.targets`` or ``.labels`` attribute.
    train_size : float or int
        If float in (0, 1]: fraction of the dataset to keep.
        If int > 1: absolute number of samples to keep.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    torch.utils.data.Subset
    """
    if isinstance(train_size, float) and train_size >= 1.0:
        return dataset
    if isinstance(train_size, int) and train_size >= len(dataset):
        return dataset

    targets = _get_targets(dataset)
    classes = np.unique(targets)
    rng = np.random.RandomState(seed)

    indices = []
    for c in classes:
        c_idx = np.where(targets == c)[0]
        if isinstance(train_size, float):
            n = max(1, int(len(c_idx) * train_size))
        else:
            n = min(train_size // len(classes), len(c_idx))
            n = max(1, n)
        chosen = rng.choice(c_idx, size=n, replace=False)
        indices.extend(chosen.tolist())

    rng.shuffle(indices)
    return torch.utils.data.Subset(dataset, indices)


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

def _cifar10_transforms(mean, std):
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
    return train_tf, val_tf


def _imagenet_transforms(mean, std):
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
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataloaders(args):
    """Build train and validation dataloaders.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: dataset, data_dir, train_size, batch_size, workers,
        seed, distributed.

    Returns
    -------
    train_loader, val_loader, info : dict
        *info* contains ``in_channels``, ``in_spatial``, ``nb_classes``,
        ``actual_train_samples``, ``test_samples``.
    """
    cfg = DATASET_CONFIGS[args.dataset]
    mean, std = cfg["mean"], cfg["std"]

    if args.dataset == "cifar10":
        train_tf, val_tf = _cifar10_transforms(mean, std)
        train_ds = datasets.CIFAR10(
            args.data_dir, train=True, download=True, transform=train_tf)
        val_ds = datasets.CIFAR10(
            args.data_dir, train=False, download=True, transform=val_tf)

    elif args.dataset == "imagenet":
        train_tf, val_tf = _imagenet_transforms(mean, std)
        traindir = os.path.join(args.data_dir, "train")
        valdir = os.path.join(args.data_dir, "val")
        train_ds = datasets.ImageFolder(traindir, transform=train_tf)
        val_ds = datasets.ImageFolder(valdir, transform=val_tf)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Subset sampling
    if args.train_size is not None:
        seed = args.seed if args.seed is not None else 42
        train_ds = stratified_subset(train_ds, args.train_size, seed=seed)

    # Samplers
    train_sampler = None
    val_sampler = None
    shuffle_train = True

    if dist.is_available() and dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds, shuffle=False)
        shuffle_train = False

    persist = args.workers > 0
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle_train,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        persistent_workers=persist)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler,
        persistent_workers=persist)

    info = {
        "in_channels": cfg["in_channels"],
        "in_spatial": cfg["in_spatial"],
        "nb_classes": cfg["nb_classes"],
        "actual_train_samples": len(train_ds),
        "test_samples": len(val_ds),
    }
    return train_loader, val_loader, info
