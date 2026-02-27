"""Checkpoint I/O, experiment JSON logger, distributed helpers."""

import json
import os
import shutil
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model wrapper helpers
# ---------------------------------------------------------------------------

def get_raw_model(model):
    """Unwrap DDP to get the underlying ScatteringNet."""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state, is_best, save_dir):
    """Save *state* as ``last.pth``; copy to ``best.pth`` when *is_best*."""
    os.makedirs(save_dir, exist_ok=True)
    last_path = os.path.join(save_dir, "last.pth")
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(save_dir, "best.pth")
        shutil.copyfile(last_path, best_path)


def load_checkpoint(path, model, optimizer=None):
    """Load a checkpoint into *model* (and optionally *optimizer*).

    Returns the checkpoint dict.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    get_raw_model(model).load_state_dict(ckpt["state_dict"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# ---------------------------------------------------------------------------
# Experiment JSON logger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """Accumulate per-epoch metrics and final results; write to JSON.

    JSON schema follows PROJECT_GUIDE.md section 4.
    """

    def __init__(self, save_dir, args):
        self.save_dir = save_dir
        self.config = vars(args) if args is not None else {}
        self.results = {}
        self.curves = {"phase_a": [], "phase_b": []}

    def log_epoch(self, phase, epoch, train_loss, val_loss, val_acc, val_acc5):
        """Append one epoch's metrics to the training curve."""
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 3),
            "val_acc5": round(val_acc5, 3),
        }
        self.curves[phase].append(entry)

    def set_result(self, key, value):
        """Store a final metric (e.g. ``head_acc``, ``finetune_gain``)."""
        if isinstance(value, float):
            value = round(value, 4)
        self.results[key] = value

    def save(self):
        """Write everything to a timestamped JSON file."""
        os.makedirs(self.save_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.save_dir, f"results_{stamp}.json")

        payload = {
            "config": self.config,
            "results": self.results,
            "curves": self.curves,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Experiment log saved -> {path}")
        return path


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_distributed():
    """Auto-detect and initialize DDP from ``torchrun`` env vars.

    Returns True if a distributed process group was created, False
    otherwise (single-GPU or CPU run).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return True
    return False


def cleanup_distributed():
    """Destroy the process group if it was initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_world_size():
    """Return the number of processes (1 when not distributed)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    """True on rank 0 or when not in distributed mode."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
