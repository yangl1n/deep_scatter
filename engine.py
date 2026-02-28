"""Training / evaluation loops, early stopping, and parameter-delta metrics."""

import time

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# AverageMeter (adapted from original main.py)
# ---------------------------------------------------------------------------

class AverageMeter:
    """Tracks running average of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------------------------
# Top-k accuracy (adapted from original main.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Return list of top-k accuracies (in %) for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


# ---------------------------------------------------------------------------
# Single-epoch train / eval
# ---------------------------------------------------------------------------

def run_epoch(loader, model, criterion, optimizer, device,
              is_training, print_freq=50, verbose=True, anchor_reg=None):
    """Run one epoch of training or evaluation.

    Parameters
    ----------
    verbose : bool
        If False, suppress batch-level prints (use for non-main DDP ranks).
    anchor_reg : AnchorPenalty or None
        If provided (and training), its output is added to the loss before
        ``.backward()``.  Expects the *raw* (unwrapped) model as argument.

    Returns
    -------
    avg_loss, top1_acc, top5_acc : float
    """
    if is_training:
        model.train()
    else:
        model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    start = time.time()

    ctx = torch.enable_grad() if is_training else torch.inference_mode()
    with ctx:
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if is_training and anchor_reg is not None:
                raw = model.module if hasattr(model, "module") else model
                loss = loss + anchor_reg(raw)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            bsz = targets.size(0)
            losses.update(loss.item(), bsz)
            top1.update(acc1, bsz)
            top5.update(acc5, bsz)

            if verbose and print_freq > 0 and (i + 1) % print_freq == 0:
                elapsed = time.time() - start
                print(f"  [{i+1}/{len(loader)}]  "
                      f"loss {losses.avg:.4f}  "
                      f"acc@1 {top1.avg:.2f}%  "
                      f"acc@5 {top5.avg:.2f}%  "
                      f"({elapsed:.1f}s)")

    # All-reduce metrics so every rank sees the same global values.
    # Without this, each DDP rank only sees its own shard of the data,
    # which would cause early-stopping and best-checkpoint decisions to
    # desync across ranks (leading to hangs).
    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor(
            [losses.sum, losses.count,
             top1.sum, top1.count,
             top5.sum, top5.count],
            device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        losses.avg = (stats[0] / stats[1]).item()
        top1.avg = (stats[2] / stats[3]).item()
        top5.avg = (stats[4] / stats[5]).item()

    return losses.avg, top1.avg, top5.avg


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        How many epochs to wait after the last improvement.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, metric):
        """Update with the latest metric value.

        Returns ``True`` if training should stop.
        """
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# L2 anchor penalty (L2-SP regularization)
# ---------------------------------------------------------------------------

class AnchorPenalty:
    """Penalise extractor weight drift from an initial snapshot.

    Computes ``lam * sum_i ||theta_i - theta_i^0||^2`` over all extractor
    parameters, where theta^0 is the snapshot taken before fine-tuning.

    Parameters
    ----------
    named_params_snapshot : dict
        Output of ``snapshot_extractor_params`` (name -> CPU tensor).
    lam : float
        Penalty strength (lambda).
    device : torch.device
        Device to store anchor tensors on (should match model device).
    """

    def __init__(self, named_params_snapshot, lam, device):
        self.anchor = {k: v.to(device) for k, v in named_params_snapshot.items()}
        self.lam = lam

    def __call__(self, raw_model):
        penalty = sum(
            (p - self.anchor[n]).pow(2).sum()
            for n, p in raw_model.blocks.named_parameters()
            if n in self.anchor
        )
        return self.lam * penalty


# ---------------------------------------------------------------------------
# Parameter snapshots and delta computation
# ---------------------------------------------------------------------------

def snapshot_extractor_params(model):
    """Deep-copy feature extractor (``model.blocks``) parameters.

    *model* should be the raw (unwrapped) ``ScatteringNet``.

    Returns
    -------
    dict : ``{name: tensor}`` on CPU.
    """
    snap = {}
    for name, param in model.blocks.named_parameters():
        snap[name] = param.detach().cpu().clone()
    return snap


def compute_filter_delta(snapshot_before, model_after):
    """Compute L2 distance and relative change between two parameter states.

    Parameters
    ----------
    snapshot_before : dict
        Output of ``snapshot_extractor_params`` taken before fine-tuning.
    model_after : nn.Module
        Raw (unwrapped) model after fine-tuning.

    Returns
    -------
    filter_delta_l2, filter_delta_relative : float
    """
    delta_sq = 0.0
    norm_sq = 0.0
    for name, param in model_after.blocks.named_parameters():
        before = snapshot_before[name]
        after = param.detach().cpu()
        delta_sq += (after - before).pow(2).sum().item()
        norm_sq += before.pow(2).sum().item()

    delta_l2 = delta_sq ** 0.5
    norm_l2 = norm_sq ** 0.5
    relative = delta_l2 / norm_l2 if norm_l2 > 0 else 0.0
    return delta_l2, relative
