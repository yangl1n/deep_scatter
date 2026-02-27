"""Scattering transform model built by stacking ScatteringBlocks.

Stacks J single-scale ScatteringBlock modules to approximate a J-order
scattering transform, followed by a linear classifier.
"""

import warnings

import torch.nn as nn

from spatial_scattering import ScatteringBlock


class ScatteringNet(nn.Module):
    """Multi-order scattering network.

    Parameters
    ----------
    n_blocks : int
        Number of stacked scattering blocks (= max scattering order).
    in_channels : int
        Input image channels (3 for RGB, 1 for grayscale).
    in_spatial : int
        Input spatial size (32 for CIFAR, 28 for MNIST).
    nb_classes : int
        Number of output classes.
    L : int
        Number of wavelet orientations per block.
    kernel_size : int
        Wavelet Conv2d kernel size (odd).
    learnable : bool
        Whether wavelet filters are learnable.
    modulus_type : str
        ``'phase_relu'`` or ``'complex_modulus'``.
    mixing_horizon : int or None
        Caps how many input channels each wavelet filter can see per block.
        ``None`` = dense convolutions (all channels visible).  Recommended
        to set (e.g. 27) when ``n_blocks >= 4`` to control memory growth.
    global_avg_pool : bool
        If True, apply adaptive average pooling to 1x1 before the
        classifier, collapsing spatial dimensions.  Dramatically reduces
        classifier size for large spatial outputs.
    lowpass_last : bool
        If True, the final block outputs only its low-pass (phi) path,
        reducing output channels from ``C_in * (1+L)`` to ``C_in``.
        This is more faithful to the standard scattering transform, which
        only uses band-pass paths as input to the next scattering order.
    """

    def __init__(self, n_blocks=3, in_channels=3, in_spatial=32,
                 nb_classes=10, L=8, kernel_size=7,
                 learnable=False, modulus_type='complex_modulus',
                 mixing_horizon=None, global_avg_pool=False,
                 lowpass_last=False):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        min_spatial = 2 ** n_blocks
        assert in_spatial >= min_spatial, (
            f"in_spatial={in_spatial} too small for {n_blocks} blocks "
            f"(need at least {min_spatial})"
        )
        self.n_blocks = n_blocks
        self.global_avg_pool = global_avg_pool
        self.lowpass_last = lowpass_last

        blocks = []
        ch = in_channels
        spatial = in_spatial
        for i in range(n_blocks):
            is_last = (i == n_blocks - 1)
            lp_only = lowpass_last and is_last
            block = ScatteringBlock(in_channels=ch, L=L,
                                    kernel_size=kernel_size,
                                    learnable=learnable,
                                    modulus_type=modulus_type,
                                    mixing_horizon=mixing_horizon,
                                    lowpass_only=lp_only)
            blocks.append(block)
            ch = block.out_channels
            spatial //= 2

        self.blocks = nn.ModuleList(blocks)

        adjusted = [
            row for row in self.block_summary()
            if row["requested_horizon"] is not None
            and row["mixing_horizon"] != row["requested_horizon"]
        ]
        if adjusted:
            details = ", ".join(
                f"block {r['block']}: {r['requested_horizon']}→{r['mixing_horizon']}"
                for r in adjusted
            )
            warnings.warn(
                f"mixing_horizon adjusted for divisibility: {details}"
            )

        self._classifier_channels = ch
        self.bn = nn.BatchNorm2d(ch)
        if global_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(ch, nb_classes)
        else:
            self.avg_pool = None
            self.classifier = nn.Linear(ch * spatial * spatial, nb_classes)

    def block_summary(self):
        """Per-block channel and mixing-horizon info for logging."""
        rows = []
        for i, b in enumerate(self.blocks):
            rows.append({
                "block": i,
                "in_channels": b.in_channels,
                "out_channels": b.out_channels,
                "mixing_horizon": b.actual_horizon,
                "requested_horizon": b.requested_horizon,
                "lowpass_only": b.lowpass_only,
            })
        return rows

    def param_summary(self):
        """Return a dict with parameter counts by component.

        Keys: ``extractor``, ``classifier``, ``total``, ``trainable``.
        """
        n_ext = sum(p.numel() for p in self.blocks.parameters())
        n_cls = (sum(p.numel() for p in self.bn.parameters())
                 + sum(p.numel() for p in self.classifier.parameters()))
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "extractor": n_ext,
            "classifier": n_cls,
            "total": n_ext + n_cls,
            "trainable": n_train,
        }

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = self.bn(x)
        x = x.flatten(1)
        return self.classifier(x)
