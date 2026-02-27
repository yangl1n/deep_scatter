"""Spatial-domain scattering building block.

Reimplements the wavelet scattering of ``phase_scattering2d_torch.py`` using
standard ``nn.Conv2d`` layers initialized with Morlet / Gabor wavelets.
No kymatio or Fourier-domain operations are needed.

Supports two nonlinearities:
- ``'phase_relu'``: 4-phase ReLU rectification (preserves phase information)
- ``'complex_modulus'``: classical complex modulus |z| = sqrt(Re² + Im²)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wavelet_kernels import gabor_2d, morlet_2d


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class AvgPoolConv(nn.Module):
    """Learnable 2x2 stride-2 convolution initialized as average pooling.

    Applies the same 2x2 kernel independently to every channel by reshaping
    the input to single-channel form.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False)
        self.conv.weight.data.fill_(0.25)

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.conv(x.reshape(B * C, 1, H, W))
        return out.reshape(B, C, out.shape[2], out.shape[3])


class WaveletConv(nn.Module):
    """Conv2d banks initialized with spatial-domain wavelet kernels.

    * ``phi_conv``  — low-pass  (Gabor at xi=0), maps C_in → C_in
    * ``psi_real``  — real parts of Morlet wavelets, maps C_in → C_in * L
    * ``psi_imag``  — imaginary parts of Morlet wavelets, maps C_in → C_in * L

    Weights are initialized with a block-diagonal structure so that each
    output group initially depends only on its corresponding input channel
    (equivalent to ``groups=C_in``).  During training the off-diagonal
    entries within each group can learn cross-channel interactions.

    Parameters
    ----------
    mixing_horizon : int or None
        Maximum number of input channels each output filter can see.
        ``None`` means dense (all channels visible).  When set, PyTorch
        grouped convolutions restrict each filter to a local group of at
        most ``mixing_horizon`` channels.  The scattering-transform
        initialization is preserved regardless of this setting.
    """

    def __init__(self, in_channels, L, kernel_size=7, sigma=0.8,
                 xi=None, slant=None, mixing_horizon=None,
                 lowpass_only=False):
        super().__init__()
        self.L = L
        self.in_channels = in_channels
        self.lowpass_only = lowpass_only
        if xi is None:
            xi = 3.0 * np.pi / 4.0
        if slant is None:
            slant = 4.0 / L

        pad = kernel_size // 2
        groups = self._compute_groups(in_channels, mixing_horizon)
        self.groups = groups
        self.actual_horizon = in_channels // groups
        self.requested_horizon = mixing_horizon

        self.phi_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                  padding=pad, bias=False, groups=groups)

        if not lowpass_only:
            self.psi_real = nn.Conv2d(in_channels, in_channels * L, kernel_size,
                                      padding=pad, bias=False, groups=groups)
            self.psi_imag = nn.Conv2d(in_channels, in_channels * L, kernel_size,
                                      padding=pad, bias=False, groups=groups)

        self._init_weights(kernel_size, sigma, xi, slant)

    @staticmethod
    def _compute_groups(c_in, mixing_horizon):
        if mixing_horizon is None or mixing_horizon >= c_in:
            return 1
        groups = max(1, c_in // mixing_horizon)
        while groups > 1 and c_in % groups != 0:
            groups -= 1
        return groups

    def _init_weights(self, kernel_size, sigma, xi, slant):
        C = self.in_channels
        fan_in = C // self.groups

        self.phi_conv.weight.data.zero_()

        phi_kernel = torch.from_numpy(
            gabor_2d(kernel_size, sigma, theta=0.0, xi=0.0, slant=1.0).real)

        for i in range(C):
            self.phi_conv.weight.data[i, i % fan_in] = phi_kernel

        if self.lowpass_only:
            return

        self.psi_real.weight.data.zero_()
        self.psi_imag.weight.data.zero_()

        for l in range(self.L):
            theta = (int(self.L - self.L / 2 - 1) - l) * np.pi / self.L
            psi = morlet_2d(kernel_size, sigma, theta, xi, slant)
            psi_r = torch.from_numpy(psi.real)
            psi_i = torch.from_numpy(psi.imag)
            for i in range(C):
                self.psi_real.weight.data[i * self.L + l, i % fan_in] = psi_r
                self.psi_imag.weight.data[i * self.L + l, i % fan_in] = psi_i


class PhaseReLU(nn.Module):
    """Phase-harmonic nonlinearity: ``relu(cat([x, -x], dim=-1))``."""

    def forward(self, x):
        return F.relu(torch.cat([x, -x], dim=-1))


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class ScatteringBlock(nn.Module):
    """Single-scale spatial-domain scattering block (J=1).

    For an input of shape ``[B, C_in, M, N]`` the output is
    ``[B, C_out, M//2, N//2]`` where ``C_out = C_in * (1 + L)`` for a full
    block and ``C_out = C_in`` for a lowpass-only block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    L : int
        Number of wavelet orientations.
    kernel_size : int
        Side length of the wavelet Conv2d kernels (should be odd).
    learnable : bool
        If *False*, all convolution weights are frozen.
    modulus_type : str
        ``'phase_relu'`` for 4-phase ReLU rectification collapsed via a
        learnable 1x1 grouped conv (initialized as ``sum / sqrt(2)`` to
        approximate the complex modulus), or ``'complex_modulus'`` for the
        classical ``|z| = sqrt(Re² + Im²)``.
    mixing_horizon : int or None
        Caps the number of input channels each wavelet filter can see.
        Forwarded to :class:`WaveletConv`.  ``None`` = dense (default).
    lowpass_only : bool
        If True, only the low-pass (phi) convolution and pooling are created.
        No band-pass parameters (psi) or phase-collapse layers are allocated.
    """

    def __init__(self, in_channels, L=8, kernel_size=7, learnable=True,
                 modulus_type='phase_relu', mixing_horizon=None,
                 lowpass_only=False):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        assert modulus_type in ('phase_relu', 'complex_modulus')
        self.L = L
        self.in_channels = in_channels
        self.modulus_type = modulus_type
        self.lowpass_only = lowpass_only
        self.out_channels = in_channels if lowpass_only else in_channels * (1 + L)

        self.wavelets = WaveletConv(in_channels, L, kernel_size=kernel_size,
                                    mixing_horizon=mixing_horizon,
                                    lowpass_only=lowpass_only)
        self.actual_horizon = self.wavelets.actual_horizon
        self.requested_horizon = self.wavelets.requested_horizon
        self.pool = AvgPoolConv()

        if not lowpass_only and modulus_type == 'phase_relu':
            self.phase_relu = PhaseReLU()
            self.phase_collapse = nn.Conv2d(
                in_channels * 4 * L, in_channels * L,
                kernel_size=1, groups=in_channels * L, bias=False)
            self.phase_collapse.weight.data.fill_(1.0 / math.sqrt(2))

        if not learnable:
            self.freeze()

    # -- forward -----------------------------------------------------------

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor of shape ``(B, C_in, M, N)``

        Returns
        -------
        Tensor of shape ``(B, C_out, M//2, N//2)``
        """
        s0 = self.wavelets.phi_conv(x)                     # [B, C, M, N]
        s0 = self.pool(s0)                                 # [B, C, M/2, N/2]

        if self.lowpass_only:
            return s0

        ur = self.wavelets.psi_real(x)                      # [B, C*L, M, N]
        ui = self.wavelets.psi_imag(x)                      # [B, C*L, M, N]
        ur = self.pool(ur)                                  # [B, C*L, M/2, N/2]
        ui = self.pool(ui)                                  # [B, C*L, M/2, N/2]

        if self.modulus_type == 'phase_relu':
            u_complex = torch.stack([ur, ui], dim=-1)       # [B, C*L, M/2, N/2, 2]
            s1 = self.phase_relu(u_complex)                 # [B, C*L, M/2, N/2, 4]
            s1 = s1.permute(0, 1, 4, 2, 3).flatten(1, 2)   # [B, 4*C*L, M/2, N/2]
            s1 = self.phase_collapse(s1)                    # [B, C*L, M/2, N/2]
        else:
            s1 = torch.sqrt(ur * ur + ui * ui + 1e-8)      # [B, C*L, M/2, N/2]

        return torch.cat([s0, s1], dim=1)                   # [B, C*(1+L), M/2, N/2]

    # -- freeze / unfreeze helpers -----------------------------------------

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
