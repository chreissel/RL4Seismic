"""
Bilinear dilated-conv feature extractor for seismic noise cancellation.

Designed specifically to make the T2L bilinear coupling representable:

    C_tilt(t) = T(t) · θ_y(t) · w_x(t)

where θ_y(t) is a slow (leaky-integrated) function of witness_y and T(t) is
an OU-drifting gain.  A linear conv stack over (w_x, w_y) can only produce
linear combinations of the two channels — it cannot directly compute the
product w_x · w_y.  We fix this with an explicit bilinear interaction layer
between per-channel temporal features, giving the policy the *minimum*
inductive bias required to represent the T2L term.

Architecture
------------
Input: flat observation = concat(witness_x[t-W+1..t], witness_y[t-W+1..t],
                                 residual[t-W..t-1])   shape (3*W,)
       reshaped to (batch, 3, W)

    1. Split into three channel branches (x / y / residual)
    2. Per-branch dilated causal conv stack (like DilatedCausalConvExtractor)
       → per-branch feature vector φ_x, φ_y, φ_r  (each shape [batch, C])
    3. Bilinear interaction layer:
           z_xy = W_xy · (φ_x ⊗ φ_y)    (low-rank bilinear pooling)
       This is a learned projection of the outer product, giving the network
       a direct path to multiplicative interactions between the two witnesses
       without instantiating a full [C × C] interaction matrix.
    4. Concatenate (φ_x, φ_y, φ_r, z_xy) → MLP → features_dim

Low-rank bilinear pooling
-------------------------
Instead of computing the full outer product φ_x ⊗ φ_y ∈ R^{C×C} and then a
linear projection (which has C² output dims and C²·D params), we use the
Hadamard-product factorisation from Kim et al. 2016:

    z_xy[k] = Σ_i Σ_j U_xy[i, k] · φ_x[i] · V_xy[j, k] · φ_y[j]
            = (U_xyᵀ φ_x) ⊙ (V_xyᵀ φ_y)       (element-wise product)

which has 2·C·R params and R output dims.  R is the bilinear rank.  This is
equivalent to a rank-R factorisation of a full bilinear layer and covers all
the cross-channel multiplications the T2L term can require.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _CausalConv1d(nn.Module):
    """Dilated causal 1D conv with residual + LayerNorm + GELU."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=0, bias=True,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        x_pad = F.pad(x, (self._pad, 0))
        out = self.conv(x_pad)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return F.gelu(out + x)


class _PerChannelEncoder(nn.Module):
    """One dilated-conv stack that processes a single observation channel."""

    def __init__(self, conv_channels: int, n_layers: int, kernel_size: int):
        super().__init__()
        self.input_proj = nn.Conv1d(1, conv_channels, kernel_size=1)
        self.layers = nn.ModuleList([
            _CausalConv1d(conv_channels, kernel_size, dilation=2 ** i)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, time) → (batch, conv_channels) at last timestep
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return h[:, :, -1]  # last timestep


class BilinearDilatedExtractor(BaseFeaturesExtractor):
    """
    Bilinear dilated-conv feature extractor.

    Parameters
    ----------
    observation_space : gymnasium Box of shape (3 * window_size,)
    window_size       : temporal window W (default 240 = 60 s @ 4 Hz)
    conv_channels     : channels per per-branch conv stack (default 48)
    n_layers          : number of dilated-conv layers per branch (default 8)
    kernel_size       : conv kernel width (default 3)
    bilinear_rank     : rank R of the low-rank bilinear pooling (default 32)
    features_dim      : output dim of the MLP head (default 256)
    """

    def __init__(
        self,
        observation_space,
        window_size: int = 240,
        conv_channels: int = 48,
        n_layers: int = 8,
        kernel_size: int = 3,
        bilinear_rank: int = 32,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        obs_dim = observation_space.shape[0]
        assert obs_dim == 3 * window_size, (
            f"obs_dim={obs_dim} must equal 3*window_size={3 * window_size}"
        )
        self.window_size = window_size
        self.conv_channels = conv_channels
        self.bilinear_rank = bilinear_rank

        # Per-branch encoders (one each for w_x, w_y, residual)
        self.encoder_x = _PerChannelEncoder(conv_channels, n_layers, kernel_size)
        self.encoder_y = _PerChannelEncoder(conv_channels, n_layers, kernel_size)
        self.encoder_r = _PerChannelEncoder(conv_channels, n_layers, kernel_size)

        # Low-rank bilinear pooling between x and y:
        #   z_xy = (U φ_x) ⊙ (V φ_y)   ∈ R^{bilinear_rank}
        self.bilinear_u = nn.Linear(conv_channels, bilinear_rank, bias=False)
        self.bilinear_v = nn.Linear(conv_channels, bilinear_rank, bias=False)

        # Inject instantaneous raw samples so the agent has exact current-time
        # access to (w_x[t], w_y[t], residual[t-1]) without any conv smoothing.
        # This is the current-time "oracle" signal path used by the policy to
        # compute the final action — crucial for precise pointwise subtraction.
        self._n_point_features = 3

        fused_dim = (
            3 * conv_channels       # per-branch features
            + bilinear_rank         # bilinear pooled term
            + self._n_point_features  # raw instantaneous samples
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(fused_dim, 2 * features_dim),
            nn.GELU(),
            nn.Linear(2 * features_dim, features_dim),
            nn.GELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        W = self.window_size

        # Reshape flat obs → (batch, 3, W)
        x = observations.view(batch, 3, W)

        w_x = x[:, 0:1, :]   # (batch, 1, W)
        w_y = x[:, 1:2, :]
        res = x[:, 2:3, :]

        phi_x = self.encoder_x(w_x)  # (batch, C)
        phi_y = self.encoder_y(w_y)
        phi_r = self.encoder_r(res)

        # Low-rank bilinear pooling of (phi_x, phi_y)
        z_xy = self.bilinear_u(phi_x) * self.bilinear_v(phi_y)  # (batch, R)

        # Instantaneous raw samples at the current timestep
        # w_x[:, 0, -1] is w_x[t]; w_y[:, 0, -1] is w_y[t]; res[:, 0, -1] is e[t-1]
        point_features = torch.stack([
            w_x[:, 0, -1], w_y[:, 0, -1], res[:, 0, -1]
        ], dim=-1)  # (batch, 3)

        fused = torch.cat([phi_x, phi_y, phi_r, z_xy, point_features], dim=-1)
        return self.output_mlp(fused)
