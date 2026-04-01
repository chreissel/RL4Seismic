"""
Dilated causal convolution feature extractor for noise cancellation.

Inspired by the DeepMind Deep Loop Shaping work (arXiv:2509.14016), which
replaces the LSTM hidden state with a dilated causal convolution stack to give
the policy a large temporal receptive field without recurrent state.
Dilated causal convolutions (WaveNet-style) provide:

  - Large receptive field with few parameters: RF = 1 + (k−1)·Σ 2^i
  - No vanishing gradients over long sequences (unlike LSTM)
  - All time steps computed in parallel during training (fast)
  - Strictly causal: no future information leaks into the representation

Architecture (default: 8 layers, kernel=3, dilation doubles each layer):

    Layer 0: dilation=1,   RF contribution =   2
    Layer 1: dilation=2,   RF contribution =   4
    Layer 2: dilation=4,   RF contribution =   8
    Layer 3: dilation=8,   RF contribution =  16
    Layer 4: dilation=16,  RF contribution =  32
    Layer 5: dilation=32,  RF contribution =  64
    Layer 6: dilation=64,  RF contribution = 128
    Layer 7: dilation=128, RF contribution = 256
                                             ───
    Total receptive field = 1 + 2+4+8+16+32+64+128+256 = 511 samples (= 127 s @ 4 Hz)

This covers the full 240-sample (60 s) seismic observation window with margin.
6 layers (RF=127) is insufficient for the seismic model — use ≥7 layers.

The extractor reshapes the flat observation vector into (n_channels, window_size)
before applying the convolutions, then takes the last-timestep features.
Using the last timestep (rather than global avg pool) preserves exact current-time
values (w1[t], tilt_proxy[t]) needed for T2L bilinear cancellation, while the
dilated conv stack still provides full 511-sample context via its receptive field.

Usage
-----
    from noise_removal.policy import DilatedCausalConvExtractor
    from stable_baselines3 import PPO

    policy_kwargs = dict(
        features_extractor_class=DilatedCausalConvExtractor,
        features_extractor_kwargs=dict(window_size=240, conv_channels=64, n_layers=8),
        net_arch=[256, 256],
    )
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CausalConv1d(nn.Module):
    """Single dilated causal convolution layer with residual connection."""

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
        # LayerNorm over channel dim, applied after transposing
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return F.gelu(out + x)  # residual + activation


class DilatedCausalConvExtractor(BaseFeaturesExtractor):
    """
    WaveNet-style dilated causal convolution feature extractor.

    Replaces the LSTM of RecurrentPPO with a stack of dilated causal
    convolutions that cover the full observation window.

    Parameters
    ----------
    observation_space : gymnasium.spaces.Box
        Flat observation vector of shape (n_obs_channels * window_size,).
    window_size       : number of time steps in the observation window.
    conv_channels     : number of convolution filters per layer.
    n_layers          : number of dilated conv layers (dilation doubles each layer).
    kernel_size       : convolution kernel width.
    features_dim      : output dimensionality after global pooling + linear projection.
    """

    def __init__(
        self,
        observation_space,
        window_size: int = 64,
        conv_channels: int = 64,
        n_layers: int = 6,
        kernel_size: int = 3,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        obs_dim = observation_space.shape[0]
        assert obs_dim % window_size == 0, (
            f"obs_dim={obs_dim} must be divisible by window_size={window_size}"
        )
        self.n_obs_channels = obs_dim // window_size
        self.window_size = window_size

        # Input projection: n_obs_channels → conv_channels
        self.input_proj = nn.Conv1d(self.n_obs_channels, conv_channels, kernel_size=1)

        # Stack of dilated causal conv layers
        self.layers = nn.ModuleList([
            CausalConv1d(conv_channels, kernel_size, dilation=2 ** i)
            for i in range(n_layers)
        ])

        # Project last-timestep features to features_dim.
        # We take x[:, :, -1] (the causally-computed features at the current
        # timestep) rather than global-average-pooling over time.  Global avg
        # pool dilutes the current-sample values (w1[t], tilt_proxy[t]) with
        # all 240 past steps, destroying the precise current-time signal the
        # agent needs to compute T_gain(t) × tilt_proxy(t) × w1(t) for T2L
        # cancellation.  The last position already encodes full context via
        # the 511-sample dilated-conv receptive field, so no context is lost.
        self.output_proj = nn.Linear(conv_channels, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, obs_dim)
        batch = observations.shape[0]
        # Reshape to (batch, n_obs_channels, window_size)
        x = observations.view(batch, self.n_obs_channels, self.window_size)

        x = self.input_proj(x)          # (batch, conv_channels, window_size)
        for layer in self.layers:
            x = layer(x)               # (batch, conv_channels, window_size)

        # Take last timestep (current time, causally conditioned on full RF)
        x = x[:, :, -1]              # (batch, conv_channels)
        return F.gelu(self.output_proj(x))  # (batch, features_dim)
