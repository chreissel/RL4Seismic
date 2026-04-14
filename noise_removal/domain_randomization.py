"""
Domain-randomised noise cancellation environment.

Wraps NoiseCancellationEnv and resamples SeismicConfig fields on every
reset from a user-supplied distribution.  The agent therefore has to learn
a policy that is *simultaneously* good across a family of seismic
environments — different OU drift magnitudes, different T2L gains,
different regime-change frequencies, different sensor-noise colours.

This is the training protocol that forces the policy to become an
*adaptive* controller (responding to observed residuals within an episode)
rather than memorising a single fixed coupling path.  It is also the
protocol most consistent with the real-world motivation: LIGO data exhibits
slow non-stationarity across many timescales, so a useful noise-cancelling
controller has to handle a range of operating regimes.

The wrapper samples a new SeismicConfig at each `reset()` call, rebuilds
the underlying NoiseCancellationEnv with that config, and delegates all
other methods to it.

Randomised fields
-----------------
  - drift_scale          ∈ [cfg.drift_scale_low, cfg.drift_scale_high]
    Multiplies gain_drift_sigma, freq_drift_sigma, t2l_gain_drift_sigma.
  - t2l_gain             ∈ [cfg.t2l_gain_low, cfg.t2l_gain_high]
  - thermal_timescale    ∈ [cfg.thermal_timescale_low, cfg.thermal_timescale_high]
    Applied to both linear and T2L OU timescales.
  - sensor_noise_exponent uniformly from cfg.sensor_noise_colors.
  - regime_changes       Bernoulli(cfg.regime_change_prob).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym

from .environment import NoiseCancellationEnv
from .signals import SeismicConfig


@dataclass
class DomainRandomizationConfig:
    """Ranges for the per-episode randomisation."""

    # Default drift sigmas from SeismicConfig:
    #   gain_drift_sigma = 0.1, freq_drift_sigma = 0.01, t2l_gain_drift_sigma = 2.7
    drift_scale_low: float = 0.5
    drift_scale_high: float = 4.0

    t2l_gain_low: float = 25.0
    t2l_gain_high: float = 60.0

    thermal_timescale_low: float = 120.0
    thermal_timescale_high: float = 1200.0

    sensor_noise_colors: Tuple[float, ...] = (0.0, 1.0, 2.0)
    sensor_noise_band: Tuple[float, float] = (0.05, 0.5)

    regime_change_prob: float = 0.3
    n_regimes: int = 4
    mean_hold_time: float = 120.0


class DomainRandomizedNoiseCancellationEnv(gym.Env):
    """
    Wraps NoiseCancellationEnv with per-episode domain randomisation.

    The underlying config is rebuilt on every reset from a fresh sample,
    but observation / action shapes and reward semantics are unchanged.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_config: Optional[SeismicConfig] = None,
        dr_config: Optional[DomainRandomizationConfig] = None,
        window_size: int = 240,
        episode_duration: float = 300.0,
        action_clip: float = 40.0,
    ):
        super().__init__()

        self.base_config = base_config if base_config is not None else SeismicConfig()
        self.dr_config = dr_config if dr_config is not None else DomainRandomizationConfig()
        self.window_size = window_size
        self.episode_duration = episode_duration
        self.action_clip = action_clip

        # Build an initial inner env with the base config so SB3 can inspect
        # observation / action spaces during construction.
        self._inner = NoiseCancellationEnv(
            config=self.base_config,
            window_size=window_size,
            episode_duration=episode_duration,
            action_clip=action_clip,
        )
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Randomisation
    # ------------------------------------------------------------------

    def _sample_config(self) -> SeismicConfig:
        d = self.dr_config
        base = self.base_config

        drift_scale = self._rng.uniform(d.drift_scale_low, d.drift_scale_high)
        t2l_gain = self._rng.uniform(d.t2l_gain_low, d.t2l_gain_high)
        tau = self._rng.uniform(d.thermal_timescale_low, d.thermal_timescale_high)
        colour_exp = float(self._rng.choice(d.sensor_noise_colors))
        regime = bool(self._rng.random() < d.regime_change_prob)

        cfg = SeismicConfig(
            fs=base.fs,
            filter_length=base.filter_length,
            coupling_gain=base.coupling_gain,
            resonance_freq=base.resonance_freq,
            resonance_q=base.resonance_q,
            drift=True,
            thermal_timescale=tau,
            gain_drift_sigma=base.gain_drift_sigma * drift_scale,
            freq_drift_sigma=base.freq_drift_sigma * drift_scale,
            sensor_noise_sigma=base.sensor_noise_sigma,
            sensor_noise_exponent=colour_exp,
            sensor_noise_band=d.sensor_noise_band,
            regime_changes=regime,
            n_regimes=d.n_regimes,
            mean_hold_time=d.mean_hold_time,
            tilt_coupling=True,
            tilt_leak_timescale=base.tilt_leak_timescale,
            t2l_gain=t2l_gain,
            t2l_gain_drift_sigma=base.t2l_gain_drift_sigma * drift_scale,
            t2l_thermal_timescale=tau,
        )
        return cfg

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cfg = self._sample_config()
        self._inner = NoiseCancellationEnv(
            config=cfg,
            window_size=self.window_size,
            episode_duration=self.episode_duration,
            action_clip=self.action_clip,
        )
        # Pass through a fresh random seed so the episode data is also random.
        episode_seed = int(self._rng.integers(0, 2**31 - 1))
        obs, info = self._inner.reset(seed=episode_seed)
        info["dr_config"] = {
            "t2l_gain": cfg.t2l_gain,
            "drift_sigma_T": cfg.t2l_gain_drift_sigma,
            "thermal_timescale": cfg.t2l_thermal_timescale,
            "sensor_noise_exponent": cfg.sensor_noise_exponent,
            "regime_changes": cfg.regime_changes,
        }
        return obs, info

    def step(self, action):
        return self._inner.step(action)
