"""
Gymnasium environment for online seismic noise cancellation (closed-loop formulation).

Closed-loop block diagram:

    w1_t ──→ [ Plant P(z) ] ──→ y_t = h(t)⊛w1_t + [T2L] + n_t
    w2_t ─┘                            │
      │                         ─── (+) ──→ e_t = y_t − a_t  (error / residual)
      │                        │
      └──→ [ Controller C ] ──→ a_t
            (RL policy π)

Single-source observation (default, 2·W floats):
    [ witness1[t-W+1..t],  residual[t-W..t-1] ]

Multi-source observation (config.multi_source=True, 3·W floats):
    [ witness1[t-W+1..t],  witness2[t-W+1..t],  residual[t-W..t-1] ]

The second witness window gives the agent the information needed to learn
the tilt-to-length bilinear cross-coupling T(t)·θ(t)·w1(t), which linear
adaptive filters (LMS/NLMS) cannot cancel.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.signal import butter, sosfilt_zi, sosfilt

from .signals import SeismicConfig, SeismicSignalSimulator


class NoiseCancellationEnv(gym.Env):
    """
    Noise-cancellation environment (single- or multi-source).

    Observation space : Box(n_witnesses·W + W,)
        [ witness1[t-W+1..t], [witness2[t-W+1..t],]  residual[t-W..t-1] ]

    Action space : Box(1,)  in  [-action_clip, +action_clip]

    Reward : y_t² − (y_t − a_t)²  (squared-error improvement, broadband)
           or, with freq_reward=True:
             y_bp_t² − e_bp_t²  (improvement in band-limited power only,
             where bp = bandpass-filtered to [freq_band_low, freq_band_high] Hz)

    Parameters
    ----------
    config           : SeismicConfig (set config.multi_source=True for two channels)
    window_size      : W — observation window length in samples
    episode_duration : episode length in seconds
    action_clip      : symmetric bound on the action space (default 25.0)
    freq_reward      : if True, compute reward on band-limited residual only
    freq_band_low    : lower edge of reward band in Hz (default 0.05 Hz)
    freq_band_high   : upper edge of reward band in Hz (default 1.5 Hz)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[SeismicConfig] = None,
        window_size: int = 240,
        episode_duration: float = 300.0,
        action_clip: float = 25.0,   # raised from 15: coupling peaks reach ~25–40 with T2L
        freq_reward: bool = False,
        freq_band_low: float = 0.05,
        freq_band_high: float = 1.5,
    ):
        super().__init__()
        self.config = config if config is not None else SeismicConfig()
        self.window_size = window_size
        self.episode_duration = episode_duration
        self.action_clip = action_clip
        self.freq_reward = freq_reward

        # obs = [w1_win, (w2_win,) residual_win]
        n_witness_channels = 2 if self.config.multi_source else 1
        obs_dim = (n_witness_channels + 1) * window_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-action_clip, high=action_clip, shape=(1,), dtype=np.float32
        )

        # Design bandpass filter for frequency-domain reward (DLS-style)
        if freq_reward:
            fs = self.config.fs
            nyq = fs / 2.0
            low = max(freq_band_low, 0.01)
            high = min(freq_band_high, nyq * 0.99)
            self._bp_sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
        else:
            self._bp_sos = None

        # Episode state (populated in reset)
        self._data: Optional[dict] = None
        self._step_idx: int = 0
        self._n_samples: int = 0
        self._action_history: Optional[np.ndarray] = None
        self._bp_zi_y: Optional[np.ndarray] = None  # bandpass state for y_t
        self._bp_zi_e: Optional[np.ndarray] = None  # bandpass state for e_t

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        sim = SeismicSignalSimulator(self.config, seed=seed)
        self._data = sim.generate_episode(
            duration=self.episode_duration,
            signal_amplitude=0.0,
        )
        self._n_samples = len(self._data["time"])
        self._step_idx = self.window_size
        self._action_history = np.zeros(self._n_samples, dtype=np.float64)

        # Reset causal bandpass filter states
        if self._bp_sos is not None:
            zi_template = sosfilt_zi(self._bp_sos)  # shape (n_sections, 2)
            self._bp_zi_y = zi_template * 0.0
            self._bp_zi_e = zi_template * 0.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        t = self._step_idx
        a = float(np.clip(action[0], -self.action_clip, self.action_clip))

        y_t = float(self._data["main"][t])
        y_clean = y_t - a

        self._action_history[t] = a

        if self._bp_sos is not None:
            # Frequency-domain reward: improve band-limited power (DLS-style)
            y_arr = np.array([y_t])
            e_arr = np.array([y_clean])
            y_bp_arr, self._bp_zi_y = sosfilt(self._bp_sos, y_arr, zi=self._bp_zi_y)
            e_bp_arr, self._bp_zi_e = sosfilt(self._bp_sos, e_arr, zi=self._bp_zi_e)
            reward = float(y_bp_arr[0] ** 2 - e_bp_arr[0] ** 2)
        else:
            reward = float(y_t**2 - y_clean**2)

        self._step_idx += 1
        terminated = self._step_idx >= self._n_samples

        obs = self._get_obs() if not terminated else self._zero_obs()
        info = {
            "t": self._data["time"][t],
            "main_raw": y_t,
            "main_clean": y_clean,
            "coupling_true": float(self._data["coupling"][t]),
            "action": a,
        }
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        t = self._step_idx
        W = self.window_size

        witness_win  = self._data["witness"][t - W + 1 : t + 1]
        main_win     = self._data["main"][t - W : t]
        action_win   = self._action_history[t - W : t]
        residual_win = (main_win - action_win).astype(np.float32)

        parts = [witness_win]
        if self.config.multi_source:
            parts.append(self._data["witness2"][t - W + 1 : t + 1])
        parts.append(residual_win)

        return np.concatenate(parts).astype(np.float32)

    def _zero_obs(self) -> np.ndarray:
        n_witness_channels = 2 if self.config.multi_source else 1
        return np.zeros((n_witness_channels + 1) * self.window_size, dtype=np.float32)
