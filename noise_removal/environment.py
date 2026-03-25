"""
Gymnasium environment for online noise cancellation (closed-loop formulation).

Closed-loop block diagram:

    w1_t ──→ [ Plant P(z) ] ──→ y_t = f(w1_t, [w2_t], t) + n_t
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
the cross-term E(t)·w1·w2, which adaptive filters cannot represent without
an explicit product feature.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .signals import SignalConfig, SignalSimulator


class NoiseCancellationEnv(gym.Env):
    """
    Noise-cancellation environment (single- or multi-source).

    Observation space : Box(n_witnesses·W + W,)
        [ witness1[t-W+1..t], [witness2[t-W+1..t],]  residual[t-W..t-1] ]

    Action space : Box(1,)  in  [-action_clip, +action_clip]

    Reward : y_t² − (y_t − a_t)²  (squared-error improvement)

    Parameters
    ----------
    config           : SignalConfig (set config.multi_source=True for two channels)
    window_size      : W — observation window length in samples
    episode_duration : episode length in seconds
    action_clip      : symmetric bound on the action space
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        window_size: int = 64,
        episode_duration: float = 30.0,
        action_clip: float = 15.0,
    ):
        super().__init__()
        self.config = config or SignalConfig()
        self.window_size = window_size
        self.episode_duration = episode_duration
        self.action_clip = action_clip

        # obs = [w1_win, (w2_win,) residual_win]
        n_witness_channels = 2 if self.config.multi_source else 1
        obs_dim = (n_witness_channels + 1) * window_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-action_clip, high=action_clip, shape=(1,), dtype=np.float32
        )

        # Episode state (populated in reset)
        self._data: Optional[dict] = None
        self._step_idx: int = 0
        self._n_samples: int = 0
        self._action_history: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        sim = SignalSimulator(self.config, seed=seed)
        self._data = sim.generate_episode(
            duration=self.episode_duration,
            signal_amplitude=0.0,
        )
        self._n_samples = len(self._data["time"])
        self._step_idx = self.window_size
        self._action_history = np.zeros(self._n_samples, dtype=np.float64)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        t = self._step_idx
        a = float(np.clip(action[0], -self.action_clip, self.action_clip))

        y_t = float(self._data["main"][t])
        y_clean = y_t - a

        self._action_history[t] = a

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
