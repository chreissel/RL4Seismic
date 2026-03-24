"""
Gymnasium environment for online noise cancellation.

At every step t the agent:
  1. Observes  : [witness[t-W+1 .. t],  main[t-W .. t-1]]  (2W floats)
                  ↑ includes *current* witness sample but NOT current main
  2. Acts      : outputs a scalar  a_t  — the estimated coupling to subtract
  3. Receives  :  reward_t = y_t² − (y_t − a_t)²
                           = 2·y_t·a_t − a_t²

Why this reward?
  E[reward] is maximised when a_t = f(w_t, t) (exact coupling subtraction).
  Because  y_t = n_t + f(w_t, t)  (no test signal during training):
    E[reward | perfect subtraction] = f² > 0
  The agent cannot do better than removing the coupling; it cannot subtract
  the unpredictable sensor noise n_t.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .signals import SignalConfig, SignalSimulator


class NoiseCancellationEnv(gym.Env):
    """
    Two-channel noise-cancellation environment.

    Observation space : Box(2·window_size,)
        [ witness[t-W+1..t],  main[t-W..t-1] ]

    Action space : Box(1,)  in  [-action_clip, +action_clip]
        Scalar coupling estimate to subtract from the current main-channel sample.

    Reward : improvement in instantaneous squared amplitude after subtraction.

    Parameters
    ----------
    config           : SignalConfig instance
    window_size      : W — number of past samples in the observation
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

        obs_dim = 2 * window_size
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

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        sim = SignalSimulator(self.config, seed=seed)
        self._data = sim.generate_episode(
            duration=self.episode_duration,
            signal_amplitude=0.0,  # no test signal during training
        )
        self._n_samples = len(self._data["time"])
        # Start once the observation window is filled
        self._step_idx = self.window_size

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        t = self._step_idx
        a = float(np.clip(action[0], -self.action_clip, self.action_clip))

        y_t = float(self._data["main"][t])
        y_clean = y_t - a

        # Reward: squared-error improvement  y_t² − y_clean²
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
        witness_win = self._data["witness"][t - W + 1 : t + 1]   # includes t
        main_win    = self._data["main"][t - W : t]               # up to t-1
        return np.concatenate([witness_win, main_win]).astype(np.float32)

    def _zero_obs(self) -> np.ndarray:
        return np.zeros(2 * self.window_size, dtype=np.float32)
