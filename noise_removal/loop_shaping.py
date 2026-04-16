"""
Frequency-domain loop-shaping reward wrapper for Deep Loop Shaping (DLS).

Implements the core idea from arXiv:2509.14016: reward the RL controller
based on the closed-loop sensitivity function |S(f)| estimated from the
power spectral densities of the uncontrolled signal and the residual,
rather than instantaneous time-domain squared error.

Two modes are available:

  **hybrid** (default) — blends a dense per-step bandpass reward with a
  spectral PSD bonus.  Good for initial exploration and stable learning.

  **segment** — pure frequency-domain reward, closer to the paper.
  Accumulates non-overlapping segments of ``psd_window`` steps, computes
  the PSD-based sensitivity reward once per segment, and repeats that
  reward for every step in the next segment.  No time-domain component.
  Better for fine-tuning once the agent already cancels noise.

Reward (spectral component)
---------------------------
    R_spectral = mean( -log₁₀(|S(f)|²)  for f ∈ [f_low, f_high] )
               - α · mean( max(0, log₁₀(|S(f)|²))  for f ∉ [f_low, f_high] )
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from scipy.signal import butter, sosfilt, sosfilt_zi


class LoopShapingWrapper(gym.Wrapper):
    """Replace the env reward with a loop-shaping reward.

    Parameters
    ----------
    env : gym.Env
        Base environment (must put ``main_raw`` and ``main_clean`` in info).
    psd_window : int
        Number of samples in the PSD estimation window.  Default 256.
    f_low, f_high : float
        Control-band edges in Hz.  Default [0.05, 0.5] Hz.
    amplification_penalty : float
        Weight α on the out-of-band amplification penalty.  Default 2.0.
    spectral_weight : float
        Mixing weight λ for the spectral bonus in hybrid mode.  Default 0.5.
    mode : str
        ``"hybrid"`` (default) — dense bandpass + spectral bonus.
        ``"segment"`` — pure spectral, segment-level PSD reward.
    fs : float
        Sampling rate in Hz.  Default 4.0.
    """

    def __init__(
        self,
        env: gym.Env,
        psd_window: int = 256,
        f_low: float = 0.05,
        f_high: float = 0.5,
        amplification_penalty: float = 2.0,
        spectral_weight: float = 0.5,
        mode: str = "hybrid",
        fs: float = 4.0,
    ):
        super().__init__(env)
        assert mode in ("hybrid", "segment"), f"Unknown mode: {mode!r}"
        self.psd_window = psd_window
        self.f_low = f_low
        self.f_high = f_high
        self.amplification_penalty = amplification_penalty
        self.spectral_weight = spectral_weight
        self.mode = mode
        self.fs = fs

        # --- Dense bandpass reward (per-step, causal) — used in hybrid mode ---
        nyq = fs / 2.0
        low = max(f_low, 0.01)
        high = min(f_high, nyq * 0.99)
        self._bp_sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")

        # --- Spectral (PSD) reward ---
        self._freqs = np.fft.rfftfreq(psd_window, d=1.0 / fs)
        self._in_band = (self._freqs >= f_low) & (self._freqs <= f_high)
        self._out_band = ~self._in_band & (self._freqs > 0)

        self._window = np.hanning(psd_window)
        self._win_power = float(np.sum(self._window ** 2))

        # Buffers (circular for hybrid, linear segment for segment mode)
        self._y_buf = np.zeros(psd_window, dtype=np.float64)
        self._e_buf = np.zeros(psd_window, dtype=np.float64)
        self._buf_pos: int = 0
        self._buf_full: bool = False

        # Segment mode state
        self._seg_pos: int = 0
        self._cached_spectral_reward: float = 0.0
        self._cached_sens_info: dict = {}

        # Bandpass filter state
        self._bp_zi_y: np.ndarray | None = None
        self._bp_zi_e: np.ndarray | None = None

    def reset(self, **kwargs):
        self._y_buf[:] = 0.0
        self._e_buf[:] = 0.0
        self._buf_pos = 0
        self._buf_full = False
        self._seg_pos = 0
        self._cached_spectral_reward = 0.0
        self._cached_sens_info = {}

        zi_template = sosfilt_zi(self._bp_sos)
        self._bp_zi_y = zi_template * 0.0
        self._bp_zi_e = zi_template * 0.0

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)

        y_t = info["main_raw"]
        e_t = info["main_clean"]

        if self.mode == "segment":
            reward = self._step_segment(y_t, e_t, info)
        else:
            reward = self._step_hybrid(y_t, e_t, info)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Hybrid mode (dense bandpass + spectral sliding-window bonus)
    # ------------------------------------------------------------------

    def _step_hybrid(self, y_t: float, e_t: float, info: dict) -> float:
        # Dense reward
        y_arr = np.array([y_t])
        e_arr = np.array([e_t])
        y_bp, self._bp_zi_y = sosfilt(self._bp_sos, y_arr, zi=self._bp_zi_y)
        e_bp, self._bp_zi_e = sosfilt(self._bp_sos, e_arr, zi=self._bp_zi_e)
        dense_reward = float(y_bp[0] ** 2 - e_bp[0] ** 2)

        # Spectral reward (sliding window)
        idx = self._buf_pos % self.psd_window
        self._y_buf[idx] = y_t
        self._e_buf[idx] = e_t
        self._buf_pos += 1
        if self._buf_pos >= self.psd_window:
            self._buf_full = True

        if self._buf_full:
            spectral_reward, sens_info = self._compute_spectral()
            info.update(sens_info)
        else:
            spectral_reward = 0.0

        reward = dense_reward + self.spectral_weight * spectral_reward
        info["dense_reward"] = dense_reward
        info["spectral_reward"] = spectral_reward
        return reward

    # ------------------------------------------------------------------
    # Segment mode (pure spectral, non-overlapping segments)
    # ------------------------------------------------------------------

    def _step_segment(self, y_t: float, e_t: float, info: dict) -> float:
        self._y_buf[self._seg_pos] = y_t
        self._e_buf[self._seg_pos] = e_t
        self._seg_pos += 1

        if self._seg_pos >= self.psd_window:
            # Segment complete — compute PSD reward
            spectral_reward, sens_info = self._compute_spectral_direct()
            self._cached_spectral_reward = spectral_reward
            self._cached_sens_info = sens_info
            self._seg_pos = 0

        info.update(self._cached_sens_info)
        info["spectral_reward"] = self._cached_spectral_reward
        return self._cached_spectral_reward

    # ------------------------------------------------------------------
    # PSD computation
    # ------------------------------------------------------------------

    def _compute_spectral(self):
        """Spectral reward from the circular buffer (hybrid mode)."""
        W = self.psd_window
        start = self._buf_pos % W
        y = np.empty(W, dtype=np.float64)
        e = np.empty(W, dtype=np.float64)
        tail = W - start
        y[:tail] = self._y_buf[start:]
        y[tail:] = self._y_buf[:start]
        e[:tail] = self._e_buf[start:]
        e[tail:] = self._e_buf[:start]
        return self._psd_reward(y, e)

    def _compute_spectral_direct(self):
        """Spectral reward from the linear segment buffer."""
        return self._psd_reward(self._y_buf.copy(), self._e_buf.copy())

    def _psd_reward(self, y: np.ndarray, e: np.ndarray):
        y_w = y * self._window
        e_w = e * self._window
        psd_y = np.abs(np.fft.rfft(y_w)) ** 2 / self._win_power
        psd_e = np.abs(np.fft.rfft(e_w)) ** 2 / self._win_power

        eps = 1e-30
        sensitivity_sq = np.clip(
            (psd_e + eps) / (psd_y + eps), 1e-6, 1e6
        )
        log_sens = np.log10(sensitivity_sq)

        if np.any(self._in_band):
            in_band_suppression = float(-np.mean(log_sens[self._in_band]))
        else:
            in_band_suppression = 0.0

        if np.any(self._out_band):
            amplification = float(
                np.mean(np.maximum(0.0, log_sens[self._out_band]))
            )
        else:
            amplification = 0.0

        spectral = in_band_suppression - self.amplification_penalty * amplification

        sens_info = {
            "loop_shaping_reward": spectral,
            "sensitivity_in_band_db": -10.0 * in_band_suppression,
            "sensitivity_out_band_db": 10.0 * amplification,
        }
        return spectral, sens_info
