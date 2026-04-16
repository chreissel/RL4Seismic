"""
Frequency-domain loop-shaping reward wrapper for Deep Loop Shaping (DLS).

Implements the core idea from arXiv:2509.14016: reward the RL controller
based on the closed-loop sensitivity function |S(f)| estimated from the
power spectral densities of the uncontrolled signal and the residual,
rather than instantaneous time-domain squared error.

The wrapper accumulates a rolling buffer of uncontrolled signal y_t and
residual e_t = y_t - a_t, computes their power spectral densities via a
Hann-windowed periodogram, and emits a reward that drives the policy to
minimise the sensitivity function inside the control band while penalising
amplification outside it (the waterbed effect from Bode's integral theorem).

Reward
------
The empirical squared sensitivity at frequency f is estimated as:

    |S(f)|^2  ≈  PSD_e(f) / PSD_y(f)

|S(f)|^2 < 1  ⟹  suppression (good);   |S(f)|^2 > 1  ⟹  amplification (bad).

The per-step reward (once the buffer is full) is:

    R = mean( -log₁₀(|S(f)|²)  for f ∈ [f_low, f_high] )               [in-band]
      - α · mean( max(0, log₁₀(|S(f)|²))  for f ∉ [f_low, f_high] )    [out-of-band]

The first term rewards in-band suppression (positive when |S| < 1).
The second penalises out-of-band amplification with weight α.
Values are in log₁₀ units (decades); multiply by 10 for dB.

During the warmup period (first ``psd_window`` steps while the buffer fills),
the wrapper passes through the base environment's reward unchanged.

Usage
-----
    from noise_removal.loop_shaping import LoopShapingWrapper

    env = NoiseCancellationEnv(config=cfg, ...)
    env = LoopShapingWrapper(env, psd_window=256, f_low=0.05, f_high=0.5)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class LoopShapingWrapper(gym.Wrapper):
    """Replace the env reward with a frequency-domain loop-shaping reward.

    Estimates the closed-loop sensitivity function from a sliding PSD
    window and rewards in-band suppression while penalising out-of-band
    amplification.

    Parameters
    ----------
    env : gym.Env
        Base environment (must put ``main_raw`` and ``main_clean`` in info).
    psd_window : int
        Number of samples in the PSD estimation window.  Larger windows give
        finer frequency resolution but longer warmup.  Default 256 (= 64 s
        @ 4 Hz, frequency resolution 0.015 Hz, ~29 bins in [0.05, 0.5] Hz).
    f_low, f_high : float
        Control-band edges in Hz.  The reward drives |S(f)| down inside
        this band.  Default [0.05, 0.5] Hz (microseismic band).
    amplification_penalty : float
        Weight α on the out-of-band amplification penalty term.
        Higher values make the agent more conservative about injecting
        noise outside the control band.  Default 2.0.
    fs : float
        Sampling rate in Hz.  Must match the environment.  Default 4.0.
    """

    def __init__(
        self,
        env: gym.Env,
        psd_window: int = 256,
        f_low: float = 0.05,
        f_high: float = 0.5,
        amplification_penalty: float = 2.0,
        fs: float = 4.0,
    ):
        super().__init__(env)
        self.psd_window = psd_window
        self.f_low = f_low
        self.f_high = f_high
        self.amplification_penalty = amplification_penalty
        self.fs = fs

        # Precompute frequency bins and band masks
        self._freqs = np.fft.rfftfreq(psd_window, d=1.0 / fs)
        self._in_band = (self._freqs >= f_low) & (self._freqs <= f_high)
        self._out_band = ~self._in_band & (self._freqs > 0)  # exclude DC

        # Hann window for spectral leakage reduction
        self._window = np.hanning(psd_window)
        # Window power (for PSD normalisation: PSD = |FFT(x·w)|² / S₂)
        self._win_power = float(np.sum(self._window ** 2))

        # Circular buffers for y_t (uncontrolled) and e_t (residual)
        self._y_buf = np.zeros(psd_window, dtype=np.float64)
        self._e_buf = np.zeros(psd_window, dtype=np.float64)
        self._buf_pos: int = 0
        self._buf_full: bool = False

    def reset(self, **kwargs):
        self._y_buf[:] = 0.0
        self._e_buf[:] = 0.0
        self._buf_pos = 0
        self._buf_full = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Append to circular buffer
        idx = self._buf_pos % self.psd_window
        self._y_buf[idx] = info["main_raw"]
        self._e_buf[idx] = info["main_clean"]
        self._buf_pos += 1
        if self._buf_pos >= self.psd_window:
            self._buf_full = True

        if self._buf_full:
            reward, sens_info = self._spectral_reward()
            info.update(sens_info)
        else:
            reward = base_reward

        return obs, reward, terminated, truncated, info

    def _spectral_reward(self):
        """Compute the loop-shaping reward from the current buffer."""
        W = self.psd_window
        start = self._buf_pos % W

        # Unroll circular buffer into time-ordered arrays
        y = np.empty(W, dtype=np.float64)
        e = np.empty(W, dtype=np.float64)
        tail = W - start
        y[:tail] = self._y_buf[start:]
        y[tail:] = self._y_buf[:start]
        e[:tail] = self._e_buf[start:]
        e[tail:] = self._e_buf[:start]

        # Windowed periodogram
        y_w = y * self._window
        e_w = e * self._window
        psd_y = np.abs(np.fft.rfft(y_w)) ** 2 / self._win_power
        psd_e = np.abs(np.fft.rfft(e_w)) ** 2 / self._win_power

        # Empirical squared sensitivity, clipped to [-60, +60] dB range
        eps = 1e-30
        sensitivity_sq = np.clip(
            (psd_e + eps) / (psd_y + eps), 1e-6, 1e6
        )
        log_sens = np.log10(sensitivity_sq)  # in decades; ×10 for dB

        # In-band: reward suppression (positive when agent is cancelling)
        if np.any(self._in_band):
            in_band_suppression = float(-np.mean(log_sens[self._in_band]))
        else:
            in_band_suppression = 0.0

        # Out-of-band: penalise amplification only (|S| > 1)
        if np.any(self._out_band):
            amplification = float(
                np.mean(np.maximum(0.0, log_sens[self._out_band]))
            )
        else:
            amplification = 0.0

        reward = in_band_suppression - self.amplification_penalty * amplification

        sens_info = {
            "loop_shaping_reward": reward,
            "sensitivity_in_band_db": -10.0 * in_band_suppression,
            "sensitivity_out_band_db": 10.0 * amplification,
        }
        return reward, sens_info
