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

The per-step reward blends a dense instantaneous term (per-sample
band-limited power improvement, for temporal credit assignment) with
a spectral shaping bonus (PSD-based sensitivity, for frequency-domain
control):

    R_t = R_bandpass(t)  +  λ · R_spectral(t)

The spectral component is:

    R_spectral = mean( -log₁₀(|S(f)|²)  for f ∈ [f_low, f_high] )
               - α · mean( max(0, log₁₀(|S(f)|²))  for f ∉ [f_low, f_high] )

The first term rewards in-band suppression (positive when |S| < 1).
The second penalises out-of-band amplification with weight α.

The dense bandpass term provides per-step gradient so the RL agent can
assign credit to individual actions.  The spectral bonus nudges the
overall closed-loop shape toward the target sensitivity profile.

During the warmup period (first ``psd_window`` steps while the buffer
fills), the spectral term is zero and only the bandpass reward is used.

Usage
-----
    from noise_removal.loop_shaping import LoopShapingWrapper

    # Base env should have freq_reward=True for the dense bandpass component
    env = NoiseCancellationEnv(config=cfg, freq_reward=True, ...)
    env = LoopShapingWrapper(env, psd_window=256, f_low=0.05, f_high=0.5)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from scipy.signal import butter, sosfilt, sosfilt_zi


class LoopShapingWrapper(gym.Wrapper):
    """Replace the env reward with a hybrid loop-shaping reward.

    Blends a dense per-step band-limited reward (causal Butterworth
    bandpass → instantaneous power improvement) with a spectral bonus
    computed from the sliding-window PSD sensitivity function.

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
        Mixing weight λ for the spectral bonus relative to the dense
        bandpass reward.  Default 0.5.
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
        fs: float = 4.0,
    ):
        super().__init__(env)
        self.psd_window = psd_window
        self.f_low = f_low
        self.f_high = f_high
        self.amplification_penalty = amplification_penalty
        self.spectral_weight = spectral_weight
        self.fs = fs

        # --- Dense bandpass reward (per-step, causal) ---
        nyq = fs / 2.0
        low = max(f_low, 0.01)
        high = min(f_high, nyq * 0.99)
        self._bp_sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")

        # --- Spectral (PSD) reward ---
        # Frequency bins and band masks
        self._freqs = np.fft.rfftfreq(psd_window, d=1.0 / fs)
        self._in_band = (self._freqs >= f_low) & (self._freqs <= f_high)
        self._out_band = ~self._in_band & (self._freqs > 0)  # exclude DC

        # Hann window for spectral leakage reduction
        self._window = np.hanning(psd_window)
        self._win_power = float(np.sum(self._window ** 2))

        # Circular buffers
        self._y_buf = np.zeros(psd_window, dtype=np.float64)
        self._e_buf = np.zeros(psd_window, dtype=np.float64)
        self._buf_pos: int = 0
        self._buf_full: bool = False

        # Bandpass filter state (initialised in reset)
        self._bp_zi_y: np.ndarray | None = None
        self._bp_zi_e: np.ndarray | None = None

    def reset(self, **kwargs):
        self._y_buf[:] = 0.0
        self._e_buf[:] = 0.0
        self._buf_pos = 0
        self._buf_full = False

        # Reset causal bandpass filter states
        zi_template = sosfilt_zi(self._bp_sos)
        self._bp_zi_y = zi_template * 0.0
        self._bp_zi_e = zi_template * 0.0

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)

        y_t = info["main_raw"]
        e_t = info["main_clean"]

        # --- Dense reward: causal bandpass → instantaneous power improvement ---
        y_arr = np.array([y_t])
        e_arr = np.array([e_t])
        y_bp, self._bp_zi_y = sosfilt(self._bp_sos, y_arr, zi=self._bp_zi_y)
        e_bp, self._bp_zi_e = sosfilt(self._bp_sos, e_arr, zi=self._bp_zi_e)
        dense_reward = float(y_bp[0] ** 2 - e_bp[0] ** 2)

        # --- Spectral reward: PSD-based sensitivity function ---
        idx = self._buf_pos % self.psd_window
        self._y_buf[idx] = y_t
        self._e_buf[idx] = e_t
        self._buf_pos += 1
        if self._buf_pos >= self.psd_window:
            self._buf_full = True

        if self._buf_full:
            spectral_reward, sens_info = self._spectral_reward()
            info.update(sens_info)
        else:
            spectral_reward = 0.0

        reward = dense_reward + self.spectral_weight * spectral_reward

        info["dense_reward"] = dense_reward
        info["spectral_reward"] = spectral_reward

        return obs, reward, terminated, truncated, info

    def _spectral_reward(self):
        """Compute the spectral loop-shaping bonus from the current buffer."""
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
        log_sens = np.log10(sensitivity_sq)

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

        spectral = in_band_suppression - self.amplification_penalty * amplification

        sens_info = {
            "loop_shaping_reward": spectral,
            "sensitivity_in_band_db": -10.0 * in_band_suppression,
            "sensitivity_out_band_db": 10.0 * amplification,
        }
        return spectral, sens_info
