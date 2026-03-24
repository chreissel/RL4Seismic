"""
Signal generation for the RL noise-removal demonstration.

Physical setup
--------------
* Main channel (y):  sampled at 128 Hz.
  y(t) = s(t)  +  n_sensor(t)  +  f(w(t), t)

  - s(t)          : optional injected test signal (zero during training)
  - n_sensor(t)   : i.i.d. Gaussian sensor noise  ~ N(0, sigma_n)
  - f(w(t), t)    : non-linear, time-varying coupling from the witness channel

* Witness channel (w): monitors a ~1 Hz environmental disturbance that leaks
  into the main channel through coupling f.

Coupling model
--------------
  f(w, t) = A(t)·w  +  B(t)·w²  +  C(t)·w³

where the coefficients A(t), B(t), C(t) drift slowly (periods 30-60 s) so
that any fixed linear or non-linear filter will become stale.  The RMS
coupling amplitude is several times larger than the sensor noise, making
subtraction critical for sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SignalConfig:
    """All physical parameters for the two-channel system."""

    # --- sampling ---
    fs: float = 128.0               # Hz

    # --- witness channel ---
    witness_freq: float = 1.0       # Hz  (the interfering environmental line)
    witness_amplitude: float = 1.0
    witness_noise_sigma: float = 0.05  # small noise on the witness itself

    # --- main channel ---
    sensor_noise_sigma: float = 0.3  # Gaussian sensor noise std

    # --- coupling slow-drift periods (seconds) ---
    coupling_periods: tuple = field(default_factory=lambda: (30.0, 47.0, 61.0))


# ---------------------------------------------------------------------------
# Time-varying, non-linear coupling
# ---------------------------------------------------------------------------

class TimeVaryingCoupling:
    """
    Implements  f(w, t) = A(t)·w + B(t)·w² + C(t)·w³

    Coefficients oscillate slowly so the coupling is *highly non-linear*
    (cubic) and *time-varying* — defeating both simple linear adaptive filters
    and any fixed non-linear model trained on past data.

    Typical ranges
    --------------
    A(t) ∈ [1.0, 3.0]   (linear gain, dominant term)
    B(t) ∈ [0.2, 0.8]   (quadratic, creates harmonic distortion)
    C(t) ∈ [0.1, 0.3]   (cubic)
    """

    def __init__(self, config: SignalConfig):
        self.config = config
        T1, T2, T3 = config.coupling_periods

        # Each coefficient: offset + amplitude * sin(2π t / period + phase)
        self._params = [
            # (offset, amplitude, period, phase)
            (2.0, 1.0, T1, 0.0),
            (0.5, 0.3, T2, np.pi / 3),
            (0.2, 0.1, T3, np.pi / 5),
        ]

    def coefficients(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (A(t), B(t), C(t)) for scalar or array t (in seconds)."""
        result = []
        for (offset, amp, period, phase) in self._params:
            result.append(offset + amp * np.sin(2 * np.pi * t / period + phase))
        return tuple(result)  # (A, B, C)

    def __call__(self, witness: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the coupling contribution.

        Parameters
        ----------
        witness : array-like, shape (N,)
        t       : array-like, shape (N,)  — time in seconds

        Returns
        -------
        coupling : ndarray, shape (N,)
        """
        w = np.asarray(witness, dtype=float)
        A, B, C = self.coefficients(np.asarray(t, dtype=float))
        return A * w + B * w**2 + C * w**3


# ---------------------------------------------------------------------------
# Full simulator
# ---------------------------------------------------------------------------

class SignalSimulator:
    """
    Generates one episode of two-channel data.

    Usage
    -----
    >>> cfg = SignalConfig()
    >>> sim = SignalSimulator(cfg, seed=0)
    >>> data = sim.generate_episode(duration=30.0)
    >>> data.keys()
    dict_keys(['time', 'witness', 'main', 'coupling', 'sensor_noise', 'true_signal'])
    """

    def __init__(self, config: SignalConfig, seed: Optional[int] = None):
        self.config = config
        self.coupling = TimeVaryingCoupling(config)
        self.rng = np.random.default_rng(seed)

    def generate_episode(
        self,
        duration: float,
        signal_amplitude: float = 0.0,
        signal_freq: float = 10.0,
    ) -> dict:
        """
        Simulate *duration* seconds of data.

        Parameters
        ----------
        duration         : length of the episode in seconds
        signal_amplitude : amplitude of a sinusoidal test signal injected into
                           the main channel (0 = no signal, used during training)
        signal_freq      : frequency of that test signal (Hz)

        Returns
        -------
        dict with keys
          'time'         : (N,) time axis in seconds
          'witness'      : (N,) witness channel
          'main'         : (N,) main channel  = true_signal + sensor_noise + coupling
          'coupling'     : (N,) true coupling contribution f(w, t)
          'sensor_noise' : (N,) Gaussian sensor noise realisation
          'true_signal'  : (N,) injected test signal (may be all zeros)
        """
        cfg = self.config
        n = int(duration * cfg.fs)
        t = np.arange(n) / cfg.fs

        # Witness channel: sinusoid at witness_freq + tiny noise
        witness = (
            cfg.witness_amplitude * np.sin(2 * np.pi * cfg.witness_freq * t)
            + self.rng.normal(0.0, cfg.witness_noise_sigma, n)
        )

        # Sensor noise on main channel
        sensor_noise = self.rng.normal(0.0, cfg.sensor_noise_sigma, n)

        # Optional test signal (higher frequency, so it lives in a different band)
        true_signal = (
            signal_amplitude * np.sin(2 * np.pi * signal_freq * t)
            if signal_amplitude > 0
            else np.zeros(n)
        )

        # Non-linear time-varying coupling
        coupling = self.coupling(witness, t)

        # Main channel
        main = true_signal + sensor_noise + coupling

        return {
            "time": t,
            "witness": witness,
            "main": main,
            "coupling": coupling,
            "sensor_noise": sensor_noise,
            "true_signal": true_signal,
        }
