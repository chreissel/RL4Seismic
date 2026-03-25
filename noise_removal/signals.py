"""
Signal generation for the RL noise-removal demonstration.

Physical setup
--------------
* Main channel (y):  sampled at 128 Hz.
  y(t) = s(t)  +  n_sensor(t)  +  f(w1(t), [w2(t)], t)

  - s(t)          : optional injected test signal (zero during training)
  - n_sensor(t)   : i.i.d. Gaussian sensor noise  ~ N(0, sigma_n)
  - f(...)        : non-linear, time-varying coupling

* Witness channel (w1): monitors a ~1 Hz environmental disturbance.

* Optional second witness (w2): monitors an independent ~2 Hz disturbance
  that interacts with w1 through a cross-term, making the coupling
  non-separable — only a method that can compute or learn the product
  w1·w2 can cancel it fully.

Coupling models
---------------
Single-source (default):
  f(w1, t) = A(t)·w1  +  B(t)·w1²  +  C(t)·w1³
  Coefficients drift smoothly with incommensurate periods (30/47/61 s).

Multi-source (--multi-source flag):
  f(w1, w2, t) = A(t)·w1  +  B(t)·w1²  +  C(t)·w1³   (polynomial on w1)
               + D(t)·w2                                (linear on w2)
               + E(t)·w1·w2                             (cross-term — not
                                                          separable!)
  The cross-term E(t)·w1·w2 cannot be represented as a linear combination
  of w1 and w2 alone.

Regime-change (--regime-changes flag):
  Same polynomial structure as single-source, but coefficients are
  piecewise-constant: they jump instantaneously between K pre-sampled
  discrete modes at Poisson-distributed switch times (mean hold ~8 s).
  Adaptive filters (LMS/IIR) need O(100) samples to re-converge after
  each jump.  A recurrent RL agent trained on many switches can learn to
  detect the residual spike, identify the new regime, and recover in far
  fewer steps.
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

    # --- witness channel 1 ---
    witness_freq: float = 1.0       # Hz
    witness_amplitude: float = 1.0
    witness_noise_sigma: float = 0.05

    # --- witness channel 2 (multi-source only) ---
    w2_freq: float = 2.0            # Hz  (different frequency → independent source)
    w2_amplitude: float = 0.8
    w2_noise_sigma: float = 0.05

    # --- main channel ---
    sensor_noise_sigma: float = 0.3

    # --- coupling slow-drift periods (seconds) ---
    coupling_periods: tuple = field(default_factory=lambda: (30.0, 47.0, 61.0))

    # --- problem variant ---
    multi_source: bool = False      # enable second witness + cross-term coupling
    regime_changes: bool = False    # piecewise-constant coupling with sudden jumps
    n_regimes: int = 4              # number of discrete coupling modes
    mean_hold_time: float = 3.0     # average seconds per regime (Poisson).
                                    # Filter convergence time ≈ 1/(2μ·power) ≈ 8 s,
                                    # so hold << 8 s keeps the filter always mid-
                                    # convergence, exposing the switch-recovery gap.


# ---------------------------------------------------------------------------
# Time-varying, non-linear coupling
# ---------------------------------------------------------------------------

class TimeVaryingCoupling:
    """
    Single-source: f(w1, t) = A(t)·w1 + B(t)·w1² + C(t)·w1³

    Coefficients oscillate slowly (incommensurate periods) so the coupling
    is non-linear and continuously time-varying.

    Typical ranges
    --------------
    A(t) ∈ [1.0, 3.0]   (linear gain, dominant term)
    B(t) ∈ [0.2, 0.8]   (quadratic)
    C(t) ∈ [0.1, 0.3]   (cubic)
    """

    def __init__(self, config: SignalConfig):
        self.config = config
        T1, T2, T3 = config.coupling_periods

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
        return tuple(result)

    def __call__(self, witness: np.ndarray, t: np.ndarray) -> np.ndarray:
        w = np.asarray(witness, dtype=float)
        A, B, C = self.coefficients(np.asarray(t, dtype=float))
        return A * w + B * w**2 + C * w**3


class MultiSourceCoupling:
    """
    Multi-source: f(w1, w2, t) = A(t)·w1 + B(t)·w1² + C(t)·w1³
                                + D(t)·w2
                                + E(t)·w1·w2

    The cross-term E(t)·w1·w2 is not separable into independent functions
    of w1 and w2 alone.  An adaptive filter receiving both channels as
    separate inputs cannot represent it without explicitly forming the
    product — but a recurrent RL agent can learn it from the observation
    window.

    Extra coefficient ranges
    ------------------------
    D(t) ∈ [0.7, 2.3]   (linear term on w2, period 37 s)
    E(t) ∈ [1.5, 4.5]   (cross-term w1·w2,  period 53 s)

    The cross-term is sized so that it accounts for ~60 % of the total
    coupling RMS, giving adaptive filters an irreducible residual floor
    of ~4–5× oracle (vs 1–2× oracle for single-source).  The RL agent,
    which sees both witness windows, can learn the product implicitly
    and in principle approach oracle.
    """

    def __init__(self, config: SignalConfig):
        self._single = TimeVaryingCoupling(config)
        # Cross-source coefficients with incommensurate periods.
        # E offset raised from 0.8 → 3.0 so the cross-term dominates
        # and creates an irreducible floor for linear baselines.
        self._d_params = (1.5, 0.8, 37.0, np.pi / 7)   # D(t): w2
        self._e_params = (3.0, 1.5, 53.0, np.pi / 4)   # E(t): w1·w2  (dominant)

    def _coeff(self, params, t):
        offset, amp, period, phase = params
        return offset + amp * np.sin(2 * np.pi * t / period + phase)

    def __call__(
        self, witness1: np.ndarray, witness2: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        w1 = np.asarray(witness1, dtype=float)
        w2 = np.asarray(witness2, dtype=float)
        t  = np.asarray(t, dtype=float)
        D  = self._coeff(self._d_params, t)
        E  = self._coeff(self._e_params, t)
        return self._single(w1, t) + D * w2 + E * w1 * w2


class RegimeChangeCoupling:
    """
    Piecewise-constant coupling: f(w, t) = A_k·w + B_k·w² + C_k·w³

    At episode start, K regimes are sampled uniformly:
        A_k ~ U[1.0, 3.0],  B_k ~ U[0.2, 0.8],  C_k ~ U[0.1, 0.3]

    The active regime switches on a Poisson process (exponentially distributed
    hold times with mean = config.mean_hold_time seconds).  Each switch picks
    a new regime uniformly from all K, including the current one.

    Impact on adaptive filters
    --------------------------
    LMS and IIR filters maintain a weight vector that tracks the current
    regime.  After a sudden jump the weights are mismatched and the residual
    spikes; convergence back to near-oracle takes O(1 / (μ · signal_power))
    samples — typically 100-500 steps at default step sizes.

    A recurrent RL agent trained across many episodes with random switch
    times can learn to detect the residual spike pattern and immediately
    output an action close to the new regime's coupling, recovering in
    O(window_size) steps.
    """

    def __init__(self, config: SignalConfig):
        self.config = config

    def make_schedule(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample K regime coefficient sets and a per-sample regime index.

        Returns
        -------
        schedule : (n,) int array  — index into self.A/B/C for each sample
        Also sets self.A, self.B, self.C as (K,) arrays for this episode.
        """
        K = self.config.n_regimes
        fs = self.config.fs
        mean_hold = self.config.mean_hold_time

        # Sample K distinct coupling regimes
        self.A = rng.uniform(1.0, 3.0, K)
        self.B = rng.uniform(0.2, 0.8, K)
        self.C = rng.uniform(0.1, 0.3, K)

        # Build schedule via Poisson-distributed hold times
        schedule = np.empty(n, dtype=np.int32)
        pos = 0
        regime = int(rng.integers(0, K))
        while pos < n:
            hold = max(1, int(rng.exponential(mean_hold * fs)))
            end = min(pos + hold, n)
            schedule[pos:end] = regime
            pos = end
            regime = int(rng.integers(0, K))
        return schedule

    def __call__(self, witness: np.ndarray, schedule: np.ndarray) -> np.ndarray:
        """
        Evaluate coupling given a pre-generated regime schedule.

        Parameters
        ----------
        witness  : (N,) witness channel
        schedule : (N,) int array from make_schedule()
        """
        w = np.asarray(witness, dtype=float)
        A = self.A[schedule]
        B = self.B[schedule]
        C = self.C[schedule]
        return A * w + B * w**2 + C * w**3


# ---------------------------------------------------------------------------
# Full simulator
# ---------------------------------------------------------------------------

class SignalSimulator:
    """
    Generates one episode of data (single- or multi-source).

    Usage
    -----
    >>> cfg = SignalConfig()
    >>> sim = SignalSimulator(cfg, seed=0)
    >>> data = sim.generate_episode(duration=30.0)
    >>> data.keys()
    dict_keys(['time', 'witness', 'main', 'coupling', 'sensor_noise', 'true_signal'])

    With multi_source=True in config, the returned dict also contains 'witness2'.
    With regime_changes=True, the returned dict also contains 'regime' (int array).
    """

    def __init__(self, config: SignalConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        if config.multi_source:
            self.coupling = MultiSourceCoupling(config)
        elif config.regime_changes:
            self.coupling = RegimeChangeCoupling(config)
        else:
            self.coupling = TimeVaryingCoupling(config)

    def generate_episode(
        self,
        duration: float,
        signal_amplitude: float = 0.0,
        signal_freq: float = 10.0,
    ) -> dict:
        """
        Simulate *duration* seconds of data.

        Returns
        -------
        dict with keys
          'time'         : (N,) time axis in seconds
          'witness'      : (N,) witness channel 1
          'witness2'     : (N,) witness channel 2  [multi-source only]
          'main'         : (N,) main channel
          'coupling'     : (N,) true coupling f(w1, [w2], t)
          'sensor_noise' : (N,) Gaussian sensor noise
          'true_signal'  : (N,) injected test signal (zeros during training)
          'regime'       : (N,) int regime index  [regime-changes only]
        """
        cfg = self.config
        n = int(duration * cfg.fs)
        t = np.arange(n) / cfg.fs

        witness = (
            cfg.witness_amplitude * np.sin(2 * np.pi * cfg.witness_freq * t)
            + self.rng.normal(0.0, cfg.witness_noise_sigma, n)
        )

        sensor_noise = self.rng.normal(0.0, cfg.sensor_noise_sigma, n)

        true_signal = (
            signal_amplitude * np.sin(2 * np.pi * signal_freq * t)
            if signal_amplitude > 0
            else np.zeros(n)
        )

        witness2 = None
        regime = None

        if cfg.multi_source:
            witness2 = (
                cfg.w2_amplitude * np.sin(2 * np.pi * cfg.w2_freq * t)
                + self.rng.normal(0.0, cfg.w2_noise_sigma, n)
            )
            coupling = self.coupling(witness, witness2, t)
        elif cfg.regime_changes:
            regime = self.coupling.make_schedule(n, self.rng)
            coupling = self.coupling(witness, regime)
        else:
            coupling = self.coupling(witness, t)

        main = true_signal + sensor_noise + coupling

        result = {
            "time": t,
            "witness": witness,
            "main": main,
            "coupling": coupling,
            "sensor_noise": sensor_noise,
            "true_signal": true_signal,
        }
        if witness2 is not None:
            result["witness2"] = witness2
        if regime is not None:
            result["regime"] = regime
        return result
