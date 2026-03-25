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
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt


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


# ---------------------------------------------------------------------------
# Seismic / LIGO-motivated model
# ---------------------------------------------------------------------------
#
# Key differences from the polynomial toy model
# -----------------------------------------------
# 1. Coupling is a LINEAR time-varying FIR filter  y(t) = h(t) ⊛ w(t)
#    This is physically correct: seismic coupling in LIGO is a mechanical
#    transfer function (resonance), not a polynomial in amplitude.
#
# 2. Ground motion is COLOURED broadband noise (approximating a seismic
#    spectrum with 1/f² roll-off below ~1 Hz), not a single-frequency sinusoid.
#
# 3. Parameter drift follows an Ornstein–Uhlenbeck (OU) process — the
#    physically appropriate model for thermal fluctuations and slow alignment
#    changes (timescale: minutes).
#
# 4. Regime changes are jumps between discrete coupling transfer functions
#    (different resonance frequency / gain), representing sudden coupling
#    path changes after lock-loss/reacquisition or alignment corrections.
#
#
# 5. Multi-source: two seismometers with INDEPENDENT linear FIR couplings.
#    This is still a linear (separable) multi-channel problem — a sufficiently
#    long LMS filter with both inputs can in principle solve it, so the
#    challenge is purely adaptation speed and filter length, not nonlinearity.
#    (Contrast with the polynomial --multi-source cross-term E·w1·w2.)
#
# 6. Tilt-to-length (T2L) cross-coupling (multi_source + tilt_coupling).
#    In LIGO-like detectors, ground tilt θ(t) is a primary noise source at
#    low frequencies.  For Rayleigh waves, tilt is proportional to the
#    horizontal derivative of vertical displacement:
#
#        θ(t)  ≈  (ω/c_R) · v_z(t)  →  H_tilt(z) ⊛ w2(t)
#
#    where H_tilt is a differentiator (highpass) and c_R the Rayleigh wave
#    phase velocity.  θ then couples bilinearly into the main channel:
#
#        C_T2L(t) = T(t) · [H_tilt ⊛ w2(t)] · w1(t)
#
#    This product of two filtered channels CANNOT be cancelled by any linear
#    two-channel filter (LMS/NLMS on [w1, w2]).  The irreducible floor for
#    classical methods is  sqrt(rms(C_T2L)² + oracle²).  An RL agent that
#    learns to compute the product can cancel it, giving a genuine advantage.
#
#    Physical justification:
#      - The horizontal seismometer (w1) measures true horizontal translation
#        plus a small tilt contamination: a_x ≈ Ẍ_x + g·θ.
#      - The pendulum converts horizontal motion to test-mass displacement.
#      - If the mirror is also tilted by θ, the effective T2L coupling is
#        proportional to θ·x_horizontal — bilinear in θ and w1.
#      - T(t) varies slowly with mirror alignment (OU process, hours timescale).

@dataclass
class SeismicConfig:
    """
    Physical parameters for the seismic noise cancellation problem.

    Models a LIGO-like witness-based noise cancellation setup:
      y(t) = h(t) ⊛ w(t)  +  n_sensor(t)
    where h(t) is a slowly drifting resonant FIR coupling filter and
    w(t) is broadband seismic ground motion.
    """

    # --- sampling ---
    fs: float = 128.0

    # --- seismic ground motion ---
    seismic_amplitude: float = 1.0      # normalised RMS of witness channel
    witness_noise_sigma: float = 0.02   # seismometer self-noise (small)

    # --- coupling filter (resonant mechanical mode) ---
    filter_length: int = 64             # FIR taps
    coupling_gain: float = 2.0          # nominal coupling RMS gain
    resonance_freq: float = 3.0         # Hz  — isolation system resonance
    resonance_q: float = 4.0            # quality factor Q = f_r / bandwidth

    # --- thermal / alignment drift (Ornstein–Uhlenbeck) ---
    thermal_timescale: float = 300.0    # seconds  (≈ 5 min thermal time const.)
    gain_drift_sigma: float = 0.5       # OU stationary std of gain fluctuation
    freq_drift_sigma: float = 0.8       # OU stationary std of freq fluctuation (Hz)

    # --- main channel sensor noise ---
    sensor_noise_sigma: float = 0.05    # residual noise after passive isolation

    # --- regime changes (sudden coupling path change) ---
    regime_changes: bool = False
    n_regimes: int = 4
    mean_hold_time: float = 15.0        # seconds — avg hold between switches

    # --- multi-source (second seismometer at independent location) ---
    multi_source: bool = False
    w2_coupling_gain: float = 1.5
    w2_resonance_freq: float = 6.0      # Hz  — different resonance
    w2_resonance_q: float = 3.0

    # --- tilt-to-length bilinear cross-coupling (requires multi_source=True) ---
    # Models the Rayleigh-wave tilt-to-length mechanism:
    #   C_T2L(t) = T(t) · [H_tilt ⊛ w2(t)] · w1(t)
    # where H_tilt is a finite-difference approximation of d/dt (tilt proxy)
    # and T(t) is an OU-drifting alignment-dependent coupling gain.
    tilt_coupling: bool = False
    t2l_gain: float = 0.8               # nominal T2L coupling gain
    t2l_gain_drift_sigma: float = 0.2   # OU fluctuation of T2L gain
    t2l_thermal_timescale: float = 600.0  # seconds — alignment changes slowly


# ---------------------------------------------------------------------------
# Seismic helpers
# ---------------------------------------------------------------------------

def _make_resonant_fir(
    gain: float, f_r: float, Q: float, M: int, fs: float
) -> np.ndarray:
    """
    FIR approximation to a damped resonance impulse response:

        h[k] = exp(−α·k) · sin(ω_d·k),   k = 0, …, M−1

    where α = π·f_r / (Q·fs) and ω_d is the damped natural frequency.
    Normalised so  ‖h‖₂ = gain.

    Physically: a seismic isolation stage modelled as a mass-spring-damper
    system (Q ≈ 3–10 for typical suspension modes, f_r ≈ 1–10 Hz).
    """
    k = np.arange(M, dtype=float)
    alpha = np.pi * f_r / (Q * fs)                   # decay per sample
    omega_r = 2.0 * np.pi * f_r / fs
    omega_d = omega_r * np.sqrt(max(1.0 - 1.0 / (4.0 * Q**2), 0.0))
    h = np.exp(-alpha * k) * np.sin(omega_d * k)
    norm = np.sqrt(np.dot(h, h))
    if norm < 1e-12:
        return np.zeros(M)
    return (gain / norm) * h


def _ou_process(
    n: int,
    mean: float,
    sigma: float,
    timescale: float,
    fs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Discrete Ornstein–Uhlenbeck process with mean-reversion:

        X[i] = exp(−θ·dt)·X[i−1] + (1−exp(−θ·dt))·μ + noise[i]

    where θ = 1/timescale, noise ~ N(0, σ·√(1−exp(−2θ·dt))).
    Starts from the stationary distribution N(μ, σ²).
    """
    dt = 1.0 / fs
    theta = 1.0 / timescale
    exp_dt = np.exp(-theta * dt)
    noise_std = sigma * np.sqrt(1.0 - exp_dt**2)

    x = np.empty(n)
    x[0] = mean + rng.normal(0.0, sigma)
    raw_noise = rng.normal(0.0, noise_std, n)
    for i in range(1, n):
        x[i] = exp_dt * x[i - 1] + (1.0 - exp_dt) * mean + raw_noise[i]
    return x


def _seismic_ground_motion(
    n: int, amplitude: float, fs: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Broadband coloured noise approximating a seismic ground-motion spectrum.

    Method: integrate white noise (→ 1/f² brownian PSD), then bandpass
    [0.1, 20] Hz.  This captures the dominant energy at low frequencies
    while avoiding DC drift and aliasing.

    The result is normalised to RMS = amplitude.
    """
    extra = 512  # discard filter transient
    white = rng.normal(0.0, 1.0, n + extra)
    brown = np.cumsum(white) / np.sqrt(n + extra)

    f_low  = 0.1 / (fs / 2.0)
    f_high = min(20.0, fs * 0.45) / (fs / 2.0)
    sos = butter(4, [f_low, f_high], btype="band", output="sos")
    filtered = sosfilt(sos, brown)[extra:]

    rms_val = np.sqrt(np.mean(filtered**2))
    return amplitude * filtered / (rms_val + 1e-12)


# ---------------------------------------------------------------------------
# Seismic simulator
# ---------------------------------------------------------------------------

class SeismicSignalSimulator:
    """
    Generates one episode of seismic noise cancellation data.

    The coupling is a linear time-varying FIR filter (resonant mechanical
    transfer function) rather than a polynomial in witness amplitude.
    Parameters drift via Ornstein–Uhlenbeck (thermal) or step-change
    (coupling-path switch).

    Returns the same dict structure as SignalSimulator so the environment
    and baselines can be used unchanged.

    Usage
    -----
    >>> cfg = SeismicConfig()
    >>> sim = SeismicSignalSimulator(cfg, seed=0)
    >>> data = sim.generate_episode(duration=30.0)
    >>> data.keys()
    dict_keys(['time', 'witness', 'main', 'coupling', 'sensor_noise', 'true_signal'])
    """

    def __init__(self, config: SeismicConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def generate_episode(
        self,
        duration: float,
        signal_amplitude: float = 0.0,
        signal_freq: float = 10.0,
    ) -> dict:
        cfg = self.config
        n = int(duration * cfg.fs)
        t = np.arange(n) / cfg.fs

        # --- witness channel (seismic ground motion + self-noise) ---
        witness = _seismic_ground_motion(n, cfg.seismic_amplitude, cfg.fs, self.rng)
        witness += self.rng.normal(0.0, cfg.witness_noise_sigma, n)

        # --- sensor noise (small residual in main channel) ---
        sensor_noise = self.rng.normal(0.0, cfg.sensor_noise_sigma, n)

        # --- injected test signal (zero during training) ---
        true_signal = (
            signal_amplitude * np.sin(2.0 * np.pi * signal_freq * t)
            if signal_amplitude > 0
            else np.zeros(n)
        )

        # --- coupling ---
        if cfg.regime_changes:
            coupling, regime = self._regime_coupling(witness, n)
        else:
            coupling = self._drifting_coupling(
                witness, n,
                cfg.coupling_gain, cfg.resonance_freq, cfg.resonance_q,
                cfg.gain_drift_sigma, cfg.freq_drift_sigma,
            )
            regime = None

        witness2 = None
        coupling_t2l = None
        if cfg.multi_source:
            witness2 = _seismic_ground_motion(
                n, cfg.seismic_amplitude * 0.8, cfg.fs, self.rng
            )
            witness2 += self.rng.normal(0.0, cfg.witness_noise_sigma, n)
            coupling2 = self._drifting_coupling(
                witness2, n,
                cfg.w2_coupling_gain, cfg.w2_resonance_freq, cfg.w2_resonance_q,
                cfg.gain_drift_sigma * 0.8, cfg.freq_drift_sigma * 0.5,
            )
            coupling = coupling + coupling2

            if cfg.tilt_coupling:
                coupling_t2l = self._tilt_to_length_coupling(witness, witness2, n)
                coupling = coupling + coupling_t2l

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
        if coupling_t2l is not None:
            result["coupling_t2l"] = coupling_t2l
        if regime is not None:
            result["regime"] = regime
        return result

    # ------------------------------------------------------------------
    # Internal coupling generators
    # ------------------------------------------------------------------

    def _drifting_coupling(
        self,
        witness: np.ndarray,
        n: int,
        nom_gain: float,
        nom_freq: float,
        Q: float,
        gain_sigma: float,
        freq_sigma: float,
    ) -> np.ndarray:
        """
        Time-varying linear FIR coupling with OU-drifting parameters.

        The resonance frequency and gain follow independent OU processes,
        representing slow thermal / alignment drift.  The filter is
        recomputed at each sample from the current parameters.
        """
        cfg = self.config
        M = cfg.filter_length

        gain_t = _ou_process(n, nom_gain, gain_sigma, cfg.thermal_timescale, cfg.fs, self.rng)
        freq_t = _ou_process(n, nom_freq, freq_sigma, cfg.thermal_timescale, cfg.fs, self.rng)
        gain_t = np.clip(gain_t, 0.3, 5.0)
        freq_t = np.clip(freq_t, 0.5, cfg.fs / 4.0)

        coupling = np.zeros(n)
        for i in range(M, n):
            h = _make_resonant_fir(gain_t[i], freq_t[i], Q, M, cfg.fs)
            coupling[i] = np.dot(h, witness[i - M:i][::-1])
        return coupling

    def _regime_coupling(
        self, witness: np.ndarray, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Piecewise-constant coupling: K pre-sampled FIR filters, Poisson switches.

        Represents sudden coupling path changes (lock-loss, alignment jump).
        Each regime has a different resonance frequency and gain, sampled
        uniformly to span a realistic range.
        """
        cfg = self.config
        K, M = cfg.n_regimes, cfg.filter_length

        # Sample K distinct FIR filters
        gains = self.rng.uniform(0.8, 3.5, K)
        freqs = self.rng.uniform(1.0, 10.0, K)

        filters = np.stack([
            _make_resonant_fir(gains[k], freqs[k], cfg.resonance_q, M, cfg.fs)
            for k in range(K)
        ])  # (K, M)

        # Poisson-distributed hold times
        schedule = np.empty(n, dtype=np.int32)
        pos = 0
        regime = int(self.rng.integers(0, K))
        while pos < n:
            hold = max(1, int(self.rng.exponential(cfg.mean_hold_time * cfg.fs)))
            end = min(pos + hold, n)
            schedule[pos:end] = regime
            pos = end
            regime = int(self.rng.integers(0, K))

        coupling = np.zeros(n)
        for i in range(M, n):
            coupling[i] = np.dot(filters[schedule[i]], witness[i - M:i][::-1])
        return coupling, schedule

    def _tilt_to_length_coupling(
        self, w1: np.ndarray, w2: np.ndarray, n: int
    ) -> np.ndarray:
        """
        Bilinear tilt-to-length (T2L) cross-coupling:

            C_T2L(t) = T(t) · θ_proxy(t) · w1(t)

        Physical mechanism
        ------------------
        For Rayleigh waves, ground tilt θ is proportional to the temporal
        derivative of vertical displacement:

            θ(ω) = (ω / c_R) · v_z(ω)   →   θ ≈ (1/c_R) · dv_z/dt

        In the time domain this is a first-difference (highpass) FIR applied
        to the vertical seismometer w2:

            θ_proxy[t] = w2[t] − w2[t−1]         (normalised)

        θ then couples bilinearly with the horizontal motion w1 into the main
        channel via the pendulum + mirror-alignment mechanism.  T(t) is the
        alignment-dependent T2L gain, which drifts on the timescale of mirror
        angular control corrections (typically tens of minutes to hours).

        Why linear filters fail
        -----------------------
        C_T2L is the point-wise product of θ_proxy (a function of w2) and w1.
        No linear combination of w1 and w2 can represent a product — the
        Volterra kernel has support off the diagonal.  The NLMS floor is:

            residual_floor ≥ sqrt(rms(C_T2L)² + oracle²)

        An RL agent that learns to multiply θ_proxy by w1 can cancel C_T2L
        and approach the oracle floor.
        """
        cfg = self.config

        # --- tilt proxy: first-difference of vertical seismometer ---
        # Δw2[t] = w2[t] − w2[t−1]  approximates dv_z/dt ∝ θ for Rayleigh waves.
        # We use a normalised 2-tap FIR to keep units consistent.
        tilt_proxy = np.empty(n)
        tilt_proxy[0] = 0.0
        tilt_proxy[1:] = w2[1:] - w2[:-1]
        # Normalise to unit RMS so T2L gain has interpretable units
        rms_tilt = float(np.sqrt(np.mean(tilt_proxy[1:]**2))) + 1e-12
        tilt_proxy /= rms_tilt

        # --- OU-drifting T2L gain (alignment-dependent) ---
        T_gain = _ou_process(
            n,
            mean=cfg.t2l_gain,
            sigma=cfg.t2l_gain_drift_sigma,
            timescale=cfg.t2l_thermal_timescale,
            fs=cfg.fs,
            rng=self.rng,
        )
        T_gain = np.clip(T_gain, 0.05, cfg.t2l_gain * 2.5)

        # --- bilinear product ---
        # Normalise w1 contribution similarly so the gain is interpretable
        rms_w1 = float(np.sqrt(np.mean(w1**2))) + 1e-12
        return T_gain * tilt_proxy * (w1 / rms_w1)
