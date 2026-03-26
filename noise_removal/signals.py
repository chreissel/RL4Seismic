"""
Signal generation for RL-based seismic noise cancellation.

Physical setup
--------------
Models a LIGO-like witness-based noise cancellation system:

    y(t) = h(t) ⊛ w(t)  +  n_sensor(t)

where:
  - w(t)          : seismic ground motion (coloured broadband noise)
  - h(t)          : slowly drifting resonant FIR coupling filter
                    (mechanical transfer function, Ornstein–Uhlenbeck drift)
  - n_sensor(t)   : i.i.d. Gaussian sensor noise

Coupling models
---------------
Single-source (default):
    y(t) = h(t) ⊛ w1(t)
    Linear FIR coupling with OU-drifting resonance frequency and gain.

Multi-source (multi_source=True):
    y(t) = h1(t) ⊛ w1(t)  +  h2(t) ⊛ w2(t)
    Two independent seismometers with separate FIR couplings.

Tilt-to-length (tilt_coupling=True, requires multi_source):
    y(t) = h1(t)⊛w1 + h2(t)⊛w2 + T(t)·θ_proxy(t)·w1(t)
    The T2L term is a bilinear product of tilt proxy and horizontal motion.
    It CANNOT be cancelled by any linear two-channel filter (LMS/NLMS).
    An RL agent that learns the product can approach oracle performance.

Regime changes (regime_changes=True):
    Coupling path jumps between K discrete FIR filters at Poisson-distributed
    times, representing sudden coupling path changes (lock-loss, alignment jump).

Defaults are aligned with arXiv:2511.19682 (Reissel et al., 2025):
  - 4 Hz sampling rate (matching their downsampled data stream)
  - 0.1–0.3 Hz microseismic band (ocean-wave-generated noise)
  - 240-tap FIR = 60 s context (matching their LSTM input window)
  - Thermal timescale ~10 min (OU drift of mechanical coupling parameters)
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
class SeismicConfig:
    """
    Physical parameters for the seismic noise cancellation problem.

    Models a LIGO-like witness-based noise cancellation setup:
      y(t) = h(t) ⊛ w(t)  +  n_sensor(t)
    where h(t) is a slowly drifting resonant FIR coupling filter and
    w(t) is broadband seismic ground motion.

    Defaults are aligned with arXiv:2511.19682 (Reissel et al., 2025):
      - 4 Hz sampling rate (matching their downsampled data stream)
      - 0.1–0.3 Hz microseismic band (ocean-wave-generated noise)
      - 240-tap FIR = 60 s context (matching their LSTM input window)
      - Thermal timescale ~10 min (OU drift of mechanical coupling parameters)
    """

    # --- sampling ---
    fs: float = 4.0                     # Hz — matches paper's 4 Hz downsampled rate

    # --- seismic ground motion ---
    seismic_amplitude: float = 1.0      # normalised RMS of witness channel
    witness_noise_sigma: float = 0.02   # seismometer self-noise (small)

    # --- coupling filter (microseismic resonance) ---
    filter_length: int = 240            # FIR taps = 60 s × 4 Hz (paper context window)
    coupling_gain: float = 2.0          # nominal coupling RMS gain
    resonance_freq: float = 0.2         # Hz  — microseismic band (0.1–0.3 Hz)
    resonance_q: float = 5.0            # quality factor Q = f_r / bandwidth

    # --- thermal / alignment drift (Ornstein–Uhlenbeck) ---
    thermal_timescale: float = 600.0    # seconds  (≈ 10 min thermal time const.)
    gain_drift_sigma: float = 0.3       # OU stationary std of gain fluctuation
    freq_drift_sigma: float = 0.03      # Hz — tight drift around 0.2 Hz resonance

    # --- main channel sensor noise ---
    sensor_noise_sigma: float = 0.05    # residual noise after passive isolation

    # --- regime changes (sudden coupling path change) ---
    regime_changes: bool = False
    n_regimes: int = 4
    mean_hold_time: float = 120.0       # seconds — longer holds at 4 Hz

    # --- multi-source (second seismometer / rotational DOF) ---
    # Mirrors the paper's multi-DOF setup where translational and rotational
    # channels couple nonlinearly (tilt-to-length mechanism).
    multi_source: bool = False
    w2_coupling_gain: float = 1.5
    w2_resonance_freq: float = 0.35     # Hz  — slightly different resonance
    w2_resonance_q: float = 4.0

    # --- tilt-to-length bilinear cross-coupling (requires multi_source=True) ---
    # Models the Rayleigh-wave tilt-to-length mechanism:
    #   C_T2L(t) = T(t) · [H_tilt ⊛ w2(t)] · w1(t)
    # where H_tilt is a finite-difference approximation of d/dt (tilt proxy)
    # and T(t) is an OU-drifting alignment-dependent coupling gain.
    # This is the dominant nonlinearity identified in arXiv:2511.19682.
    tilt_coupling: bool = False
    t2l_gain: float = 0.8               # nominal T2L coupling gain
    t2l_gain_drift_sigma: float = 0.2   # OU fluctuation of T2L gain
    t2l_thermal_timescale: float = 600.0  # seconds — alignment changes slowly


# ---------------------------------------------------------------------------
# Helpers
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
    system (Q ≈ 3–10 for typical suspension modes, f_r ≈ 0.1–0.5 Hz).
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
    [0.05–1.5 Hz].  This captures the dominant energy at low frequencies
    while avoiding DC drift and aliasing.

    The result is normalised to RMS = amplitude.
    """
    extra = min(512, n // 2)  # discard filter transient
    white = rng.normal(0.0, 1.0, n + extra)
    brown = np.cumsum(white) / np.sqrt(n + extra)

    # Target microseismic band (0.05–1.5 Hz), clamped to Nyquist
    f_low  = max(0.02, 0.05) / (fs / 2.0)
    f_high = min(1.5, fs * 0.45) / (fs / 2.0)
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
    transfer function) with parameters drifting via Ornstein–Uhlenbeck
    (thermal) or step-change (coupling-path switch).

    Optionally includes a second witness channel (multi_source) and/or
    a bilinear tilt-to-length coupling term (tilt_coupling).

    Usage
    -----
    >>> cfg = SeismicConfig()
    >>> sim = SeismicSignalSimulator(cfg, seed=0)
    >>> data = sim.generate_episode(duration=300.0)
    >>> data.keys()
    dict_keys(['time', 'witness', 'main', 'coupling', 'sensor_noise', 'true_signal'])

    With multi_source=True the dict also contains 'witness2'.
    With tilt_coupling=True the dict also contains 'coupling_t2l'.
    With regime_changes=True the dict also contains 'regime'.
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
        """
        Simulate *duration* seconds of data.

        Returns
        -------
        dict with keys
          'time'         : (N,) time axis in seconds
          'witness'      : (N,) witness channel 1 (horizontal seismometer)
          'witness2'     : (N,) witness channel 2  [multi_source only]
          'main'         : (N,) main channel
          'coupling'     : (N,) total coupling (sum of all terms)
          'coupling_t2l' : (N,) T2L bilinear term  [tilt_coupling only]
          'sensor_noise' : (N,) Gaussian sensor noise
          'true_signal'  : (N,) injected test signal (zeros during training)
          'regime'       : (N,) int regime index  [regime_changes only]
        """
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

        # --- primary coupling ---
        if cfg.regime_changes:
            coupling, regime = self._regime_coupling(witness, n)
        else:
            coupling = self._drifting_coupling(
                witness, n,
                cfg.coupling_gain, cfg.resonance_freq, cfg.resonance_q,
                cfg.gain_drift_sigma, cfg.freq_drift_sigma,
            )
            regime = None

        # --- optional second seismometer ---
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

        # Sample K distinct FIR filters spanning the microseismic band
        gains = self.rng.uniform(0.8, 3.5, K)
        freqs = self.rng.uniform(0.1, min(0.5, cfg.fs * 0.4), K)

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
        tilt_proxy = np.empty(n)
        tilt_proxy[0] = 0.0
        tilt_proxy[1:] = w2[1:] - w2[:-1]
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
        rms_w1 = float(np.sqrt(np.mean(w1**2))) + 1e-12
        return T_gain * tilt_proxy * (w1 / rms_w1)
