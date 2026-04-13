"""
Signal generation for RL-based seismic noise cancellation.

Physical setup (arXiv:2511.19682, Reissel et al. 2025)
-------------------------------------------------------
Models a LIGO-like witness-based noise cancellation system with one obtaining
channel and two witness channels measuring seismic disturbance in X and Y:

    y(t) = h_x(t) ⊛ w_x(t)  +  C_tilt(t)  +  n_GS13X(t)

where:
  - w_x(t)        : seismic ground motion in X direction (witness X channel)
  - w_y(t)        : seismic ground motion in Y direction (witness Y channel)
  - h_x(t)        : slowly drifting resonant FIR coupling filter for X
                    (mechanical transfer function, Ornstein–Uhlenbeck drift)
  - C_tilt(t)     : tilt-horizontal coupling from Y seismic (see below)
  - n_GS13X(t)    : GS13X seismometer sensor noise in the obtaining channel

Obtaining channel
-----------------
The main (obtaining) channel is modelled as a GS13X broadband seismometer
measuring the LIGO test-mass degree of freedom.  Its self-noise floor is
captured by additive Gaussian noise n_GS13X(t) ~ N(0, σ²_sensor).

Witness channels
----------------
Two horizontal seismometers:
  w_x(t) : X-direction ground motion (inline with interferometer arm)
  w_y(t) : Y-direction ground motion (perpendicular to interferometer arm)

Each has a small additive self-noise term.

Tilt-horizontal coupling
------------------------
Low-frequency seismic waves (Rayleigh waves) tilt the ground and that tilt
couples into the main channel via the alignment-dependent gain T(t):

    C_tilt(t) = T(t) · θ_y_proxy(t)

where θ_y_proxy(t) is a low-passed version of the Y witness obtained by
passing w_y through a **double leaky integrator** (two cascaded first-order
low-pass / leaky-integrator stages):

    x1[t] = (1−α)·x1[t−1] + α·w_y[t]
    θ_y_proxy[t] = (1−α)·θ_y_proxy[t−1] + α·x1[t]

with α = 1 − exp(−Δt / τ_leak) and τ_leak = ``tilt_leak_timescale``.  This
gives a bounded, smooth tilt proxy whose PSD rolls off as 1/ω⁴ at high
frequency — avoiding the artificial high-frequency amplification of a
finite-difference (≈ d/dt) proxy, which boosts the PSD above the
microseismic band where tilt is not physically relevant.  T(t) is an
OU-drifting alignment gain representing the slowly changing mirror
alignment.

For constant T, any linear adaptive filter can cancel C_tilt by learning the
transfer function θ_y → main.  With drifting T(t) the filter must track the
gain; RL can exploit the slowly varying structure more effectively.

Setting drift=False freezes T(t) = t2l_gain (constant), making C_tilt a
purely linear function of w_y — cancellable by a static linear filter.

Regime changes (regime_changes=True)
-------------------------------------
Coupling path jumps between K discrete FIR filters at Poisson-distributed
times, representing sudden coupling path changes (lock-loss, alignment jump).

Signal processing parameters follow arXiv:2511.19682 (Reissel et al., 2025):
  - 4 Hz sampling rate (matching their downsampled data stream)
  - 0.1–0.3 Hz microseismic band (ocean-wave-generated noise)
  - 240-tap FIR = 60 s context (matching their LSTM input window)

The non-stationarity model (OU drift) is our own synthetic approximation.
arXiv:2511.19682 trains on real LIGO data, where coupling varies across many
timescales: slow thermal/alignment drift (minutes–hours), seasonal modulation
(months), and sudden discontinuities from lock-loss or maintenance.  The OU
process captures only the slow mean-reverting component; use --regime-changes
to additionally model sudden coupling path switches.
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

    Models a LIGO-like witness-based noise cancellation setup with one
    obtaining channel (GS13X sensor) and two witness channels (X and Y):

      y(t) = h_x(t) ⊛ w_x(t)  +  C_tilt(t)  +  n_GS13X(t)

    Signal processing parameters follow arXiv:2511.19682 (Reissel et al., 2025):
      - 4 Hz sampling rate (matching their downsampled data stream)
      - 0.1–0.3 Hz microseismic band (ocean-wave-generated noise)
      - 240-tap FIR = 60 s context (matching their LSTM input window)

    The OU drift model is a synthetic approximation of physical non-stationarity.
    Real LIGO coupling varies across many timescales simultaneously (minutes–hours
    for thermal/alignment drift, months for seasonal microseism modulation, plus
    sudden discontinuities from lock-loss and maintenance).  The OU process here
    captures only the slow mean-reverting component; --regime-changes adds sudden
    coupling path switches on top.

    Setting drift=False disables all Ornstein–Uhlenbeck drift, giving stationary
    coupling parameters.  This is useful for testing linear baselines: with
    drift=False and tilt_coupling=False, NLMS should converge to near-oracle
    performance.
    """

    # --- sampling ---
    fs: float = 4.0                      # Hz — matches paper's 4 Hz downsampled rate

    # --- seismic ground motion (both witness channels) ---
    seismic_amplitude: float = 1.0       # normalised RMS of each witness channel
    witness_noise_sigma: float = 0.02    # seismometer self-noise (both X and Y channels)

    # --- X-direction coupling filter (translational, inline with interferometer arm) ---
    filter_length: int = 240             # FIR taps = 60 s × 4 Hz (paper context window)
    coupling_gain: float = 0.5           # nominal coupling RMS gain (reduced from 2.0 so
                                         # NLMS converges quickly to its linear floor,
                                         # cleanly exposing the tilt coupling residual)
    resonance_freq: float = 0.14         # Hz  — secondary microseism peak (7 s period)
    resonance_q: float = 8.0             # quality factor Q = f_r / bandwidth (sharper)

    # --- thermal / alignment drift (Ornstein–Uhlenbeck) ---
    # Set drift=False to freeze all coupling parameters (stationary system).
    # Useful for verifying linear baselines and isolating the tilt-coupling
    # challenge from the drift challenge.
    drift: bool = True                   # enable OU-drifting coupling parameters
    thermal_timescale: float = 600.0     # seconds  (≈ 10 min thermal time const.)
    gain_drift_sigma: float = 0.1        # OU stationary std of gain fluctuation
    freq_drift_sigma: float = 0.01       # Hz — resonance frequency drift std

    # --- GS13X obtaining channel sensor noise ---
    # The GS13X (Geotech S-13) is a short-period (1 Hz) broadband seismometer
    # used at LIGO to sense test-mass motion.  Its self-noise floor is modelled
    # as additive white Gaussian noise with amplitude sensor_noise_sigma.
    sensor_noise_sigma: float = 0.05     # GS13X-like sensor noise (obtaining channel)

    # --- regime changes (sudden coupling path change) ---
    regime_changes: bool = False
    n_regimes: int = 4
    mean_hold_time: float = 120.0        # seconds — longer holds at 4 Hz

    # --- tilt-horizontal coupling (Y seismic → ground tilt → main channel) ---
    # Rayleigh waves propagating along Y create ground tilt about the X axis.
    # This tilt couples into the main channel via the alignment-dependent gain T(t):
    #
    #   C_tilt(t) = T(t) · θ_y_proxy(t)
    #
    # θ_y_proxy(t) is obtained by passing w_y through a **double leaky
    # integrator** (two cascaded first-order low-pass / leaky-integrator
    # stages with time constant ``tilt_leak_timescale``).  This gives a
    # bounded, smooth tilt proxy whose PSD rolls off as 1/ω⁴ at high
    # frequency, avoiding the high-frequency amplification of a naïve
    # finite-difference (≈ d/dt) proxy, which inflates the PSD above the
    # microseismic band where tilt is not physically relevant.  T(t) is the
    # OU-drifting alignment-dependent gain.
    #
    # With drift=True, T(t) is non-stationary (OU), so static linear filters
    # cannot fully cancel C_tilt without re-adapting.  With drift=False, T is
    # constant and a static linear filter CAN cancel C_tilt.
    tilt_coupling: bool = True
    tilt_leak_timescale: float = 2.0    # seconds — time constant of each of the
                                         # two cascaded leaky-integrator stages.
                                         # 2 s ≈ 0.08 Hz corner, just below the
                                         # 0.14 Hz secondary microseism peak:
                                         # preserves in-band tilt energy while
                                         # rolling off the high-frequency tail.
    t2l_gain: float = 43.0              # mean tilt-horizontal coupling gain.
                                         # Acts on the double-leaky-integrated
                                         # witness_y (not normalised), so the
                                         # resulting C_tilt RMS depends on
                                         # tilt_leak_timescale.  With the
                                         # defaults above C_tilt is tuned to
                                         # dominate over linear coupling ≈ 0.5.
    t2l_gain_drift_sigma: float = 2.7   # OU fluctuation of T(t)
    t2l_thermal_timescale: float = 600.0 # seconds — alignment changes slowly


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


def _double_leaky_integrator(
    x: np.ndarray, timescale: float, fs: float
) -> np.ndarray:
    """
    Apply two cascaded first-order leaky-integrator / low-pass stages:

        y1[t] = (1−α)·y1[t−1] + α·x[t]
        y2[t] = (1−α)·y2[t−1] + α·y1[t]

    with α = 1 − exp(−Δt / τ), τ = ``timescale``.  The resulting filter has
    unity DC gain and a 1/ω² magnitude rolloff per stage (1/ω⁴ cascaded) for
    frequencies well above the corner f_c ≈ 1 / (2π·τ).

    Used for the tilt-horizontal coupling proxy: the witness is smoothed with
    a well-behaved low-pass instead of a high-pass-like finite difference, so
    the resulting tilt proxy does not artificially amplify high frequencies in
    its PSD — only the physically relevant low-frequency (microseismic) band
    contributes meaningfully.
    """
    dt = 1.0 / fs
    alpha = 1.0 - np.exp(-dt / timescale)
    one_minus_alpha = 1.0 - alpha

    y1 = np.empty_like(x)
    y2 = np.empty_like(x)
    y1_prev = 0.0
    y2_prev = 0.0
    for i in range(x.shape[0]):
        y1_prev = one_minus_alpha * y1_prev + alpha * x[i]
        y2_prev = one_minus_alpha * y2_prev + alpha * y1_prev
        y1[i] = y1_prev
        y2[i] = y2_prev
    return y2


def _seismic_ground_motion(
    n: int, amplitude: float, fs: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Coloured noise approximating the Peterson NLNM seismic ground-motion
    spectrum as seen in downsampled LIGO data (arXiv:2511.19682).

    Method:
      1. 1/f² Brownian background (integrate white noise once).
      2. Resonant secondary microseism peak at ~0.14 Hz (7 s ocean period),
         the dominant spectral feature in the LIGO microseismic band.
      3. Mix both components, then bandpass [0.05–0.5 Hz] — matching the
         paper's 4 Hz downsampled stream where the anti-aliasing filter
         kills energy above ~0.5 Hz.

    The double-hump structure (primary ~0.07 Hz, secondary ~0.14 Hz) of the
    Peterson NLNM is approximated by the Brownian floor + the resonant peak.
    The result is normalised to RMS = amplitude.
    """
    extra = min(512, n // 2)  # discard filter transient
    N = n + extra
    white = rng.normal(0.0, 1.0, N)

    # Component 1: 1/f² Brownian background (low-freq energy floor)
    brown = np.cumsum(white) / np.sqrt(N)

    # Component 2: secondary microseism peak (~0.14 Hz, bandwidth ±0.04 Hz)
    # This is the dominant spectral feature in real LIGO seismic data.
    f_mic  = 0.14  # Hz — secondary microseism (7 s ocean period)
    bw_mic = 0.08  # Hz — peak width
    f_lo   = max(f_mic - bw_mic / 2, 0.02) / (fs / 2.0)
    f_hi   = min(f_mic + bw_mic / 2, fs * 0.45) / (fs / 2.0)
    sos_mic = butter(3, [f_lo, f_hi], btype="band", output="sos")
    peak_sec = sosfilt(sos_mic, white)

    # Mix: Brownian floor + dominant microseismic peak (2× weight)
    mixed = brown + 2.0 * peak_sec

    # Final bandpass: 0.05–0.5 Hz (matches paper's anti-aliased 4 Hz stream)
    f_low  = 0.05 / (fs / 2.0)
    f_high = min(0.5, fs * 0.45) / (fs / 2.0)
    sos_bp = butter(4, [f_low, f_high], btype="band", output="sos")
    filtered = sosfilt(sos_bp, mixed)[extra:]

    rms_val = np.sqrt(np.mean(filtered**2))
    return amplitude * filtered / (rms_val + 1e-12)


# ---------------------------------------------------------------------------
# Seismic simulator
# ---------------------------------------------------------------------------

class SeismicSignalSimulator:
    """
    Generates one episode of seismic noise cancellation data.

    Physical setup:
      - One obtaining channel (GS13X sensor) measuring interferometer output
      - Two witness channels: X direction (translational) and Y direction
      - X coupling: linear time-varying FIR filter with optional OU drift
      - Tilt-horizontal coupling: Y seismic → ground tilt → main channel

    The tilt-horizontal coupling C_tilt(t) = T(t) · θ_y_proxy(t) is a bilinear
    product of the slowly drifting alignment gain T(t) and a double-leaky-
    integrated version of witness_y (the tilt proxy).  With drift=True, static
    linear filters cannot track the drifting T(t); with drift=False they can.

    Usage
    -----
    >>> cfg = SeismicConfig()
    >>> sim = SeismicSignalSimulator(cfg, seed=0)
    >>> data = sim.generate_episode(duration=300.0)
    >>> data.keys()
    dict_keys(['time', 'witness_x', 'witness_y', 'main', 'coupling',
               'coupling_tilt', 'sensor_noise', 'true_signal'])

    With regime_changes=True the dict also contains 'regime'.

    Legacy keys 'witness' (= witness_x) and 'witness2' (= witness_y) are also
    included for backward compatibility.
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
        dict with keys:
          'time'          : (N,) time axis in seconds
          'witness_x'     : (N,) witness channel X (horizontal, inline with arm)
          'witness_y'     : (N,) witness channel Y (horizontal, perpendicular)
          'main'          : (N,) obtaining channel (GS13X sensor + coupling)
          'coupling'      : (N,) total coupling (X FIR + tilt-horizontal term)
          'coupling_tilt' : (N,) tilt-horizontal coupling term  [tilt_coupling only]
          'sensor_noise'  : (N,) GS13X sensor noise
          'true_signal'   : (N,) injected test signal (zeros during training)
          'regime'        : (N,) int regime index  [regime_changes only]

        Legacy keys 'witness' (alias for witness_x) and 'witness2' (alias for
        witness_y) are also present for backward compatibility.
        """
        cfg = self.config
        n = int(duration * cfg.fs)
        t = np.arange(n) / cfg.fs

        # --- witness X: seismic ground motion in X direction + self-noise ---
        witness_x = _seismic_ground_motion(n, cfg.seismic_amplitude, cfg.fs, self.rng)
        witness_x += self.rng.normal(0.0, cfg.witness_noise_sigma, n)

        # --- witness Y: seismic ground motion in Y direction + self-noise ---
        # Y is partially correlated with X (same wavefield) but independent
        witness_y = _seismic_ground_motion(
            n, cfg.seismic_amplitude * 0.8, cfg.fs, self.rng
        )
        witness_y += self.rng.normal(0.0, cfg.witness_noise_sigma, n)

        # --- GS13X sensor noise (obtaining channel) ---
        sensor_noise = self.rng.normal(0.0, cfg.sensor_noise_sigma, n)

        # --- injected test signal (zero during training) ---
        true_signal = (
            signal_amplitude * np.sin(2.0 * np.pi * signal_freq * t)
            if signal_amplitude > 0
            else np.zeros(n)
        )

        # --- X coupling: translational FIR from inline seismic (with optional drift) ---
        if cfg.regime_changes:
            coupling, regime = self._regime_coupling(witness_x, n)
        else:
            coupling = self._drifting_coupling(
                witness_x, n,
                cfg.coupling_gain, cfg.resonance_freq, cfg.resonance_q,
                cfg.gain_drift_sigma, cfg.freq_drift_sigma,
            )
            regime = None

        # --- tilt-horizontal coupling: Y seismic → ground tilt → main channel ---
        coupling_tilt = None
        if cfg.tilt_coupling:
            coupling_tilt = self._tilt_horizontal_coupling(witness_y, n)
            coupling = coupling + coupling_tilt

        main = true_signal + sensor_noise + coupling

        result = {
            "time": t,
            "witness_x": witness_x,
            "witness_y": witness_y,
            # legacy aliases (backward compatibility)
            "witness": witness_x,
            "witness2": witness_y,
            "main": main,
            "coupling": coupling,
            "sensor_noise": sensor_noise,
            "true_signal": true_signal,
        }
        if coupling_tilt is not None:
            result["coupling_tilt"] = coupling_tilt
            result["coupling_t2l"] = coupling_tilt   # legacy alias
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
        Time-varying linear FIR coupling with optional OU-drifting parameters.

        When config.drift=True, the resonance frequency and gain follow
        independent OU processes, representing slow thermal / alignment drift.
        When config.drift=False, fixed nominal values are used (stationary
        system — linear baselines should converge to near-oracle performance).
        """
        cfg = self.config
        M = cfg.filter_length

        if cfg.drift:
            gain_t = _ou_process(n, nom_gain, gain_sigma, cfg.thermal_timescale, cfg.fs, self.rng)
            freq_t = _ou_process(n, nom_freq, freq_sigma, cfg.thermal_timescale, cfg.fs, self.rng)
            gain_t = np.clip(gain_t, 0.3, 5.0)
            freq_t = np.clip(freq_t, max(0.02, nom_freq * 0.5), cfg.fs / 4.0)
        else:
            # Stationary coupling: fixed nominal filter throughout the episode
            gain_t = np.full(n, nom_gain)
            freq_t = np.full(n, nom_freq)

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

    def _tilt_horizontal_coupling(
        self, witness_y: np.ndarray, n: int
    ) -> np.ndarray:
        """
        Tilt-horizontal coupling from Y-direction seismic to main channel:

            C_tilt(t) = T(t) · θ_y_proxy(t)

        Physical mechanism
        ------------------
        Rayleigh waves propagating along Y tilt the ground about the X axis,
        and this tilt couples into the main channel via the alignment-dependent
        gain T(t), representing the mechanical lever arm between mirror tilt
        and length noise (controlled by mirror angular alignment).

        Tilt proxy
        ----------
        θ_y_proxy(t) is obtained by passing witness_y through a **double
        leaky integrator** — two cascaded first-order low-pass stages with
        time constant ``tilt_leak_timescale`` (see `_double_leaky_integrator`).
        The resulting filter has unity DC gain and a 1/ω⁴ magnitude rolloff
        well above the corner frequency, which avoids the high-frequency PSD
        amplification of a finite-difference (≈ d/dt) proxy.  Frequencies
        above the microseismic band therefore contribute negligibly to the
        tilt proxy — matching the physical observation that only low-frequency
        ground motion tilts the test masses in a way that matters.

        Drift
        -----
        When config.drift=True, T(t) follows an Ornstein–Uhlenbeck process
        (mean = t2l_gain, σ = t2l_gain_drift_sigma), representing slow
        changes in mirror alignment.  Any static linear filter applied to
        witness_y must re-adapt as T drifts.

        When config.drift=False, T(t) = t2l_gain (constant).  A static
        linear filter can then perfectly cancel C_tilt once converged.
        """
        cfg = self.config

        # --- tilt proxy: double leaky integrator of the Y seismometer ---
        # No per-episode normalisation: C_tilt = T(t) · tilt_proxy directly.
        # This ensures the coupling is a deterministic function of witness_y
        # and the coupling parameters, so offline methods (Wiener filter)
        # generalise across episodes without gain mismatch.
        tilt_proxy = _double_leaky_integrator(
            witness_y, cfg.tilt_leak_timescale, cfg.fs
        )

        # --- T(t): alignment-dependent coupling gain (drifts or fixed) ---
        if cfg.drift:
            T_gain = _ou_process(
                n,
                mean=cfg.t2l_gain,
                sigma=cfg.t2l_gain_drift_sigma,
                timescale=cfg.t2l_thermal_timescale,
                fs=cfg.fs,
                rng=self.rng,
            )
            T_gain = np.clip(T_gain, 0.5, cfg.t2l_gain * 2.5)
        else:
            # Stationary: constant coupling gain (linear filter can cancel)
            T_gain = np.full(n, cfg.t2l_gain)

        # --- tilt-horizontal coupling: C_tilt = T(t) · θ_y_proxy(t) ---
        return T_gain * tilt_proxy
