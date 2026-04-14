"""
Sweep OU drift magnitude and measure Wiener-filter performance.

Uses the same config as the classical benchmark:
  - tilt coupling (double leaky integrator)
  - brownian sensor noise, band-pinned to [0.05, 0.5] Hz
  - non-ML baselines only (Wiener and IIR)

Varies the OU drift of T(t) (and the linear FIR coupling parameters) by
scaling the *_drift_sigma fields and/or shortening the thermal timescale.
This lets us see how Wiener's steady-state floor degrades as the system
becomes more non-stationary — the regime where online methods (IIR, RL)
begin to matter.
"""

from __future__ import annotations

import numpy as np

from noise_removal import SeismicConfig, SeismicSignalSimulator
from evaluate import run_wiener, run_iir, band_rms


EVAL_BAND = (0.05, 0.5)
DURATION = 300.0
TRAIN_DURATION = 900.0
SEED = 7
WARMUP = 0.25


def run_one(drift_scale: float, timescale: float) -> dict:
    cfg = SeismicConfig(
        drift=True,
        tilt_coupling=True,
        sensor_noise_exponent=2.0,             # brownian
        sensor_noise_band=EVAL_BAND,
        gain_drift_sigma=0.1 * drift_scale,
        freq_drift_sigma=0.01 * drift_scale,
        t2l_gain_drift_sigma=2.7 * drift_scale,
        thermal_timescale=timescale,
        t2l_thermal_timescale=timescale,
    )

    sim = SeismicSignalSimulator(cfg, seed=SEED)
    data = sim.generate_episode(duration=DURATION, signal_amplitude=0.0)

    train_sim = SeismicSignalSimulator(cfg, seed=SEED + 1000)
    train_data = train_sim.generate_episode(duration=TRAIN_DURATION, signal_amplitude=0.0)

    wiener_clean = run_wiener(train_data, data,
                              filter_length=cfg.filter_length, use_witness_y=True)
    iir_clean = run_iir(data,
                        feedforward_length=cfg.filter_length, use_witness_y=True)

    n = len(data["main"])
    ss = int(n * WARMUP)

    fs = cfg.fs
    def brms(x):
        return band_rms(x, fs, EVAL_BAND)

    oracle = brms(data["sensor_noise"])
    oracle_ss = brms(data["sensor_noise"][ss:])
    tilt = brms(data["coupling_tilt"])
    raw = brms(data["main"])
    wiener_full = brms(wiener_clean)
    wiener_ss_ = brms(wiener_clean[ss:])
    iir_full = brms(iir_clean)
    iir_ss_ = brms(iir_clean[ss:])

    return {
        "drift_scale": drift_scale,
        "timescale": timescale,
        "oracle": oracle,
        "tilt_rms_x_oracle": tilt / oracle,
        "raw_x_oracle": raw / oracle,
        "wiener_full_x_oracle": wiener_full / oracle,
        "wiener_ss_x_oracle": wiener_ss_ / oracle_ss,
        "iir_full_x_oracle": iir_full / oracle,
        "iir_ss_x_oracle": iir_ss_ / oracle_ss,
    }


def fmt(r):
    return (f"  scale={r['drift_scale']:<5.2f}  τ={r['timescale']:<6.0f}s  "
            f"tilt={r['tilt_rms_x_oracle']:6.1f}x  raw={r['raw_x_oracle']:6.1f}x  "
            f"Wiener(full)={r['wiener_full_x_oracle']:6.1f}x  "
            f"Wiener(ss)={r['wiener_ss_x_oracle']:6.1f}x  "
            f"IIR(ss)={r['iir_ss_x_oracle']:6.1f}x")


def main():
    print("=" * 110)
    print("  Wiener / IIR vs OU drift magnitude  (tilt coupling, brownian noise, band-RMS [0.05, 0.5] Hz)")
    print("=" * 110)
    print(f"  Baseline config: gain_drift_sigma=0.1, freq_drift_sigma=0.01, "
          f"t2l_gain_drift_sigma=2.7, τ=600s")
    print(f"  Eval: {DURATION:.0f} s, train: {TRAIN_DURATION:.0f} s, warmup: {WARMUP:.0%}")
    print("-" * 110)

    print("\n  Sweep A: scale ALL drift sigmas (τ held at 600 s)")
    results_a = []
    for scale in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        r = run_one(drift_scale=scale, timescale=600.0)
        results_a.append(r)
        print(fmt(r))

    print("\n  Sweep B: shorten thermal timescale τ (sigma scale = 1.0)")
    results_b = []
    for tau in [6000.0, 1200.0, 600.0, 300.0, 120.0, 60.0]:
        r = run_one(drift_scale=1.0, timescale=tau)
        results_b.append(r)
        print(fmt(r))

    print()


if __name__ == "__main__":
    main()
