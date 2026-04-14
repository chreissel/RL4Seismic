"""
All non-ML filters at drift_scale=2, long evaluation episodes.

Config:
  - tilt coupling (double leaky integrator)
  - brownian sensor noise, band-pinned to [0.05, 0.5] Hz
  - OU drift sigmas scaled ×2, τ unchanged (600 s)
  - Wiener, IIR, and NLMS (all non-ML baselines)

Tests durations > 1200 s to see how each filter behaves once the OU drift
has time to explore a significant portion of its stationary distribution.
"""

from __future__ import annotations

import numpy as np

from noise_removal import SeismicConfig, SeismicSignalSimulator
from baselines import LMSFilter
from evaluate import run_wiener, run_iir, band_rms


EVAL_BAND = (0.05, 0.5)
SEED = 7
WARMUP = 0.25
DRIFT_SCALE = 2.0


def run_nlms(data, feedforward_length, step_size=0.5, use_witness_y=True):
    filt = LMSFilter(
        filter_length=feedforward_length,
        step_size=step_size,
        normalized=True,
    )
    wy = data["witness_y"] if use_witness_y else None
    return filt.run(data["witness_x"], data["main"], witness_y=wy)


def run_one(duration: float) -> dict:
    cfg = SeismicConfig(
        drift=True,
        tilt_coupling=True,
        sensor_noise_exponent=2.0,
        sensor_noise_band=EVAL_BAND,
        gain_drift_sigma=0.1 * DRIFT_SCALE,
        freq_drift_sigma=0.01 * DRIFT_SCALE,
        t2l_gain_drift_sigma=2.7 * DRIFT_SCALE,
    )

    sim = SeismicSignalSimulator(cfg, seed=SEED)
    data = sim.generate_episode(duration=duration, signal_amplitude=0.0)

    train_sim = SeismicSignalSimulator(cfg, seed=SEED + 1000)
    train_data = train_sim.generate_episode(duration=duration * 3.0, signal_amplitude=0.0)

    wiener_clean = run_wiener(train_data, data,
                              filter_length=cfg.filter_length, use_witness_y=True)
    iir_clean = run_iir(data,
                        feedforward_length=cfg.filter_length, use_witness_y=True)
    nlms_clean = run_nlms(data,
                          feedforward_length=cfg.filter_length, use_witness_y=True)

    n = len(data["main"])
    ss = int(n * WARMUP)
    fs = cfg.fs

    def brms(x):
        return band_rms(x, fs, EVAL_BAND)

    oracle = brms(data["sensor_noise"])
    oracle_ss = brms(data["sensor_noise"][ss:])

    return {
        "duration": duration,
        "n_samples": n,
        "oracle": oracle,
        "tilt_x_oracle": brms(data["coupling_tilt"]) / oracle,
        "raw_x_oracle": brms(data["main"]) / oracle,
        "wiener_full": brms(wiener_clean) / oracle,
        "wiener_ss": brms(wiener_clean[ss:]) / oracle_ss,
        "iir_full": brms(iir_clean) / oracle,
        "iir_ss": brms(iir_clean[ss:]) / oracle_ss,
        "nlms_full": brms(nlms_clean) / oracle,
        "nlms_ss": brms(nlms_clean[ss:]) / oracle_ss,
    }


def main():
    print("=" * 130)
    print(f"  Non-ML filters at drift_scale = {DRIFT_SCALE}, long episodes  "
          "(tilt coupling, brownian noise, band-RMS [0.05, 0.5] Hz)")
    print("=" * 130)
    print(f"  σ_g=0.2, σ_f=0.02, σ_T=5.4, τ=600 s  (all numbers × oracle)")
    print("-" * 130)

    header = (f"  {'T (s)':<7}{'N':<7}{'tilt':>8}{'raw':>8}"
              f"{'Wnr full':>10}{'Wnr ss':>10}"
              f"{'IIR full':>10}{'IIR ss':>10}"
              f"{'NLMS full':>11}{'NLMS ss':>10}")
    print(header)
    print("-" * 130)

    for duration in [1500.0, 2400.0, 3600.0, 4800.0, 7200.0]:
        r = run_one(duration)
        row = (f"  {r['duration']:<7.0f}{r['n_samples']:<7d}"
               f"{r['tilt_x_oracle']:>8.1f}{r['raw_x_oracle']:>8.1f}"
               f"{r['wiener_full']:>10.1f}{r['wiener_ss']:>10.1f}"
               f"{r['iir_full']:>10.1f}{r['iir_ss']:>10.1f}"
               f"{r['nlms_full']:>11.1f}{r['nlms_ss']:>10.1f}")
        print(row)

    print()


if __name__ == "__main__":
    main()
