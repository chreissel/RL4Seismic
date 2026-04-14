"""
Longer-T sweep at drift_scale = 2 — does NLMS keep improving?

Same config as scripts/classical_filters_long_drift.py but extends the
evaluation duration well beyond 7200 s to check whether NLMS continues
to tighten toward oracle or plateaus at some drift-limited floor.
"""

from __future__ import annotations

import time

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


def run_one(duration: float, train_mult: float = 2.0) -> dict:
    cfg = SeismicConfig(
        drift=True,
        tilt_coupling=True,
        sensor_noise_exponent=2.0,
        sensor_noise_band=EVAL_BAND,
        gain_drift_sigma=0.1 * DRIFT_SCALE,
        freq_drift_sigma=0.01 * DRIFT_SCALE,
        t2l_gain_drift_sigma=2.7 * DRIFT_SCALE,
    )

    t0 = time.time()
    sim = SeismicSignalSimulator(cfg, seed=SEED)
    data = sim.generate_episode(duration=duration, signal_amplitude=0.0)

    train_sim = SeismicSignalSimulator(cfg, seed=SEED + 1000)
    train_data = train_sim.generate_episode(
        duration=duration * train_mult, signal_amplitude=0.0
    )

    wiener_clean = run_wiener(train_data, data,
                              filter_length=cfg.filter_length, use_witness_y=True)
    iir_clean = run_iir(data,
                        feedforward_length=cfg.filter_length, use_witness_y=True)
    nlms_clean = run_nlms(data,
                          feedforward_length=cfg.filter_length, use_witness_y=True)
    elapsed = time.time() - t0

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
        "elapsed": elapsed,
        "oracle": oracle,
        "tilt_x_oracle": brms(data["coupling_tilt"]) / oracle,
        "raw_x_oracle": brms(data["main"]) / oracle,
        "wiener_ss": brms(wiener_clean[ss:]) / oracle_ss,
        "iir_ss": brms(iir_clean[ss:]) / oracle_ss,
        "nlms_ss": brms(nlms_clean[ss:]) / oracle_ss,
    }


def main():
    print("=" * 110)
    print(f"  Non-ML filters at drift_scale = {DRIFT_SCALE}, very long episodes")
    print("  (tilt coupling, brownian noise, band-RMS [0.05, 0.5] Hz, steady-state only)")
    print("=" * 110)

    header = (f"  {'T (s)':<8}{'N':<9}{'tilt':>8}{'raw':>8}"
              f"{'Wnr ss':>10}{'IIR ss':>10}{'NLMS ss':>10}{'wall (s)':>12}")
    print(header)
    print("-" * 110)

    for duration in [7200.0, 14400.0, 28800.0, 57600.0, 115200.0]:
        r = run_one(duration)
        row = (f"  {r['duration']:<8.0f}{r['n_samples']:<9d}"
               f"{r['tilt_x_oracle']:>8.1f}{r['raw_x_oracle']:>8.1f}"
               f"{r['wiener_ss']:>10.1f}{r['iir_ss']:>10.1f}{r['nlms_ss']:>10.1f}"
               f"{r['elapsed']:>12.1f}")
        print(row, flush=True)

    print()


if __name__ == "__main__":
    main()
