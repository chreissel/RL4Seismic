"""
Sweep the evaluation episode duration (default settings otherwise).

Same config as the classical benchmark:
  - tilt coupling (double leaky integrator)
  - brownian sensor noise, band-pinned to [0.05, 0.5] Hz
  - default OU drift (sigmas and τ unchanged)
  - non-ML baselines only (Wiener and IIR)

As the episode lengthens, the OU process explores more of its stationary
distribution → the within-episode variance of T·θ_y grows toward its
long-run value, which should degrade Wiener's steady-state floor (it
fits a single session-averaged gain).  IIR, being online, should be
roughly independent of duration.
"""

from __future__ import annotations

from noise_removal import SeismicConfig, SeismicSignalSimulator
from evaluate import run_wiener, run_iir, band_rms


EVAL_BAND = (0.05, 0.5)
SEED = 7
WARMUP = 0.25
TRAIN_MULT = 3.0  # training segment = 3 × eval duration


def run_one(duration: float) -> dict:
    cfg = SeismicConfig(
        drift=True,
        tilt_coupling=True,
        sensor_noise_exponent=2.0,
        sensor_noise_band=EVAL_BAND,
    )

    sim = SeismicSignalSimulator(cfg, seed=SEED)
    data = sim.generate_episode(duration=duration, signal_amplitude=0.0)

    train_sim = SeismicSignalSimulator(cfg, seed=SEED + 1000)
    train_data = train_sim.generate_episode(
        duration=duration * TRAIN_MULT, signal_amplitude=0.0
    )

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

    return {
        "duration": duration,
        "n_samples": n,
        "oracle": oracle,
        "tilt_x_oracle": brms(data["coupling_tilt"]) / oracle,
        "raw_x_oracle": brms(data["main"]) / oracle,
        "wiener_full_x_oracle": brms(wiener_clean) / oracle,
        "wiener_ss_x_oracle": brms(wiener_clean[ss:]) / oracle_ss,
        "iir_full_x_oracle": brms(iir_clean) / oracle,
        "iir_ss_x_oracle": brms(iir_clean[ss:]) / oracle_ss,
    }


def fmt(r):
    return (
        f"  T={r['duration']:<6.0f}s  N={r['n_samples']:<6d}  "
        f"tilt={r['tilt_x_oracle']:6.1f}x  raw={r['raw_x_oracle']:6.1f}x  "
        f"Wiener(full)={r['wiener_full_x_oracle']:7.1f}x  "
        f"Wiener(ss)={r['wiener_ss_x_oracle']:6.1f}x  "
        f"IIR(full)={r['iir_full_x_oracle']:6.1f}x  "
        f"IIR(ss)={r['iir_ss_x_oracle']:6.1f}x"
    )


def main():
    print("=" * 130)
    print("  Wiener / IIR vs eval episode duration (default drift σ, τ=600 s, "
          "tilt coupling, brownian noise, band-RMS [0.05, 0.5] Hz)")
    print("=" * 130)
    print(f"  Training segment = {TRAIN_MULT:.0f} × eval duration, warmup = {WARMUP:.0%}")
    print("-" * 130)

    for duration in [60.0, 150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0]:
        r = run_one(duration)
        print(fmt(r))

    print()


if __name__ == "__main__":
    main()
