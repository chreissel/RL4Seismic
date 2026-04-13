"""
Lightning plots illustrating the current RL4Seismic toy setup.

Produces three PNGs under ``results/``:

  1. toy_setup_timedomain.png  — 60 s excerpt of witnesses, coupling terms,
     sensor noise and main channel.
  2. toy_setup_tilt_proxy.png  — PSD of the tilt-horizontal coupling for the
     new double-leaky-integrator proxy vs the old finite-difference proxy.
  3. toy_setup_sensor_colors.png — PSD of the sensor (oracle) noise for
     white / pink / brown spectra, shown both with the default broadband
     RMS normalisation and with an in-band [0.05, 0.5] Hz normalisation.

Run standalone (no gymnasium required):

    python scripts/plot_toy_setup.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Import signals.py directly so we don't pull in the env / gymnasium stack
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "noise_removal"))
import signals as sg  # noqa: E402


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Plot 1 — time-domain anatomy of one episode
# ---------------------------------------------------------------------------
def plot_timedomain():
    cfg = sg.SeismicConfig()  # defaults: drift=True, tilt_coupling=True, white sensor
    sim = sg.SeismicSignalSimulator(cfg, seed=0)
    d = sim.generate_episode(duration=300.0)

    t = d["time"]
    # The FIR coupling needs cfg.filter_length samples (60 s @ 4 Hz) to warm
    # up, so plot the second 60 s window instead of the first.
    mask = (t >= 60.0) & (t < 120.0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True)

    axes[0].plot(t[mask], d["witness_x"][mask], lw=0.8, color="tab:blue",
                 label="witness_x (inline)")
    axes[0].plot(t[mask], d["witness_y"][mask], lw=0.8, color="tab:cyan",
                 label="witness_y (perpendicular)")
    axes[0].set_ylabel("Ground motion")
    axes[0].set_title("Witnesses (two seismometers, coloured 0.05–0.5 Hz)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    linear = d["coupling"] - d["coupling_tilt"]
    lin_rms = float(np.sqrt(np.mean(linear**2)))
    tilt_rms = float(np.sqrt(np.mean(d["coupling_tilt"]**2)))

    # Twin y-axes so the linear term (RMS ~1.8) is not visually crushed by the
    # tilt term (RMS ~10) — both terms share the same time base but get their
    # own amplitude scale, coloured to match their trace.
    ax_lin = axes[1]
    ax_tilt = ax_lin.twinx()

    l1, = ax_lin.plot(t[mask], linear[mask], lw=0.9, color="tab:orange",
                      label=f"h_x(t) ⊛ w_x  (linear FIR, OU drift)  RMS={lin_rms:.2f}")
    l2, = ax_tilt.plot(t[mask], d["coupling_tilt"][mask], lw=0.9, color="tab:red",
                       label=f"C_tilt = T(t) · θ_y_proxy(t)  (bilinear)  RMS={tilt_rms:.2f}")

    # Symmetric limits sized to each term's own peak excursion (with 15 % pad)
    lin_lim = 1.15 * np.max(np.abs(linear[mask]))
    tilt_lim = 1.15 * np.max(np.abs(d["coupling_tilt"][mask]))
    ax_lin.set_ylim(-lin_lim, lin_lim)
    ax_tilt.set_ylim(-tilt_lim, tilt_lim)

    ax_lin.set_ylabel("Linear coupling", color="tab:orange")
    ax_tilt.set_ylabel("Tilt coupling", color="tab:red")
    ax_lin.tick_params(axis="y", labelcolor="tab:orange")
    ax_tilt.tick_params(axis="y", labelcolor="tab:red")
    ax_lin.axhline(0, color="grey", lw=0.5, alpha=0.5)

    ax_lin.set_title(
        f"Coupling anatomy — linear (OU-drifting) + tilt (bilinear)   "
        f"[tilt/linear RMS ≈ {tilt_rms/lin_rms:.1f}×, separate y-axes]"
    )
    ax_lin.legend([l1, l2], [l1.get_label(), l2.get_label()],
                  loc="upper right", fontsize=8)
    ax_lin.grid(alpha=0.3)

    axes[2].plot(t[mask], d["main"][mask], lw=0.8, color="black",
                 label="main y(t)")
    axes[2].plot(t[mask], d["sensor_noise"][mask], lw=0.8, color="tab:green",
                 alpha=0.9, label="sensor noise (oracle floor)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Obtaining channel")
    axes[2].set_title("y(t) = h_x(t)⊛w_x(t) + C_tilt(t) + n_GS13X(t)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("Toy setup: one episode, 60–120 s (after FIR warm-up)",
                 y=1.00, fontsize=11)
    fig.tight_layout()
    _save(fig, "toy_setup_timedomain.png")


# ---------------------------------------------------------------------------
# Plot 2 — tilt proxy: double leaky integrator vs finite difference
# ---------------------------------------------------------------------------
def plot_tilt_proxy():
    cfg = sg.SeismicConfig()
    sim = sg.SeismicSignalSimulator(cfg, seed=1)
    d = sim.generate_episode(duration=1000.0)
    fs = cfg.fs
    wy = d["witness_y"]

    # New proxy: double leaky integrator (what the simulator now uses)
    proxy_new = sg._double_leaky_integrator(wy, cfg.tilt_leak_timescale, fs)
    # Old proxy: finite-difference (what the simulator used before)
    proxy_old = np.empty_like(wy)
    proxy_old[0] = 0.0
    proxy_old[1:] = wy[1:] - wy[:-1]

    # Normalise to unit RMS for a clean shape-only comparison
    proxy_new_n = proxy_new / (np.sqrt(np.mean(proxy_new**2)) + 1e-12)
    proxy_old_n = proxy_old / (np.sqrt(np.mean(proxy_old**2)) + 1e-12)
    wy_n = wy / (np.sqrt(np.mean(wy**2)) + 1e-12)

    nperseg = min(1024, len(wy))
    f, Pwy = welch(wy_n, fs=fs, nperseg=nperseg)
    _, Pnew = welch(proxy_new_n, fs=fs, nperseg=nperseg)
    _, Pold = welch(proxy_old_n, fs=fs, nperseg=nperseg)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
    ax.loglog(f[1:], Pwy[1:], color="tab:cyan", lw=1.1, alpha=0.7,
              label="witness_y  (reference)")
    ax.loglog(f[1:], Pold[1:], color="tab:red", lw=1.4, ls="--",
              label="old proxy: w_y[t] − w_y[t−1]  (∝ ω², high-freq blow-up)")
    ax.loglog(f[1:], Pnew[1:], color="tab:green", lw=1.6,
              label=f"new proxy: double leaky integrator  (τ={cfg.tilt_leak_timescale}s)")
    ax.axvspan(0.05, 0.5, color="grey", alpha=0.12,
               label="microseismic band  [0.05, 0.5] Hz")
    ax.axvline(1.0 / (2 * np.pi * cfg.tilt_leak_timescale), color="tab:green",
               lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD  (unit-RMS normalised)")
    ax.set_title("Tilt proxy PSD — old (finite-diff) vs new (double leaky integrator)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(which="both", alpha=0.3)
    _save(fig, "toy_setup_tilt_proxy.png")


# ---------------------------------------------------------------------------
# Plot 3 — sensor noise colours and broadband vs band-limited normalisation
# ---------------------------------------------------------------------------
def plot_sensor_colors():
    fs = 4.0
    n = 8000
    rng = np.random.default_rng(42)
    band = (0.05, 0.5)
    sigma = 0.05
    colours = [("white", 0.0, "tab:gray"),
               ("pink",  1.0, "tab:orange"),
               ("brown", 2.0, "tab:red")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    nperseg = 1024
    for label, alpha, color in colours:
        # Broadband normalisation (default)
        x_bb = sg._colored_noise(n, sigma=sigma, exponent=alpha, fs=fs, rng=rng,
                                 band=None)
        # Band-limited normalisation on [0.05, 0.5] Hz
        x_bl = sg._colored_noise(n, sigma=sigma, exponent=alpha, fs=fs, rng=rng,
                                 band=band)

        f, P_bb = welch(x_bb, fs=fs, nperseg=nperseg)
        _, P_bl = welch(x_bl, fs=fs, nperseg=nperseg)

        axes[0].loglog(f[1:], P_bb[1:], color=color, lw=1.4,
                       label=f"{label}  (α={alpha:g})")
        axes[1].loglog(f[1:], P_bl[1:], color=color, lw=1.4,
                       label=f"{label}  (α={alpha:g})")

    for ax, title in zip(
        axes,
        [f"Broadband RMS = {sigma}\n(--sensor-noise-band unset)",
         f"In-band RMS on {band} Hz = {sigma}\n(--sensor-noise-band 0.05 0.5)"],
    ):
        ax.axvspan(*band, color="grey", alpha=0.12)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(title, fontsize=10)
        ax.grid(which="both", alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("PSD")
    fig.suptitle("Sensor (oracle) noise PSD — spectral colour × RMS normalisation",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, "toy_setup_sensor_colors.png")


if __name__ == "__main__":
    print("Writing diagnostic plots to", OUT_DIR)
    plot_timedomain()
    plot_tilt_proxy()
    plot_sensor_colors()
    print("Done.")
