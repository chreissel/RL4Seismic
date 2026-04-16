"""
Evaluate a trained MPO agent on the seismic noise-cancellation task.

Runs the MPO agent alongside classical baselines (NLMS, Wiener, IIR,
Volterra, EKF) on a hard evaluation episode and produces comparison plots.

Usage
-----
    # Evaluate a trained MPO model
    PYTHONPATH=. python scripts/evaluate_mpo.py \\
        --model-path models/mpo_bilinear

    # Evaluate an intermediate checkpoint
    PYTHONPATH=. python scripts/evaluate_mpo.py \\
        --model-path models/mpo_bilinear_10000_steps

    # Baselines only (no MPO)
    PYTHONPATH=. python scripts/evaluate_mpo.py --no-rl

    # Custom episode
    PYTHONPATH=. python scripts/evaluate_mpo.py \\
        --model-path models/mpo_bilinear --duration 3600 --drift-scale 4

Produces
--------
    results/mpo_eval/mpo_eval.png  — 4-panel comparison figure
    Console table of in-band RMS (× oracle) for all methods
"""

from __future__ import annotations

import argparse
import os
import re as _re
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch

from noise_removal import (
    NoiseCancellationEnv,
    SeismicConfig,
    SeismicSignalSimulator,
)
from noise_removal.mpo import MPO
from noise_removal.bilinear_policy import BilinearDilatedExtractor
from baselines import (
    EKFFilter,
    IIRFilter,
    LMSFilter,
    VolterraFilter,
    WienerFilter,
)
from evaluate import band_rms


EVAL_BAND = (0.05, 0.5)


# ---------------------------------------------------------------------------
# Filter runners
# ---------------------------------------------------------------------------

def run_wiener(train_data, eval_data, filter_length):
    filt = WienerFilter(filter_length=filter_length)
    filt.fit(train_data["witness_x"], train_data["main"],
             witness_y=train_data["witness_y"])
    return filt.run(eval_data["witness_x"], eval_data["main"],
                    witness_y=eval_data["witness_y"])


def run_iir(data, filter_length):
    filt = IIRFilter(
        feedforward_length=filter_length, feedback_length=0,
        step_size=0.1, normalized=True,
    )
    return filt.run(data["witness_x"], data["main"],
                    witness_y=data["witness_y"])


def run_nlms(data, filter_length):
    filt = LMSFilter(
        filter_length=filter_length, step_size=0.5, normalized=True,
    )
    return filt.run(data["witness_x"], data["main"],
                    witness_y=data["witness_y"])


def run_volterra(train_data, eval_data, filter_length):
    filt = VolterraFilter(
        filter_length=filter_length, bilinear_length=20,
        bilinear_stride=12, regularization=1e-2,
    )
    filt.fit(train_data["witness_x"], train_data["main"],
             witness_y=train_data["witness_y"])
    return filt.run(eval_data["witness_x"], eval_data["main"],
                    witness_y=eval_data["witness_y"])


def run_ekf(data, cfg):
    filt = EKFFilter(
        fs=cfg.fs, tilt_leak_timescale=cfg.tilt_leak_timescale,
        t2l_prior=cfg.t2l_gain,
        t2l_thermal_timescale=cfg.t2l_thermal_timescale,
        t2l_drift_sigma=cfg.t2l_gain_drift_sigma,
        measurement_sigma=cfg.sensor_noise_sigma,
    )
    return filt.run(data["witness_x"], data["main"],
                    witness_y=data["witness_y"])


# ---------------------------------------------------------------------------
# MPO runner
# ---------------------------------------------------------------------------

def _resolve_vecnorm_path(model_path: str) -> str:
    candidates = [model_path + "_vecnorm.pkl"]
    m = _re.search(r"_(\d+)_steps$", model_path)
    if m:
        steps = m.group(1)
        prefix = model_path[: model_path.rindex(f"_{steps}_steps")]
        candidates.append(f"{prefix}_vecnormalize_{steps}_steps.pkl")
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No VecNormalize file found for '{model_path}'. Tried: {candidates}"
    )


def run_mpo_agent(
    data: dict,
    cfg: SeismicConfig,
    model_path: str,
    window_size: int = 240,
    conv_channels: int = 24,
    conv_layers: int = 6,
    bilinear_rank: int = 16,
    features_dim: int = 192,
    action_clip: float = 15.0,
) -> np.ndarray:
    """Roll out a trained MPO agent sample-by-sample on pre-generated data."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vecnorm_path = _resolve_vecnorm_path(model_path)

    # Build a dummy env to get observation/action spaces and load VecNormalize
    _cfg = cfg
    _ws = window_size
    _ac = action_clip
    dummy = DummyVecEnv([
        lambda: NoiseCancellationEnv(config=_cfg, window_size=_ws, action_clip=_ac)
    ])
    dummy_norm = VecNormalize.load(vecnorm_path, dummy)
    dummy_norm.training = False
    dummy_norm.norm_reward = False

    # Reconstruct MPO with the same architecture and load weights
    model = MPO(
        observation_space=dummy_norm.observation_space,
        action_space=dummy.action_space,
        features_extractor_class=BilinearDilatedExtractor,
        features_extractor_kwargs=dict(
            window_size=window_size,
            conv_channels=conv_channels,
            n_layers=conv_layers,
            bilinear_rank=bilinear_rank,
            features_dim=features_dim,
        ),
        device="cpu",
    )
    model.load(model_path)

    # Set up env with pre-generated data
    env = NoiseCancellationEnv(config=cfg, window_size=window_size, action_clip=action_clip)
    env._data = data
    env._n_samples = len(data["time"])
    env._step_idx = window_size
    env._action_history = np.zeros(len(data["time"]), dtype=np.float64)

    n = len(data["time"])
    cleaned = data["main"].copy()
    obs = env._get_obs()

    for t in range(window_size, n):
        obs_norm = dummy_norm.normalize_obs(obs[np.newaxis, :])
        action = model.predict(obs_norm.squeeze(0), deterministic=True)
        a = float(np.clip(action[0], -action_clip, action_clip))
        cleaned[t] = data["main"][t] - a

        env._action_history[t] = a
        env._step_idx = t + 1
        if t + 1 < n:
            obs = env._get_obs()

    return cleaned


# ---------------------------------------------------------------------------
# Metrics and plotting
# ---------------------------------------------------------------------------

def compute_metrics(data, cleaned_by_method, fs, warmup=0.25):
    N = len(data["main"])
    ss = int(N * warmup)
    oracle_ss = band_rms(data["sensor_noise"][ss:], fs, EVAL_BAND)

    rows: Dict[str, Dict[str, float]] = {}
    rows["Oracle"] = {
        "ss_in_band": oracle_ss,
        "ss_x_oracle": 1.0,
    }
    rows["Raw"] = {
        "ss_in_band": band_rms(data["main"][ss:], fs, EVAL_BAND),
        "ss_x_oracle": band_rms(data["main"][ss:], fs, EVAL_BAND) / oracle_ss,
    }
    for name, clean in cleaned_by_method.items():
        rows[name] = {
            "ss_in_band": band_rms(clean[ss:], fs, EVAL_BAND),
            "ss_x_oracle": band_rms(clean[ss:], fs, EVAL_BAND) / oracle_ss,
        }
    return rows


def find_regime_change(data, min_offset=600):
    regime = data.get("regime")
    if regime is None:
        return None
    d = np.diff(regime.astype(int))
    jumps = np.where(d != 0)[0] + 1
    jumps = jumps[jumps >= min_offset]
    return int(jumps[0]) if len(jumps) else None


def rolling_rms(x, win):
    x2 = x.astype(np.float64) ** 2
    out = np.zeros_like(x2)
    csum = np.cumsum(x2)
    for i in range(len(x2)):
        lo = max(0, i - win + 1)
        s = csum[i] - (csum[lo - 1] if lo > 0 else 0.0)
        out[i] = np.sqrt(s / (i - lo + 1))
    return out


def plot_eval(data, cleaned_by_method, metrics, fs, save_dir, regime_t=None):
    os.makedirs(save_dir, exist_ok=True)

    method_order = [
        "Raw", "Wiener", "IIR", "NLMS", "Volterra", "EKF", "MPO", "Oracle"
    ]
    colors = {
        "Raw": "grey", "Wiener": "tab:orange", "IIR": "tab:purple",
        "NLMS": "tab:brown", "Volterra": "tab:pink", "EKF": "tab:cyan",
        "MPO": "tab:blue", "Oracle": "tab:red",
    }
    available = [m for m in method_order if m in metrics]

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        "MPO + Deep Loop Shaping vs classical filters",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # Bar chart
    ax_bar = fig.add_subplot(gs[0, :])
    vals = [metrics[m]["ss_x_oracle"] for m in available]
    bars = ax_bar.bar(
        available, vals, color=[colors.get(m, "k") for m in available],
        edgecolor="black", lw=0.7, alpha=0.85,
    )
    ax_bar.set_yscale("log")
    ax_bar.set_ylabel("Steady-state in-band RMS (× oracle)")
    ax_bar.set_title(
        f"Band-limited RMS on {EVAL_BAND[0]}–{EVAL_BAND[1]} Hz "
        f"(steady-state, > 25% warmup)"
    )
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, v * 1.05,
                    f"{v:.1f}×", ha="center", va="bottom", fontsize=10)
    ax_bar.grid(alpha=0.3, axis="y", which="both")

    # Post-regime-change transient
    ax_tr = fig.add_subplot(gs[1, :])
    if regime_t is not None:
        win_samples = int(5.0 * fs)
        span = int(120.0 * fs)
        t_arr = data["time"]
        lo = max(0, regime_t - span)
        hi = min(len(t_arr), regime_t + span)
        for name, clean in cleaned_by_method.items():
            r = rolling_rms(clean, win_samples)[lo:hi]
            ax_tr.plot(t_arr[lo:hi] - t_arr[regime_t], r,
                       label=name, color=colors.get(name, "k"), lw=1.1)
        ax_tr.axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.6,
                      label="regime change")
        oracle_r = rolling_rms(data["sensor_noise"], win_samples)[lo:hi]
        ax_tr.plot(t_arr[lo:hi] - t_arr[regime_t], oracle_r,
                   color="tab:red", ls="--", lw=0.9, alpha=0.7, label="Oracle")
        ax_tr.set_yscale("log")
        ax_tr.set_xlabel("Time relative to regime change (s)")
        ax_tr.set_ylabel("Rolling RMS (5 s window)")
        ax_tr.set_title("Post-regime-change adaptation transient")
        ax_tr.legend(fontsize=8, ncol=4)
        ax_tr.grid(alpha=0.3, which="both")
    else:
        ax_tr.text(0.5, 0.5, "No regime-change event in this episode",
                   ha="center", va="center", transform=ax_tr.transAxes)
        ax_tr.set_axis_off()

    # Rolling RMS (full episode)
    ax_roll = fig.add_subplot(gs[2, 0])
    t_arr = data["time"]
    full_win = int(30.0 * fs)
    for name, clean in cleaned_by_method.items():
        r = rolling_rms(clean, full_win)
        ax_roll.plot(t_arr, r, label=name, color=colors.get(name, "k"),
                     lw=0.9, alpha=0.85)
    ax_roll.plot(t_arr, rolling_rms(data["sensor_noise"], full_win),
                 color="tab:red", ls="--", lw=0.9, alpha=0.7, label="Oracle")
    ax_roll.set_yscale("log")
    ax_roll.set_xlabel("Time (s)")
    ax_roll.set_ylabel("Rolling RMS (30 s window)")
    ax_roll.set_title("Rolling RMS over full episode")
    ax_roll.legend(fontsize=7, ncol=2)
    ax_roll.grid(alpha=0.3, which="both")

    # PSD
    ax_psd = fig.add_subplot(gs[2, 1])
    nperseg = min(1024, len(data["main"]))
    f_psd, pxx_raw = welch(data["main"], fs=fs, nperseg=nperseg)
    ax_psd.semilogy(f_psd, pxx_raw, color="grey", lw=1.0, alpha=0.7, label="Raw")
    for name, clean in cleaned_by_method.items():
        f_psd, pxx = welch(clean, fs=fs, nperseg=nperseg)
        ax_psd.semilogy(f_psd, pxx, label=name, color=colors.get(name, "k"),
                        lw=1.0, alpha=0.85)
    f_psd, pxx_o = welch(data["sensor_noise"], fs=fs, nperseg=nperseg)
    ax_psd.semilogy(f_psd, pxx_o, color="tab:red", ls="--", lw=1.0, alpha=0.8,
                    label="Oracle")
    ax_psd.axvspan(EVAL_BAND[0], EVAL_BAND[1], color="yellow", alpha=0.1,
                   label="eval band")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD")
    ax_psd.set_title("Power Spectral Density")
    ax_psd.legend(fontsize=7, ncol=2)
    ax_psd.grid(alpha=0.3, which="both")
    ax_psd.set_xlim(0, fs / 2)

    out = os.path.join(save_dir, "mpo_eval.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained MPO agent")
    p.add_argument("--model-path", default="models/mpo_bilinear",
                   help="MPO model path prefix (without .zip)")
    p.add_argument("--no-rl", action="store_true",
                   help="Skip MPO agent (baselines only)")

    # Episode config
    p.add_argument("--duration", type=float, default=1800.0)
    p.add_argument("--train-duration-mult", type=float, default=2.0)
    p.add_argument("--drift-scale", type=float, default=2.0)
    p.add_argument("--regime-changes", action="store_true", default=True)
    p.add_argument("--no-regime-changes", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--window-size", type=int, default=240)

    # MPO architecture (must match training config)
    p.add_argument("--conv-channels", type=int, default=24)
    p.add_argument("--conv-layers", type=int, default=6)
    p.add_argument("--bilinear-rank", type=int, default=16)
    p.add_argument("--features-dim", type=int, default=192)
    p.add_argument("--action-clip", type=float, default=15.0)

    p.add_argument("--save-dir", default="results/mpo_eval")
    return p.parse_args()


def main():
    args = parse_args()
    regime_changes = args.regime_changes and not args.no_regime_changes

    cfg = SeismicConfig(
        drift=True,
        tilt_coupling=True,
        sensor_noise_exponent=2.0,
        sensor_noise_band=EVAL_BAND,
        gain_drift_sigma=0.1 * args.drift_scale,
        freq_drift_sigma=0.01 * args.drift_scale,
        t2l_gain_drift_sigma=2.7 * args.drift_scale,
        regime_changes=regime_changes,
    )

    print(f"Generating evaluation episode: {args.duration:.0f} s  "
          f"(drift_scale={args.drift_scale}, regime_changes={regime_changes})")
    sim = SeismicSignalSimulator(cfg, seed=args.seed)
    data = sim.generate_episode(duration=args.duration, signal_amplitude=0.0)

    train_duration = args.duration * args.train_duration_mult
    print(f"Generating training segment: {train_duration:.0f} s")
    train_sim = SeismicSignalSimulator(cfg, seed=args.seed + 1000)
    train_data = train_sim.generate_episode(
        duration=train_duration, signal_amplitude=0.0
    )

    cleaned: Dict[str, np.ndarray] = {}

    print("Running Wiener (offline)…")
    cleaned["Wiener"] = run_wiener(train_data, data, filter_length=args.window_size)

    print("Running IIR (online)…")
    cleaned["IIR"] = run_iir(data, filter_length=args.window_size)

    print("Running NLMS (online)…")
    cleaned["NLMS"] = run_nlms(data, filter_length=args.window_size)

    print("Running Volterra (offline nonlinear)…")
    cleaned["Volterra"] = run_volterra(train_data, data, filter_length=args.window_size)

    print("Running EKF (model-based online)…")
    cleaned["EKF"] = run_ekf(data, cfg)

    if not args.no_rl:
        model_zip = args.model_path + ".zip"
        if os.path.exists(model_zip):
            try:
                _resolve_vecnorm_path(args.model_path)
            except FileNotFoundError as e:
                print(f"[warn] {e}  — skipping MPO.")
            else:
                print(f"Running MPO from {model_zip}…")
                cleaned["MPO"] = run_mpo_agent(
                    data, cfg, args.model_path,
                    window_size=args.window_size,
                    conv_channels=args.conv_channels,
                    conv_layers=args.conv_layers,
                    bilinear_rank=args.bilinear_rank,
                    features_dim=args.features_dim,
                    action_clip=args.action_clip,
                )
        else:
            print(f"[warn] MPO model not found at {model_zip}, skipping.")

    metrics = compute_metrics(data, cleaned, fs=cfg.fs)

    print("\n" + "=" * 70)
    print(f"  Steady-state in-band RMS (× oracle) on {EVAL_BAND} Hz")
    print("=" * 70)
    for name, row in metrics.items():
        print(f"  {name:<12s}  {row['ss_x_oracle']:6.2f}×   "
              f"({row['ss_in_band']:.4f})")
    print("=" * 70)

    regime_t = find_regime_change(data, min_offset=args.window_size)
    if regime_t is not None:
        print(f"First regime change at t={data['time'][regime_t]:.1f} s "
              f"(sample {regime_t})")
    plot_eval(data, cleaned, metrics, fs=cfg.fs,
              save_dir=args.save_dir, regime_t=regime_t)


if __name__ == "__main__":
    main()
