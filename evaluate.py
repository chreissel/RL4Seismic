"""
Evaluate a trained PPO agent and compare against baselines.

Usage
-----
    python evaluate.py                            # use saved model
    python evaluate.py --model-path models/ppo_noise_cancellation
    python evaluate.py --no-model                 # skip RL, compare baselines only

Produces
--------
  results/noise_cancellation_overview.png  — time-domain & spectral comparison
  results/coupling_tracking.png            — how well each method tracks f(w,t)
  results/rms_comparison.png               — RMS noise floor bar chart

Metrics reported
----------------
  RMS of the output signal in three conditions:
    - No subtraction     (raw main channel)
    - LMS filter         (linear adaptive baseline)
    - RL agent (PPO)     (this work)
    - Oracle             (perfect coupling subtraction, sensor noise only)
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch

from noise_removal import NoiseCancellationEnv, SignalConfig, SignalSimulator
from baselines import LMSFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def run_lms(data: dict, filter_length: int = 64, step_size: float = 5e-4) -> np.ndarray:
    filt = LMSFilter(filter_length=filter_length, step_size=step_size)
    return filt.run(data["witness"], data["main"])


def run_rl_agent(data: dict, model, vec_norm, window_size: int = 64) -> np.ndarray:
    """Roll out the RL agent sample-by-sample on pre-generated data."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from noise_removal.environment import NoiseCancellationEnv

    n = len(data["time"])
    cleaned = data["main"].copy()

    # Build a single-env wrapper that replays the pre-generated data
    cfg = SignalConfig()
    env = NoiseCancellationEnv(config=cfg, window_size=window_size)

    # Manually set the episode data so we evaluate on the *same* data as LMS
    env._data = data
    env._n_samples = n
    env._step_idx = window_size

    obs, _ = env.reset.__wrapped__(env, seed=None) if hasattr(env.reset, "__wrapped__") else (None, None)
    # Rebuild obs manually from pre-set data
    obs = env._get_obs()

    # Wrap in VecNormalize-compatible format using DummyVecEnv
    dummy = DummyVecEnv([lambda: NoiseCancellationEnv(config=cfg, window_size=window_size)])
    dummy_norm = VecNormalize.load(vec_norm, dummy)
    dummy_norm.training = False
    dummy_norm.norm_reward = False

    for t in range(window_size, n):
        obs_norm = dummy_norm.normalize_obs(obs[np.newaxis, :])
        action, _ = model.predict(obs_norm, deterministic=True)
        a = float(np.clip(action[0][0], -15.0, 15.0))
        cleaned[t] = data["main"][t] - a

        env._step_idx = t + 1
        if t + 1 < n:
            obs = env._get_obs()

    return cleaned


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(
    data: dict,
    lms_clean: np.ndarray,
    rl_clean: Optional[np.ndarray],
    save_dir: str,
    fs: float = 128.0,
):
    t = data["time"]
    oracle_clean = data["sensor_noise"]  # best possible: coupling perfectly removed

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("RL Noise Cancellation — Method Comparison", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # ---- Coupling coefficients over time ----
    ax0 = fig.add_subplot(gs[0, :])
    from noise_removal.signals import TimeVaryingCoupling
    coupling_obj = TimeVaryingCoupling(SignalConfig())
    A, B, C = coupling_obj.coefficients(t)
    ax0.plot(t, A, label="A(t) — linear", color="tab:blue")
    ax0.plot(t, B, label="B(t) — quadratic", color="tab:orange")
    ax0.plot(t, C, label="C(t) — cubic", color="tab:green")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Coupling coefficient")
    ax0.set_title("Time-varying coupling coefficients  f(w,t) = A(t)·w + B(t)·w² + C(t)·w³")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(alpha=0.3)

    # ---- Time-domain: zoom on first 5 s ----
    mask = t < 5.0
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t[mask], data["main"][mask], alpha=0.6, lw=0.8, label="Raw", color="grey")
    ax1.plot(t[mask], lms_clean[mask], alpha=0.8, lw=0.9, label="LMS", color="tab:orange")
    if rl_clean is not None:
        ax1.plot(t[mask], rl_clean[mask], alpha=0.9, lw=0.9, label="RL (PPO)", color="tab:blue")
    ax1.plot(t[mask], oracle_clean[mask], "--", lw=0.8, label="Oracle", color="tab:red", alpha=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time domain (first 5 s)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ---- PSD comparison ----
    ax2 = fig.add_subplot(gs[1, 1])
    nperseg = 512
    for sig, label, color, lw in [
        (data["main"], "Raw",    "grey",      1.0),
        (lms_clean,    "LMS",    "tab:orange", 1.2),
        (oracle_clean, "Oracle", "tab:red",    1.0),
    ]:
        f_psd, pxx = welch(sig, fs=fs, nperseg=nperseg)
        ax2.semilogy(f_psd, pxx, label=label, color=color, lw=lw, alpha=0.8)
    if rl_clean is not None:
        f_psd, pxx = welch(rl_clean, fs=fs, nperseg=nperseg)
        ax2.semilogy(f_psd, pxx, label="RL (PPO)", color="tab:blue", lw=1.3)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD")
    ax2.set_title("Power Spectral Density")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    # ---- RMS over rolling 2-s windows ----
    ax3 = fig.add_subplot(gs[2, 0])
    win = int(2.0 * fs)
    def rolling_rms(x):
        return np.array([
            np.sqrt(np.mean(x[max(0,i-win):i+1]**2))
            for i in range(len(x))
        ])
    ax3.plot(t, rolling_rms(data["main"]), lw=0.8, label="Raw", color="grey", alpha=0.7)
    ax3.plot(t, rolling_rms(lms_clean), lw=0.9, label="LMS", color="tab:orange")
    if rl_clean is not None:
        ax3.plot(t, rolling_rms(rl_clean), lw=0.9, label="RL (PPO)", color="tab:blue")
    ax3.axhline(rms(oracle_clean), ls="--", color="tab:red", lw=0.9, label="Oracle RMS")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Rolling RMS (2 s window)")
    ax3.set_title("Rolling RMS — noise floor over time")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # ---- Summary bar chart ----
    ax4 = fig.add_subplot(gs[2, 1])
    labels = ["Raw", "LMS", "Oracle"]
    values = [rms(data["main"]), rms(lms_clean), rms(oracle_clean)]
    colors = ["grey", "tab:orange", "tab:red"]
    if rl_clean is not None:
        labels.insert(2, "RL (PPO)")
        values.insert(2, rms(rl_clean))
        colors.insert(2, "tab:blue")
    bars = ax4.bar(labels, values, color=colors, alpha=0.8, edgecolor="black", lw=0.7)
    ax4.set_ylabel("RMS amplitude")
    ax4.set_title("Overall RMS comparison")
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, val * 1.02, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax4.grid(alpha=0.3, axis="y")

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "noise_cancellation_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def print_metrics(data, lms_clean, rl_clean):
    oracle_rms = rms(data["sensor_noise"])
    raw_rms    = rms(data["main"])
    lms_rms    = rms(lms_clean)

    print("\n" + "=" * 55)
    print("  Performance summary")
    print("=" * 55)
    print(f"  Oracle (sensor noise floor) : {oracle_rms:.4f}")
    print(f"  Raw main channel            : {raw_rms:.4f}  ({raw_rms/oracle_rms:.1f}× oracle)")
    print(f"  LMS filter                  : {lms_rms:.4f}  ({lms_rms/oracle_rms:.1f}× oracle)")
    if rl_clean is not None:
        rl_rms = rms(rl_clean)
        print(f"  RL agent (PPO)              : {rl_rms:.4f}  ({rl_rms/oracle_rms:.1f}× oracle)")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="models/ppo_noise_cancellation")
    p.add_argument("--no-model", action="store_true",
                   help="Skip RL evaluation (compare LMS vs oracle only)")
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--duration", type=float, default=60.0,
                   help="Evaluation episode duration in seconds (default: 60)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = SignalConfig()
    sim = SignalSimulator(cfg, seed=args.seed)
    data = sim.generate_episode(duration=args.duration, signal_amplitude=0.0)

    print(f"Generated {args.duration:.0f} s of evaluation data  "
          f"(fs={cfg.fs} Hz, {len(data['time'])} samples)")

    # --- LMS baseline ---
    lms_clean = run_lms(data, filter_length=args.window_size)
    print("LMS filter done.")

    # --- RL agent ---
    rl_clean = None
    if not args.no_model:
        model_zip = args.model_path + ".zip"
        vecnorm   = args.model_path + "_vecnorm.pkl"
        if os.path.exists(model_zip) and os.path.exists(vecnorm):
            from stable_baselines3 import PPO
            model = PPO.load(model_zip)
            print(f"Loaded model from {model_zip}")
            rl_clean = run_rl_agent(data, model, vecnorm, window_size=args.window_size)
            print("RL rollout done.")
        else:
            print(f"No model found at {model_zip} — run  python train.py  first.")
            print("Proceeding without RL agent.")

    print_metrics(data, lms_clean, rl_clean)
    plot_overview(data, lms_clean, rl_clean, save_dir=args.save_dir, fs=cfg.fs)
    print("\nDone. Figures saved to", args.save_dir)


if __name__ == "__main__":
    main()
