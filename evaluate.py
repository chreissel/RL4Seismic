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
  RMS of the output signal in four conditions:
    - No subtraction     (raw main channel)
    - LMS filter         (linear FIR adaptive baseline)
    - IIR filter         (closed-loop adaptive baseline, mirrors RL observation)
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
from noise_removal import SeismicConfig, SeismicSignalSimulator
from baselines import LMSFilter, IIRFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def run_lms(
    data: dict,
    filter_length: int = 64,
    step_size: float = 5e-4,
    normalized: bool = False,
) -> np.ndarray:
    filt = LMSFilter(filter_length=filter_length, step_size=step_size, normalized=normalized)
    w2 = data.get("witness2")
    return filt.run(data["witness"], data["main"], witness2=w2)


def run_iir(
    data: dict,
    feedforward_length: int = 64,
    feedback_length: int = 64,
    step_size: float = 5e-4,
    normalized: bool = False,
) -> np.ndarray:
    filt = IIRFilter(
        feedforward_length=feedforward_length,
        feedback_length=feedback_length,
        step_size=step_size,
        normalized=normalized,
    )
    w2 = data.get("witness2")
    return filt.run(data["witness"], data["main"], witness2=w2)


def run_rl_agent(data: dict, model, vec_norm, window_size: int = 64) -> np.ndarray:
    """Roll out the RL agent sample-by-sample on pre-generated data.

    Supports both plain PPO (MlpPolicy) and RecurrentPPO (MlpLstmPolicy).
    For recurrent models the LSTM state is carried step-to-step so the agent
    can adapt within the episode, just as it did during training.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from noise_removal.environment import NoiseCancellationEnv

    n = len(data["time"])
    cleaned = data["main"].copy()

    cfg = SignalConfig()
    env = NoiseCancellationEnv(config=cfg, window_size=window_size)

    # Manually set the episode data so we evaluate on the *same* data as LMS
    env._data = data
    env._n_samples = n
    env._step_idx = window_size
    env._action_history = np.zeros(n, dtype=np.float64)  # required for closed-loop obs

    obs = env._get_obs()

    dummy = DummyVecEnv([lambda: NoiseCancellationEnv(config=cfg, window_size=window_size)])
    dummy_norm = VecNormalize.load(vec_norm, dummy)
    dummy_norm.training = False
    dummy_norm.norm_reward = False

    # Recurrent models need LSTM state and episode-start flag carried across steps
    is_recurrent = hasattr(model, "policy") and hasattr(model.policy, "lstm_actor")
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    for t in range(window_size, n):
        obs_norm = dummy_norm.normalize_obs(obs[np.newaxis, :])

        if is_recurrent:
            action, lstm_states = model.predict(
                obs_norm, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs_norm, deterministic=True)

        a = float(np.clip(action[0][0], -15.0, 15.0))
        cleaned[t] = data["main"][t] - a

        env._action_history[t] = a
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
    iir_clean: np.ndarray,
    rl_clean: Optional[np.ndarray],
    save_dir: str,
    fs: float = 128.0,
    seismic: bool = False,
):
    t = data["time"]
    oracle_clean = data["sensor_noise"]  # best possible: coupling perfectly removed

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("RL Noise Cancellation — Method Comparison", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # ---- Top panel: coupling info ----
    ax0 = fig.add_subplot(gs[0, :])
    if seismic:
        # Show the coupling waveform and witness for seismic model
        ax0.plot(t, data["coupling"], lw=0.7, alpha=0.8, label="Coupling h(t)⊛w(t)", color="tab:blue")
        ax0.plot(t, data["witness"], lw=0.6, alpha=0.5, label="Witness w(t)", color="tab:orange")
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("Amplitude")
        ax0.set_title("Seismic coupling: linear time-varying FIR  y(t) = h(t) ⊛ w(t) + n(t)")
        ax0.legend(loc="upper right", fontsize=9)
    else:
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
    ax1.plot(t[mask], iir_clean[mask], alpha=0.8, lw=0.9, label="IIR", color="tab:purple")
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
        (data["main"], "Raw",    "grey",       1.0),
        (lms_clean,    "LMS",    "tab:orange", 1.2),
        (iir_clean,    "IIR",    "tab:purple", 1.2),
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
    ax3.plot(t, rolling_rms(iir_clean), lw=0.9, label="IIR", color="tab:purple")
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
    labels = ["Raw", "LMS", "IIR", "Oracle"]
    values = [rms(data["main"]), rms(lms_clean), rms(iir_clean), rms(oracle_clean)]
    colors = ["grey", "tab:orange", "tab:purple", "tab:red"]
    if rl_clean is not None:
        labels.insert(3, "RL (PPO)")
        values.insert(3, rms(rl_clean))
        colors.insert(3, "tab:blue")
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


def print_metrics(data, lms_clean, iir_clean, rl_clean):
    oracle_rms = rms(data["sensor_noise"])
    raw_rms    = rms(data["main"])
    lms_rms    = rms(lms_clean)
    iir_rms    = rms(iir_clean)

    print("\n" + "=" * 55)
    print("  Performance summary")
    print("=" * 55)
    print(f"  Oracle (sensor noise floor) : {oracle_rms:.4f}")
    print(f"  Raw main channel            : {raw_rms:.4f}  ({raw_rms/oracle_rms:.1f}× oracle)")
    print(f"  LMS filter (FIR)            : {lms_rms:.4f}  ({lms_rms/oracle_rms:.1f}× oracle)")
    print(f"  IIR adaptive filter         : {iir_rms:.4f}  ({iir_rms/oracle_rms:.1f}× oracle)")
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
    p.add_argument("--multi-source", action="store_true",
                   help="Evaluate on multi-source coupling with cross-term w1·w2")
    p.add_argument("--regime-changes", action="store_true",
                   help="Evaluate on piecewise-constant coupling with sudden regime switches")
    p.add_argument("--seismic", action="store_true",
                   help="Use physically motivated seismic model (linear FIR + OU drift)")
    p.add_argument("--tilt-coupling", action="store_true",
                   help="Add bilinear tilt-to-length cross-coupling "
                        "(seismic + multi-source only): C_T2L = T(t)·θ(t)·w1(t); "
                        "creates an irreducible floor for all linear baselines")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seismic:
        cfg = SeismicConfig(
            multi_source=args.multi_source,
            regime_changes=args.regime_changes,
            tilt_coupling=args.tilt_coupling,
        )
        sim = SeismicSignalSimulator(cfg, seed=args.seed)
    else:
        cfg = SignalConfig(
            multi_source=args.multi_source,
            regime_changes=args.regime_changes,
        )
        sim = SignalSimulator(cfg, seed=args.seed)
    data = sim.generate_episode(duration=args.duration, signal_amplitude=0.0)

    print(f"Generated {args.duration:.0f} s of evaluation data  "
          f"(fs={cfg.fs} Hz, {len(data['time'])} samples)")

    # --- LMS baseline ---
    # NLMS is used for the seismic model: coloured 1/f² ground motion creates
    # a large eigenvalue spread that causes plain LMS to diverge.
    lms_clean = run_lms(data, filter_length=args.window_size, normalized=args.seismic)
    print("LMS filter done.")

    # --- IIR baseline ---
    # For the seismic model the coupling is purely FIR (no IIR feedback in the
    # physics), so the optimal IIR feedback weights are zero.  We set
    # feedback_length=0 to avoid the equation-error instability that arises
    # from a non-zero feedback length on coloured seismic signals.
    iir_fb = 0 if args.seismic else args.window_size
    iir_clean = run_iir(
        data,
        feedforward_length=args.window_size,
        feedback_length=iir_fb,
        normalized=args.seismic,
    )
    print("IIR filter done.")

    # --- RL agent ---
    rl_clean = None
    if not args.no_model:
        model_zip = args.model_path + ".zip"
        vecnorm   = args.model_path + "_vecnorm.pkl"
        if os.path.exists(model_zip) and os.path.exists(vecnorm):
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(model_zip)
            print(f"Loaded model from {model_zip}")
            rl_clean = run_rl_agent(data, model, vecnorm, window_size=args.window_size)
            print("RL rollout done.")
        else:
            print(f"No model found at {model_zip} — run  python train.py  first.")
            print("Proceeding without RL agent.")

    print_metrics(data, lms_clean, iir_clean, rl_clean)
    plot_overview(
        data, lms_clean, iir_clean, rl_clean,
        save_dir=args.save_dir, fs=cfg.fs,
        seismic=args.seismic,
    )
    print("\nDone. Figures saved to", args.save_dir)


if __name__ == "__main__":
    main()
