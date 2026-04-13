"""
Evaluate a trained PPO agent on the seismic noise cancellation task
and compare against classical and learned baselines.

Usage
-----
    python evaluate.py                            # use saved model (tilt coupling on by default)
    python evaluate.py --model-path models/ppo_noise_cancellation
    python evaluate.py --no-model                 # skip RL, compare baselines only
    python evaluate.py --no-tilt-coupling         # X coupling only (easier baseline)
    python evaluate.py --no-drift                 # disable OU drift (stationary system)
    python evaluate.py --no-lstm --no-model       # fastest: baselines only

Produces
--------
  results/noise_cancellation_overview.png  — time-domain & spectral comparison

Metrics reported
----------------
  RMS of the output signal:
    - No subtraction     (raw main channel)
    - Wiener filter      (batch-optimal linear FIR, trained offline)
    - IIR adaptive filter (online closed-loop baseline)
    - Supervised LSTM    (offline-trained, from arXiv:2511.19682)
    - RL agent (PPO)     (online adaptive, this work)
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

from noise_removal import NoiseCancellationEnv, SeismicConfig, SeismicSignalSimulator
from baselines import IIRFilter, SupervisedLSTM, WienerFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def run_wiener(
    train_data: dict,
    eval_data: dict,
    filter_length: int = 240,
    use_witness_y: bool = True,
) -> np.ndarray:
    """Train a Wiener filter on *train_data* and apply it to *eval_data*."""
    wy_train = train_data["witness_y"] if use_witness_y else None
    wy_eval = eval_data["witness_y"] if use_witness_y else None
    filt = WienerFilter(filter_length=filter_length)
    filt.fit(train_data["witness_x"], train_data["main"], witness_y=wy_train)
    return filt.run(eval_data["witness_x"], eval_data["main"], witness_y=wy_eval)


def run_iir(
    data: dict,
    feedforward_length: int = 240,
    step_size: float = 0.1,
    use_witness_y: bool = True,
) -> np.ndarray:
    # feedback_length=0: the seismic coupling is purely FIR; non-zero feedback
    # on coloured seismic signals leads to equation-error instability.
    filt = IIRFilter(
        feedforward_length=feedforward_length,
        feedback_length=0,
        step_size=step_size,
        normalized=True,
    )
    wy = data["witness_y"] if use_witness_y else None
    return filt.run(data["witness_x"], data["main"], witness_y=wy)


def run_supervised_lstm(
    train_data: dict,
    eval_data: dict,
    window_size: int = 240,
    hidden_size: int = 128,
    n_layers: int = 3,
    n_epochs: int = 30,
    verbose: bool = False,
) -> np.ndarray:
    """
    Train a supervised LSTM on train_data and evaluate on eval_data.

    Mirrors arXiv:2511.19682: causal 3-layer LSTM (hidden 128), MSE loss,
    trained offline on a separate episode, then run causally on the eval episode.
    """
    lstm = SupervisedLSTM(
        window_size=window_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_epochs=n_epochs,
    )
    if verbose:
        print(f"  Training supervised LSTM ({n_epochs} epochs)…")
    lstm.fit(train_data, verbose=verbose)

    return lstm.run(eval_data["witness_x"], eval_data["main"], witness_y=eval_data["witness_y"])


def run_rl_agent(
    data: dict,
    model,
    vec_norm,
    window_size: int = 240,
    config: Optional[SeismicConfig] = None,
) -> np.ndarray:
    """Roll out the RL agent sample-by-sample on pre-generated data.

    Supports both plain PPO (MlpPolicy) and RecurrentPPO (MlpLstmPolicy).
    For recurrent models the LSTM state is carried step-to-step so the agent
    can adapt within the episode, just as it did during training.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from noise_removal.environment import NoiseCancellationEnv

    if config is None:
        config = SeismicConfig()

    n = len(data["time"])
    cleaned = data["main"].copy()

    env = NoiseCancellationEnv(config=config, window_size=window_size)
    env._data = data
    env._n_samples = n
    env._step_idx = window_size
    env._action_history = np.zeros(n, dtype=np.float64)

    obs = env._get_obs()

    _cfg = config
    dummy = DummyVecEnv([lambda: NoiseCancellationEnv(config=_cfg, window_size=window_size)])
    dummy_norm = VecNormalize.load(vec_norm, dummy)
    dummy_norm.training = False
    dummy_norm.norm_reward = False

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

        a = float(np.clip(action[0][0], -env.action_clip, env.action_clip))
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
    wiener_clean: np.ndarray,
    iir_clean: np.ndarray,
    rl_clean: Optional[np.ndarray],
    save_dir: str,
    fs: float = 4.0,
    lstm_clean: Optional[np.ndarray] = None,
):
    t = data["time"]
    oracle_clean = data["sensor_noise"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("RL Seismic Noise Cancellation — Method Comparison", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # ---- Top panel: coupling waveform ----
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, data["coupling"], lw=0.7, alpha=0.8, label="Total coupling", color="tab:blue")
    ax0.plot(t, data["witness_x"], lw=0.6, alpha=0.5, label="Witness X (inline)", color="tab:orange")
    ax0.plot(t, data["witness_y"], lw=0.6, alpha=0.4, label="Witness Y (perpendicular)", color="tab:green")
    if "coupling_tilt" in data:
        ax0.plot(t, data["coupling_tilt"], lw=0.6, alpha=0.7, label="Tilt-horizontal term C_tilt(t)", color="tab:red")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Seismic coupling: y(t) = h_x(t)⊛w_x(t) + C_tilt(t) + n_GS13X(t)")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(alpha=0.3)

    # ---- Time-domain: first 30 s ----
    mask = t < 30.0
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t[mask], data["main"][mask], alpha=0.6, lw=0.8, label="Raw", color="grey")
    ax1.plot(t[mask], wiener_clean[mask], alpha=0.8, lw=0.9, label="Wiener", color="tab:orange")
    ax1.plot(t[mask], iir_clean[mask], alpha=0.8, lw=0.9, label="IIR", color="tab:purple")
    if lstm_clean is not None:
        ax1.plot(t[mask], lstm_clean[mask], alpha=0.9, lw=0.9, label="LSTM (supervised)", color="tab:green")
    if rl_clean is not None:
        ax1.plot(t[mask], rl_clean[mask], alpha=0.9, lw=0.9, label="RL (PPO)", color="tab:blue")
    ax1.plot(t[mask], oracle_clean[mask], "--", lw=0.8, label="Oracle", color="tab:red", alpha=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time domain (first 30 s)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ---- PSD comparison ----
    ax2 = fig.add_subplot(gs[1, 1])
    nperseg = min(512, len(data["main"]))
    for sig, label, color, lw in [
        (data["main"],   "Raw",    "grey",       1.0),
        (wiener_clean,   "Wiener", "tab:orange", 1.2),
        (iir_clean,      "IIR",    "tab:purple", 1.2),
        (oracle_clean,   "Oracle", "tab:red",    1.0),
    ]:
        f_psd, pxx = welch(sig, fs=fs, nperseg=nperseg)
        ax2.semilogy(f_psd, pxx, label=label, color=color, lw=lw, alpha=0.8)
    if lstm_clean is not None:
        f_psd, pxx = welch(lstm_clean, fs=fs, nperseg=nperseg)
        ax2.semilogy(f_psd, pxx, label="LSTM (supervised)", color="tab:green", lw=1.2)
    if rl_clean is not None:
        f_psd, pxx = welch(rl_clean, fs=fs, nperseg=nperseg)
        ax2.semilogy(f_psd, pxx, label="RL (PPO)", color="tab:blue", lw=1.3)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD")
    ax2.set_title("Power Spectral Density")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    # ---- Rolling RMS (30 s window) ----
    ax3 = fig.add_subplot(gs[2, 0])
    win = int(30.0 * fs)
    def rolling_rms(x):
        return np.array([
            np.sqrt(np.mean(x[max(0, i - win):i + 1]**2))
            for i in range(len(x))
        ])
    ax3.plot(t, rolling_rms(data["main"]), lw=0.8, label="Raw", color="grey", alpha=0.7)
    ax3.plot(t, rolling_rms(wiener_clean), lw=0.9, label="Wiener", color="tab:orange")
    ax3.plot(t, rolling_rms(iir_clean), lw=0.9, label="IIR", color="tab:purple")
    if lstm_clean is not None:
        ax3.plot(t, rolling_rms(lstm_clean), lw=0.9, label="LSTM (supervised)", color="tab:green")
    if rl_clean is not None:
        ax3.plot(t, rolling_rms(rl_clean), lw=0.9, label="RL (PPO)", color="tab:blue")
    ax3.axhline(rms(oracle_clean), ls="--", color="tab:red", lw=0.9, label="Oracle RMS")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Rolling RMS (30 s window)")
    ax3.set_title("Rolling RMS — noise floor over time")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # ---- Summary bar chart ----
    ax4 = fig.add_subplot(gs[2, 1])
    labels = ["Raw", "Wiener", "IIR", "Oracle"]
    values = [rms(data["main"]), rms(wiener_clean), rms(iir_clean), rms(oracle_clean)]
    colors = ["grey", "tab:orange", "tab:purple", "tab:red"]
    if lstm_clean is not None:
        labels.insert(3, "LSTM")
        values.insert(3, rms(lstm_clean))
        colors.insert(3, "tab:green")
    if rl_clean is not None:
        labels.insert(-1, "RL (PPO)")
        values.insert(-1, rms(rl_clean))
        colors.insert(-1, "tab:blue")
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


def print_metrics(data, wiener_clean, iir_clean, rl_clean, lstm_clean=None,
                  warmup_fraction: float = 0.25):
    """Print RMS metrics for all methods.

    Reports both full-episode and steady-state (after warmup) performance.
    Online adaptive filters (IIR) need time to converge; the steady-state
    metric excludes the initial convergence transient.  The Wiener filter
    has no transient (trained offline) so both columns should be similar.
    """
    oracle_rms  = rms(data["sensor_noise"])
    raw_rms     = rms(data["main"])
    wiener_rms  = rms(wiener_clean)
    iir_rms     = rms(iir_clean)

    n = len(data["main"])
    ss = int(n * warmup_fraction)  # steady-state start index
    oracle_ss  = rms(data["sensor_noise"][ss:])
    wiener_ss  = rms(wiener_clean[ss:])
    iir_ss     = rms(iir_clean[ss:])

    print("\n" + "=" * 72)
    print("  Performance summary")
    print("=" * 72)
    print(f"  {'Method':<30s}  {'Full episode':>14s}  {'Steady-state':>14s}")
    print(f"  {'':30s}  {'(x oracle)':>14s}  {'(x oracle, >' + f'{warmup_fraction:.0%})':>14s}")
    print("-" * 72)
    print(f"  Oracle (sensor noise floor)   : {oracle_rms:.4f}")
    if "coupling_tilt" in data:
        tilt_rms = rms(data["coupling_tilt"])
        linear_floor = float(np.sqrt(tilt_rms**2 + oracle_rms**2))
        print(f"  Tilt-horizontal coupling RMS  : {tilt_rms:.4f}  ({tilt_rms/oracle_rms:.1f}x oracle)")
    print(f"  Raw main channel              : {raw_rms/oracle_rms:6.1f}x")
    print(f"  Wiener filter (offline)       : {wiener_rms/oracle_rms:6.1f}x{wiener_ss/oracle_ss:15.1f}x")
    print(f"  IIR adaptive filter (online)  : {iir_rms/oracle_rms:6.1f}x{iir_ss/oracle_ss:15.1f}x")
    if lstm_clean is not None:
        lstm_rms = rms(lstm_clean)
        lstm_ss  = rms(lstm_clean[ss:])
        print(f"  Supervised LSTM               : {lstm_rms/oracle_rms:6.1f}x{lstm_ss/oracle_ss:15.1f}x")
    if rl_clean is not None:
        rl_rms = rms(rl_clean)
        rl_ss  = rms(rl_clean[ss:])
        beats = ""
        if "coupling_tilt" in data:
            linear_floor = float(np.sqrt(rms(data["coupling_tilt"])**2 + oracle_rms**2))
            if rl_rms < linear_floor:
                beats = " *** BEATS LINEAR FLOOR ***"
        print(f"  RL agent (RecurrentPPO)       : {rl_rms/oracle_rms:6.1f}x{rl_ss/oracle_ss:15.1f}x{beats}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="models/ppo_noise_cancellation")
    p.add_argument("--no-model", action="store_true",
                   help="Skip RL evaluation (compare baselines only)")
    p.add_argument("--window-size", type=int, default=None,
                   help="Context window in samples (default: 240 = 60 s @ 4 Hz)")
    p.add_argument("--duration", type=float, default=300.0,
                   help="Eval episode duration in seconds (default: 300)")
    p.add_argument("--train-duration", type=float, default=None,
                   help="Training episode duration for supervised LSTM "
                        "(default: 3× eval duration)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-dir", default="results")
    p.add_argument("--no-drift", action="store_true",
                   help="Disable OU drift of coupling parameters (stationary system; "
                        "linear baselines should reach near-oracle performance)")
    p.add_argument("--regime-changes", action="store_true",
                   help="Evaluate with piecewise-constant coupling regime switches")
    p.add_argument("--tilt-coupling", action="store_true", default=True,
                   help="Include tilt-horizontal coupling from Y seismic channel "
                        "(default: on). C_tilt(t) = T(t)·θ_y_proxy(t)")
    p.add_argument("--no-tilt-coupling", action="store_true",
                   help="Disable tilt-horizontal coupling (X coupling only)")
    p.add_argument("--no-lstm", action="store_true",
                   help="Skip supervised LSTM baseline (faster evaluation)")
    p.add_argument("--lstm-epochs", type=int, default=30,
                   help="Training epochs for the supervised LSTM (default: 30)")
    p.add_argument("--dilated-conv", action="store_true",
                   help="Load model as plain PPO (DLS) instead of RecurrentPPO")
    p.add_argument("--sensor-noise-color",
                   choices=("white", "pink", "brown"),
                   default="white",
                   help="Spectral shape of the GS13X sensor noise (oracle noise "
                        "floor). 'white' = flat PSD (default), 'pink' = 1/f, "
                        "'brown' = 1/f². RMS is always sensor_noise_sigma.")
    p.add_argument("--sensor-noise-exponent", type=float, default=None,
                   help="Override --sensor-noise-color with a custom spectral "
                        "exponent α (PSD ∝ 1/f^α).")
    p.add_argument("--sensor-noise-band", type=float, nargs=2,
                   metavar=("F_LOW", "F_HIGH"), default=None,
                   help="Optional (f_low, f_high) Hz window over which to "
                        "enforce the sensor-noise RMS. When given, "
                        "sensor_noise_sigma is interpreted as the in-band RMS "
                        "on this window instead of the broadband RMS.")
    return p.parse_args()


def main():
    args = parse_args()

    color_to_exp = {"white": 0.0, "pink": 1.0, "brown": 2.0}
    sensor_exp = (
        args.sensor_noise_exponent
        if args.sensor_noise_exponent is not None
        else color_to_exp[args.sensor_noise_color]
    )

    sensor_band = tuple(args.sensor_noise_band) if args.sensor_noise_band else None

    cfg = SeismicConfig(
        drift=not args.no_drift,
        regime_changes=args.regime_changes,
        tilt_coupling=not args.no_tilt_coupling,
        sensor_noise_exponent=sensor_exp,
        sensor_noise_band=sensor_band,
    )

    window_size = args.window_size if args.window_size is not None else cfg.filter_length
    train_duration = args.train_duration if args.train_duration is not None else args.duration * 3

    sim = SeismicSignalSimulator(cfg, seed=args.seed)
    data = sim.generate_episode(duration=args.duration, signal_amplitude=0.0)

    print(f"Generated {args.duration:.0f} s of evaluation data  "
          f"(fs={cfg.fs} Hz, {len(data['time'])} samples)")

    # Only use witness Y for baselines when tilt coupling is active;
    # otherwise it doubles the filter size for zero benefit.
    use_wy = cfg.tilt_coupling

    # --- Training data for offline methods (Wiener, supervised LSTM) ---
    print(f"Generating {train_duration:.0f} s of training data  "
          f"(seed={args.seed + 1000})…")
    train_sim  = SeismicSignalSimulator(cfg, seed=args.seed + 1000)
    train_data = train_sim.generate_episode(duration=train_duration, signal_amplitude=0.0)

    # --- Wiener filter baseline (batch-optimal linear) ---
    wiener_clean = run_wiener(train_data, data, filter_length=window_size,
                              use_witness_y=use_wy)
    print("Wiener filter done.")

    # --- IIR adaptive filter baseline (online) ---
    iir_clean = run_iir(data, feedforward_length=window_size, step_size=0.1,
                        use_witness_y=use_wy)
    print("IIR filter done.")

    # --- Supervised LSTM baseline (paper approach) ---
    lstm_clean = None
    if not args.no_lstm:
        print(f"Training supervised LSTM ({args.lstm_epochs} epochs)…")
        lstm_clean = run_supervised_lstm(
            train_data, data,
            window_size=window_size,
            n_epochs=args.lstm_epochs,
            verbose=True,
        )
        print("Supervised LSTM done.")

    # --- RL agent ---
    rl_clean = None
    if not args.no_model:
        model_zip = args.model_path + ".zip"
        import re as _re
        vecnorm = args.model_path + "_vecnorm.pkl"
        if not os.path.exists(vecnorm):
            m = _re.search(r"_(\d+)_steps$", args.model_path)
            if m:
                steps = m.group(1)
                prefix = args.model_path[: args.model_path.rindex(f"_{steps}_steps")]
                candidate = f"{prefix}_vecnormalize_{steps}_steps.pkl"
                if os.path.exists(candidate):
                    vecnorm = candidate
        if os.path.exists(model_zip) and os.path.exists(vecnorm):
            if args.dilated_conv:
                from stable_baselines3 import PPO
                model = PPO.load(model_zip)
            else:
                from sb3_contrib import RecurrentPPO
                model = RecurrentPPO.load(model_zip)
            print(f"Loaded model from {model_zip}")
            rl_clean = run_rl_agent(data, model, vecnorm, window_size=window_size, config=cfg)
            print("RL rollout done.")
        else:
            print(f"No model found at {model_zip} — run  python train.py  first.")
            print("Proceeding without RL agent.")

    print_metrics(data, wiener_clean, iir_clean, rl_clean, lstm_clean=lstm_clean)
    plot_overview(
        data, wiener_clean, iir_clean, rl_clean,
        save_dir=args.save_dir, fs=cfg.fs,
        lstm_clean=lstm_clean,
    )
    print("\nDone. Figures saved to", args.save_dir)


if __name__ == "__main__":
    main()
