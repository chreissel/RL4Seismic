"""
Train a RecurrentPPO agent on the NoiseCancellationEnv.

Uses sb3-contrib RecurrentPPO with MlpLstmPolicy so the agent maintains
LSTM hidden state across timesteps within an episode.  This allows it to
behave like an adaptive filter — observing residuals, updating its implicit
internal model, and correcting — rather than applying a fixed open-loop
mapping as a plain MLP would.

Usage
-----
    python train.py                          # default settings
    python train.py --timesteps 2_000_000   # longer run
    python train.py --save-path models/ppo  # custom output path

The trained model is saved to  models/ppo_noise_cancellation.zip
and VecNormalize stats to      models/ppo_noise_cancellation_vecnorm.pkl

Quick sanity check after training:
    python evaluate.py
"""

import argparse
import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from noise_removal import NoiseCancellationEnv, SignalConfig, SeismicConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO noise-cancellation agent")
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total training timesteps (default: 2 000 000)")
    p.add_argument("--n-envs", type=int, default=4,
                   help="Number of parallel training environments (default: 4)")
    p.add_argument("--window-size", type=int, default=None,
                   help="Observation window in samples "
                        "(default: 240 for seismic = 60 s @ 4 Hz, 64 otherwise)")
    p.add_argument("--episode-duration", type=float, default=None,
                   help="Episode length in seconds "
                        "(default: 300 for seismic, 30 otherwise)")
    p.add_argument("--save-path", type=str, default="models/ppo_noise_cancellation",
                   help="Path to save the trained model (without .zip)")
    p.add_argument("--log-dir", type=str, default="logs/ppo_noise_cancellation",
                   help="Tensorboard log directory")
    p.add_argument("--multi-source", action="store_true",
                   help="Enable second witness channel with cross-term coupling "
                        "(harder for linear adaptive filters)")
    p.add_argument("--regime-changes", action="store_true",
                   help="Enable sudden coupling regime switches (Poisson process); "
                        "adaptive filters must re-converge after each jump")
    p.add_argument("--seismic", action="store_true",
                   help="Use physically motivated seismic model: linear FIR coupling "
                        "with OU-drifting resonance parameters (replaces polynomial model)")
    p.add_argument("--tilt-coupling", action="store_true",
                   help="Add bilinear tilt-to-length cross-coupling "
                        "(seismic + multi-source only): C_T2L = T(t)·θ(t)·w1(t)")
    return p.parse_args()


def make_env(config, window_size, episode_duration):
    def _init():
        return NoiseCancellationEnv(
            config=config,
            window_size=window_size,
            episode_duration=episode_duration,
        )
    return _init


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.seismic:
        config = SeismicConfig(
            multi_source=args.multi_source,
            regime_changes=args.regime_changes,
            tilt_coupling=args.tilt_coupling,
        )
    else:
        config = SignalConfig(
            multi_source=args.multi_source,
            regime_changes=args.regime_changes,
        )

    # Seismic-aware defaults: 4 Hz sampling needs a larger window and longer episodes
    window_size      = args.window_size      if args.window_size      is not None else (
        config.filter_length if args.seismic else 64
    )
    episode_duration = args.episode_duration if args.episode_duration is not None else (
        300.0 if args.seismic else 30.0
    )

    print("=" * 60)
    print("  RL Noise Cancellation — RecurrentPPO Training")
    print("=" * 60)
    print(f"  Sampling rate  : {config.fs} Hz")
    if hasattr(config, "witness_freq"):
        print(f"  Witness freq   : {config.witness_freq} Hz")
    print(f"  Sensor noise σ : {config.sensor_noise_sigma}")
    if args.seismic:
        if config.multi_source and getattr(config, "tilt_coupling", False):
            print(f"  Coupling model : h1(t)⊛w1 + h2(t)⊛w2 + T(t)·θ(t)·w1  (seismic + T2L)")
        elif config.multi_source:
            print(f"  Coupling model : h1(t)⊛w1 + h2(t)⊛w2  (seismic multi-source)")
        else:
            print(f"  Coupling model : h(t)⊛w  (seismic: OU-drifting resonant FIR)")
    elif config.multi_source:
        print(f"  Coupling model : A·w1 + B·w1² + C·w1³ + D·w2 + E·w1·w2  (multi-source)")
    elif config.regime_changes:
        print(f"  Coupling model : A_k·w + B_k·w² + C_k·w³  ({config.n_regimes} regimes, "
              f"mean hold {config.mean_hold_time:.0f} s)")
    else:
        print(f"  Coupling model : A(t)·w + B(t)·w² + C(t)·w³  (single-source)")
    print(f"  Window size    : {window_size} samples"
          f" = {window_size/config.fs:.3f} s")
    print(f"  Episode length : {episode_duration} s")
    print(f"  Total steps    : {args.timesteps:,}")
    print(f"  Parallel envs  : {args.n_envs}")
    print("=" * 60)

    # Vectorised environment with observation normalisation
    vec_env = make_vec_env(
        make_env(config, window_size, episode_duration),
        n_envs=args.n_envs,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # RecurrentPPO — LSTM hidden state lets the agent adapt within an episode,
    # mimicking an online adaptive filter rather than a fixed open-loop mapping.
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=1e-4,
        policy_kwargs=dict(
            net_arch=[256, 256],
            lstm_hidden_size=256,
            n_lstm_layers=1,
        ),
        tensorboard_log=None,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=os.path.dirname(args.save_path),
        name_prefix=os.path.basename(args.save_path),
        save_vecnormalize=True,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb, progress_bar=False)

    # Save model and normalisation statistics together
    model.save(args.save_path)
    vec_env.save(args.save_path + "_vecnorm.pkl")

    print(f"\nModel saved to  {args.save_path}.zip")
    print(f"VecNormalize    {args.save_path}_vecnorm.pkl")
    print("Run  python evaluate.py  to see results.")


if __name__ == "__main__":
    main()
