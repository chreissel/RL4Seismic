"""
Train a PPO agent on the NoiseCancellationEnv.

Usage
-----
    python train.py                          # default settings
    python train.py --timesteps 1_000_000   # longer run
    python train.py --save-path models/ppo  # custom output path

The trained model is saved to  models/ppo_noise_cancellation.zip
and a tensorboard log to  logs/ppo_noise_cancellation/

Quick sanity check after training:
    python evaluate.py
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from noise_removal import NoiseCancellationEnv, SignalConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO noise-cancellation agent")
    p.add_argument("--timesteps", type=int, default=500_000,
                   help="Total training timesteps (default: 500 000)")
    p.add_argument("--n-envs", type=int, default=4,
                   help="Number of parallel training environments (default: 4)")
    p.add_argument("--window-size", type=int, default=64,
                   help="Observation window in samples (default: 64 = 0.5 s)")
    p.add_argument("--episode-duration", type=float, default=30.0,
                   help="Episode length in seconds (default: 30)")
    p.add_argument("--save-path", type=str, default="models/ppo_noise_cancellation",
                   help="Path to save the trained model (without .zip)")
    p.add_argument("--log-dir", type=str, default="logs/ppo_noise_cancellation",
                   help="Tensorboard log directory")
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
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    config = SignalConfig()

    print("=" * 60)
    print("  RL Noise Cancellation — PPO Training")
    print("=" * 60)
    print(f"  Sampling rate  : {config.fs} Hz")
    print(f"  Witness freq   : {config.witness_freq} Hz")
    print(f"  Sensor noise σ : {config.sensor_noise_sigma}")
    print(f"  Coupling model : A(t)·w + B(t)·w² + C(t)·w³  (time-varying)")
    print(f"  Window size    : {args.window_size} samples"
          f" = {args.window_size/config.fs:.3f} s")
    print(f"  Episode length : {args.episode_duration} s")
    print(f"  Total steps    : {args.timesteps:,}")
    print(f"  Parallel envs  : {args.n_envs}")
    print("=" * 60)

    # Vectorised environment with observation normalisation
    vec_env = make_vec_env(
        make_env(config, args.window_size, args.episode_duration),
        n_envs=args.n_envs,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # PPO agent with a small MLP (observations are 128-dim)
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=1e-3,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=args.log_dir,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # Save model and normalisation statistics together
    model.save(args.save_path)
    vec_env.save(args.save_path + "_vecnorm.pkl")

    print(f"\nModel saved to  {args.save_path}.zip")
    print(f"VecNormalize    {args.save_path}_vecnorm.pkl")
    print("Run  python evaluate.py  to see results.")


if __name__ == "__main__":
    main()
