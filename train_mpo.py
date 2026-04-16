"""
MPO training for seismic noise cancellation with loop-shaping reward.

Uses Maximum a Posteriori Policy Optimization (MPO), the same RL algorithm
as the DeepMind Deep Loop Shaping paper (arXiv:2509.14016), combined with
the bilinear dilated-conv feature extractor and the hybrid loop-shaping
reward.

MPO is an off-policy actor-critic method that uses a KL-constrained
policy update (E-step / M-step decomposition).  Compared to SAC:

  - The policy update is constrained by separate KL bounds on mean and
    variance, preventing premature collapse.
  - The temperature η is optimised per-batch via convex dual, not
    learned end-to-end like SAC's entropy coefficient α.
  - MPO tends to be more stable on control tasks with shaped rewards.

Usage
-----
    # Quick smoke test
    python train_mpo.py --timesteps 20000 --episode-duration 180

    # Full training with loop-shaping reward
    python train_mpo.py --timesteps 200000 --episode-duration 600 \\
        --loop-shaping --no-dr --drift-scale 2.0 \\
        --conv-channels 24 --conv-layers 6 --bilinear-rank 16 \\
        --features-dim 192 --batch-size 256 \\
        --save-path models/mpo_bilinear
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from noise_removal import (
    DomainRandomizedNoiseCancellationEnv,
    DomainRandomizationConfig,
    LoopShapingWrapper,
    NoiseCancellationEnv,
    SeismicConfig,
)
from noise_removal.bilinear_policy import BilinearDilatedExtractor
from noise_removal.mpo import MPO


class ActionSmoothnessWrapper(gym.Wrapper):
    """Penalise high-frequency action chatter: -λ·(a_t − a_{t-1})²."""

    def __init__(self, env: gym.Env, smoothness_lambda: float = 0.01):
        super().__init__(env)
        self.smoothness_lambda = smoothness_lambda
        self._last_action = 0.0

    def reset(self, **kwargs):
        self._last_action = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        a = float(action[0]) if hasattr(action, "__len__") else float(action)
        penalty = self.smoothness_lambda * (a - self._last_action) ** 2
        self._last_action = a
        return obs, reward - penalty, term, trunc, info


def make_env(args, seed: int):
    def _thunk():
        if args.no_dr:
            cfg = SeismicConfig(
                drift=True,
                tilt_coupling=True,
                sensor_noise_exponent=2.0,
                sensor_noise_band=(0.05, 0.5),
                gain_drift_sigma=0.1 * args.drift_scale,
                freq_drift_sigma=0.01 * args.drift_scale,
                t2l_gain_drift_sigma=2.7 * args.drift_scale,
            )
            env: gym.Env = NoiseCancellationEnv(
                config=cfg,
                window_size=args.window_size,
                episode_duration=args.episode_duration,
                action_clip=args.action_clip,
            )
        else:
            dr = DomainRandomizationConfig(
                drift_scale_low=args.dr_drift_low,
                drift_scale_high=args.dr_drift_high,
                regime_change_prob=args.dr_regime_prob,
            )
            env = DomainRandomizedNoiseCancellationEnv(
                dr_config=dr,
                window_size=args.window_size,
                episode_duration=args.episode_duration,
                action_clip=args.action_clip,
            )
        if args.loop_shaping:
            _fs = cfg.fs if args.no_dr else 4.0
            env = LoopShapingWrapper(
                env,
                psd_window=args.psd_window,
                f_low=args.freq_band_low,
                f_high=args.freq_band_high,
                amplification_penalty=args.amplification_penalty,
                spectral_weight=args.spectral_weight,
                mode=args.reward_mode,
                fs=_fs,
            )
        env = ActionSmoothnessWrapper(env, smoothness_lambda=args.smoothness_lambda)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk


def parse_args():
    p = argparse.ArgumentParser(description="Train MPO for seismic noise cancellation")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--window-size", type=int, default=240)
    p.add_argument("--episode-duration", type=float, default=300.0)

    p.add_argument("--no-dr", action="store_true",
                   help="Disable domain randomisation.")
    p.add_argument("--drift-scale", type=float, default=2.0)
    p.add_argument("--dr-drift-low", type=float, default=0.5)
    p.add_argument("--dr-drift-high", type=float, default=4.0)
    p.add_argument("--dr-regime-prob", type=float, default=0.3)

    # Feature extractor
    p.add_argument("--conv-channels", type=int, default=48)
    p.add_argument("--conv-layers", type=int, default=8)
    p.add_argument("--bilinear-rank", type=int, default=32)
    p.add_argument("--features-dim", type=int, default=256)

    # MPO hyperparameters
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--lr-dual", type=float, default=1e-2,
                   help="Learning rate for MPO dual variables (η, α_μ, α_σ)")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="KL bound ε for the E-step (non-parametric improvement)")
    p.add_argument("--epsilon-mean", type=float, default=0.01,
                   help="KL bound ε_μ for the M-step (policy mean constraint)")
    p.add_argument("--epsilon-var", type=float, default=1e-4,
                   help="KL bound ε_σ for the M-step (policy variance constraint)")
    p.add_argument("--n-action-samples", type=int, default=64,
                   help="Number of action samples K per state in E-step")
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=1_000)

    # Reward shaping
    p.add_argument("--smoothness-lambda", type=float, default=0.01)
    p.add_argument("--action-clip", type=float, default=15.0)

    # Loop-shaping reward
    p.add_argument("--loop-shaping", action="store_true", default=True,
                   help="Use hybrid loop-shaping reward (default: on)")
    p.add_argument("--no-loop-shaping", dest="loop_shaping", action="store_false")
    p.add_argument("--freq-band-low", type=float, default=0.05)
    p.add_argument("--freq-band-high", type=float, default=0.5)
    p.add_argument("--psd-window", type=int, default=256)
    p.add_argument("--amplification-penalty", type=float, default=2.0)
    p.add_argument("--spectral-weight", type=float, default=0.5)
    p.add_argument("--reward-mode", choices=["hybrid", "segment"], default="hybrid",
                   help="'hybrid' = dense bandpass + spectral bonus (default). "
                        "'segment' = pure spectral, segment-level PSD reward "
                        "(closer to the DLS paper, best for fine-tuning).")

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-path", default="models/mpo_bilinear")
    p.add_argument("--checkpoint-freq", type=int, default=10_000,
                   help="Save intermediate checkpoint every N env steps "
                        "(default 10000; 0 to disable)")
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    vec_env = DummyVecEnv([make_env(args, seed=args.seed)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    model = MPO(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        features_extractor_class=BilinearDilatedExtractor,
        features_extractor_kwargs=dict(
            window_size=args.window_size,
            conv_channels=args.conv_channels,
            n_layers=args.conv_layers,
            bilinear_rank=args.bilinear_rank,
            features_dim=args.features_dim,
        ),
        net_arch=[256, 256],
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_dual=args.lr_dual,
        epsilon=args.epsilon,
        epsilon_mean=args.epsilon_mean,
        epsilon_var=args.epsilon_var,
        n_action_samples=args.n_action_samples,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        device=args.device,
    )

    if args.loop_shaping:
        reward_desc = (f"loop-shaping |S(f)|² [{args.freq_band_low}, "
                       f"{args.freq_band_high}] Hz (mode={args.reward_mode}, "
                       f"PSD={args.psd_window}, α={args.amplification_penalty}"
                       f"{', λ=' + str(args.spectral_weight) if args.reward_mode == 'hybrid' else ''})")
    else:
        reward_desc = "broadband y_t^2 - e_t^2"

    print(f"Training MPO for {args.timesteps:,} timesteps  "
          f"(episode = {args.episode_duration:.0f} s, window = {args.window_size})")
    print(f"  Feature extractor: BilinearDilated "
          f"(channels={args.conv_channels}, layers={args.conv_layers}, "
          f"bilinear_rank={args.bilinear_rank})")
    print(f"  Domain randomisation: "
          f"{'OFF (drift_scale=' + str(args.drift_scale) + ')' if args.no_dr else 'ON'}")
    print(f"  Reward          : {reward_desc}")
    print(f"  Action clip     : [-{args.action_clip}, +{args.action_clip}]")
    print(f"  MPO ε={args.epsilon}, ε_μ={args.epsilon_mean}, ε_σ={args.epsilon_var}")
    print(f"  K={args.n_action_samples} action samples, γ={args.gamma}, τ={args.tau}")
    print(f"  Device          : {model.device}")

    if args.checkpoint_freq > 0:
        print(f"  Checkpoints     : every {args.checkpoint_freq:,} steps → "
              f"{os.path.dirname(args.save_path) or '.'}/"
              f"{os.path.basename(args.save_path)}_<steps>_steps.zip")

    model.learn(
        total_timesteps=args.timesteps,
        env=vec_env,
        vec_normalize=vec_env,
        log_interval=args.log_interval,
        verbose=1,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=args.save_path,
    )

    model.save(args.save_path)
    vec_env.save(args.save_path + "_vecnorm.pkl")
    print(f"\nSaved model to {args.save_path}.zip")
    print(f"Saved VecNormalize to {args.save_path}_vecnorm.pkl")


if __name__ == "__main__":
    main()
