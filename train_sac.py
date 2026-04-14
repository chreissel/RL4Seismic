"""
SAC training for seismic noise cancellation with bilinear feature extractor
and domain randomisation.

This is the RL showcase training script for the T2L-bilinear benchmark.
Key design choices:

  - SAC (off-policy, entropy-regularised actor-critic).  More sample-
    efficient than PPO on low-dimensional continuous control, and its
    deterministic policy gradient is a good fit for a regression-flavoured
    noise-subtraction task.

  - Bilinear dilated-conv feature extractor (see
    noise_removal.bilinear_policy.BilinearDilatedExtractor).  The bilinear
    pooling gives the network the minimum inductive bias required to
    represent the T2L term  T(t)·θ_y(t)·w_x(t).

  - Domain randomisation (see
    noise_removal.domain_randomization.DomainRandomizedNoiseCancellationEnv).
    Each training episode samples fresh drift scale, T2L gain, thermal
    timescale, sensor-noise colour, and regime-change flag.  This forces
    the policy to *adapt* within an episode from observed residuals, since
    there is no single fixed coupling to memorise.

  - Action smoothness regulariser via VecMonitor + reward shaping: we add
    a small penalty  λ·(a_t − a_{t-1})²  to the reward inside a Gym
    wrapper.  Seismic control tasks benefit from smooth actuator outputs.

Usage
-----
    # Short smoke run (quick pipeline check)
    python train_sac.py --timesteps 20_000 --episode-duration 180

    # Full training run
    python train_sac.py --timesteps 1_000_000 --episode-duration 600
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from noise_removal import (
    DomainRandomizedNoiseCancellationEnv,
    DomainRandomizationConfig,
    NoiseCancellationEnv,
    SeismicConfig,
)
from noise_removal.bilinear_policy import BilinearDilatedExtractor


class ActionSmoothnessWrapper(gym.Wrapper):
    """
    Reward shaping: subtract lambda * (a_t - a_{t-1})^2 from the reward
    to penalise high-frequency action chatter, favouring smooth control.
    """

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
            )
        env = ActionSmoothnessWrapper(env, smoothness_lambda=args.smoothness_lambda)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--window-size", type=int, default=240)
    p.add_argument("--episode-duration", type=float, default=300.0)

    p.add_argument("--no-dr", action="store_true",
                   help="Disable domain randomisation (use fixed drift-scale config).")
    p.add_argument("--drift-scale", type=float, default=2.0,
                   help="Drift scale when domain randomisation is off.")
    p.add_argument("--dr-drift-low", type=float, default=0.5)
    p.add_argument("--dr-drift-high", type=float, default=4.0)
    p.add_argument("--dr-regime-prob", type=float, default=0.3)

    p.add_argument("--conv-channels", type=int, default=48)
    p.add_argument("--conv-layers", type=int, default=8)
    p.add_argument("--bilinear-rank", type=int, default=32)
    p.add_argument("--features-dim", type=int, default=256)

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=1_000)

    p.add_argument("--smoothness-lambda", type=float, default=0.01)

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-path", default="models/sac_bilinear")
    p.add_argument("--checkpoint-freq", type=int, default=0,
                   help="Save checkpoint every N steps (0 = disabled).")
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

    policy_kwargs = dict(
        features_extractor_class=BilinearDilatedExtractor,
        features_extractor_kwargs=dict(
            window_size=args.window_size,
            conv_channels=args.conv_channels,
            n_layers=args.conv_layers,
            bilinear_rank=args.bilinear_rank,
            features_dim=args.features_dim,
        ),
        net_arch=[256, 256],
    )

    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        learning_starts=args.learning_starts,
        seed=args.seed,
        verbose=1,
        device="auto",
    )

    print(f"Training SAC for {args.timesteps:,} timesteps  "
          f"(episode = {args.episode_duration:.0f} s, window = {args.window_size})")
    print(f"  Feature extractor: BilinearDilated "
          f"(channels={args.conv_channels}, layers={args.conv_layers}, "
          f"bilinear_rank={args.bilinear_rank})")
    print(f"  Domain randomisation: "
          f"{'OFF (drift_scale=' + str(args.drift_scale) + ')' if args.no_dr else 'ON'}")

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=os.path.dirname(args.save_path) or ".",
            name_prefix=os.path.basename(args.save_path),
        ))

    model.learn(total_timesteps=args.timesteps, callback=callbacks if callbacks else None)

    model.save(args.save_path)
    vec_env.save(args.save_path + "_vecnorm.pkl")
    print(f"Saved model to {args.save_path}.zip and VecNormalize to "
          f"{args.save_path}_vecnorm.pkl")


if __name__ == "__main__":
    main()
