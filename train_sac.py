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

  - **Band-limited reward (``--freq-reward``, on by default)**.  The env
    computes reward on the bandpass-filtered residual only, so the training
    signal is concentrated on the microseismic [0.05, 0.5] Hz band.  With
    brownian sensor noise this is essential: a broadband reward is
    dominated by uncancelable 1/f^2 power near DC, which drowns out the
    useful in-band gradient and causes the policy to converge to a
    "do-small-actions" local optimum below the no-op baseline.

  - **Tighter action clip (``--action-clip 15``)**.  With y_t ~ 10-20 the
    default ±40 range wastes half the action space on damaging magnitudes
    and slows convergence.

  - **Softer target entropy (``--target-entropy -0.2``)**.  SAC's default
    ``-dim(A)=-1`` drove the auto-tuned entropy coefficient from 1.0 to
    ~7e-4 in 10 episodes, collapsing exploration before the policy found
    the canceling regime.  -0.2 keeps more exploration during training.

Usage
-----
    # Short smoke run (quick pipeline check)
    python train_sac.py --timesteps 20_000 --episode-duration 180

    # Full training run -- Option A recommended config
    python train_sac.py --timesteps 200_000 --episode-duration 600 \\
        --no-dr --drift-scale 2.0 \\
        --conv-channels 24 --conv-layers 6 --bilinear-rank 16 \\
        --features-dim 192 --batch-size 256 \\
        --train-freq 1 --gradient-steps 1 \\
        --save-path models/sac_bilinear_option_a

Option A fixes vs. the vanilla run:
  * freq_reward=True             (band-limited reward on microseismic band)
  * action_clip=15               (down from 40)
  * target_entropy=-0.2          (softer than SAC default -1)
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
                action_clip=args.action_clip,
                freq_reward=args.freq_reward,
                freq_band_low=args.freq_band_low,
                freq_band_high=args.freq_band_high,
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
            # The DR wrapper rebuilds the inner env on every reset, so we
            # need to ensure the underlying NoiseCancellationEnv also gets
            # the freq_reward flag.  Patch the thunk by wrapping reset:
            if args.freq_reward:
                _inner_reset = env.reset

                def _reset_with_freq_reward(*a, **kw):
                    obs, info = _inner_reset(*a, **kw)
                    # Re-instantiate the inner env with freq_reward=True
                    # using the config already sampled by the DR wrapper
                    inner = env._inner
                    inner.freq_reward = True
                    from scipy.signal import butter, sosfilt_zi
                    fs = inner.config.fs
                    nyq = fs / 2.0
                    low = max(args.freq_band_low, 0.01)
                    high = min(args.freq_band_high, nyq * 0.99)
                    inner._bp_sos = butter(
                        4, [low, high], btype="bandpass", fs=fs, output="sos"
                    )
                    zi_template = sosfilt_zi(inner._bp_sos)
                    inner._bp_zi_y = zi_template * 0.0
                    inner._bp_zi_e = zi_template * 0.0
                    return obs, info

                env.reset = _reset_with_freq_reward
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
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=1_000)

    p.add_argument("--smoothness-lambda", type=float, default=0.01)

    # --- Option A fixes: clean reward signal + tighter action + softer entropy ---
    p.add_argument("--action-clip", type=float, default=15.0,
                   help="Symmetric action bound (default 15.0, reduced from 40.0 "
                        "so exploration stays in a sensible range given y_t ~ 10-20).")
    p.add_argument("--freq-reward", action="store_true", default=True,
                   help="Use band-limited reward on [freq_band_low, freq_band_high] "
                        "Hz instead of broadband y_t^2 - e_t^2.  This is critical "
                        "under brownian sensor noise: broadband reward is dominated "
                        "by uncancelable low-frequency 1/f^2 power, which drowns out "
                        "the useful in-band gradient signal (default: on).")
    p.add_argument("--no-freq-reward", dest="freq_reward", action="store_false",
                   help="Disable band-limited reward (fall back to broadband).")
    p.add_argument("--freq-band-low", type=float, default=0.05)
    p.add_argument("--freq-band-high", type=float, default=0.5)

    p.add_argument("--target-entropy", type=float, default=-0.2,
                   help="SAC target entropy.  Default -0.2 (softer than SAC's "
                        "default -dim(A)=-1), which keeps more exploration and "
                        "prevents the entropy coefficient from collapsing to ~1e-3 "
                        "within 10 episodes before the policy has found the "
                        "cancelling regime.  Set to -1.0 to match SAC default.")

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-path", default="models/sac_bilinear")
    p.add_argument("--checkpoint-freq", type=int, default=10_000,
                   help="Save an intermediate checkpoint every N env steps "
                        "(default 10000; set to 0 to disable).  Each checkpoint "
                        "writes <save_path>_<steps>_steps.zip (model) and "
                        "<save_path>_vecnormalize_<steps>_steps.pkl (VecNormalize "
                        "stats) so every intermediate model can be evaluated "
                        "independently with scripts/rl_showcase.py.")
    p.add_argument("--log-interval", type=int, default=1,
                   help="SAC log_interval in episodes (default 1 = every episode).")
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
        target_entropy=args.target_entropy,
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
    print(f"  Reward          : "
          f"{'band-limited [' + str(args.freq_band_low) + ', ' + str(args.freq_band_high) + '] Hz' if args.freq_reward else 'broadband y_t^2 - e_t^2'}")
    print(f"  Action clip     : [-{args.action_clip}, +{args.action_clip}]")
    print(f"  Target entropy  : {args.target_entropy}  "
          f"(SAC default would be -dim(A)=-1.0)")

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=os.path.dirname(args.save_path) or ".",
            name_prefix=os.path.basename(args.save_path),
            save_vecnormalize=True,
        ))
        print(f"  Checkpoints    : every {args.checkpoint_freq:,} steps → "
              f"{os.path.dirname(args.save_path) or '.'}/"
              f"{os.path.basename(args.save_path)}_<steps>_steps.zip "
              f"(+ _vecnormalize_<steps>_steps.pkl)")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks if callbacks else None,
        log_interval=args.log_interval,
    )

    model.save(args.save_path)
    vec_env.save(args.save_path + "_vecnorm.pkl")
    print(f"Saved model to {args.save_path}.zip and VecNormalize to "
          f"{args.save_path}_vecnorm.pkl")


if __name__ == "__main__":
    main()
