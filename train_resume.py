"""
Resume training from the latest checkpoint and run for a fixed number of
additional steps.  Designed for short-burst training when long background
processes get killed.

Usage
-----
    python train_resume.py --checkpoint models/ppo_noise_cancellation_100000_steps \\
                           --extra-steps 200000

    python train_resume.py --checkpoint models/ppo_noise_cancellation_100000_steps \\
                           --extra-steps 200000 --multi-source --tilt-coupling
"""

import argparse
import os
import re

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from noise_removal import NoiseCancellationEnv, SeismicConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint zip (without .zip)")
    p.add_argument("--extra-steps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--window-size", type=int, default=None,
                   help="Observation window in samples (default: 240 = 60 s @ 4 Hz)")
    p.add_argument("--episode-duration", type=float, default=300.0)
    p.add_argument("--multi-source", action="store_true")
    p.add_argument("--regime-changes", action="store_true")
    p.add_argument("--tilt-coupling", action="store_true")
    return p.parse_args()


def find_vecnorm(checkpoint_path):
    """Find the VecNormalize file matching a checkpoint path."""
    base = os.path.basename(checkpoint_path)
    dirp = os.path.dirname(checkpoint_path) or "models"

    m = re.search(r"_(\d+)_steps$", base)
    if m:
        steps = m.group(1)
        prefix = base[: base.rindex(f"_{steps}_steps")]
        candidate = os.path.join(dirp, f"{prefix}_vecnormalize_{steps}_steps.pkl")
        if os.path.exists(candidate):
            return candidate

    candidate2 = checkpoint_path + "_vecnorm.pkl"
    if os.path.exists(candidate2):
        return candidate2

    raise FileNotFoundError(f"Cannot find VecNormalize for {checkpoint_path}")


def make_env(config, window_size, episode_duration):
    def _init():
        return NoiseCancellationEnv(config=config, window_size=window_size,
                                    episode_duration=episode_duration)
    return _init


def main():
    args = parse_args()

    config = SeismicConfig(
        multi_source=args.multi_source,
        regime_changes=args.regime_changes,
        tilt_coupling=args.tilt_coupling,
    )

    window_size = args.window_size if args.window_size is not None else config.filter_length

    vecnorm_path = find_vecnorm(args.checkpoint)
    print(f"Resuming from : {args.checkpoint}.zip")
    print(f"VecNormalize  : {vecnorm_path}")
    print(f"Extra steps   : {args.extra_steps:,}")

    env_fns = [make_env(config, window_size, args.episode_duration)
               for _ in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = True

    model = RecurrentPPO.load(args.checkpoint + ".zip", env=vec_env)

    base = os.path.basename(args.checkpoint)
    dirp = os.path.dirname(args.checkpoint) or "models"
    m = re.search(r"_(\d+)_steps$", base)
    prefix = base[: base.rindex(f"_{m.group(1)}_steps")] if m else base

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=dirp,
        name_prefix=prefix,
        save_vecnormalize=True,
        verbose=1,
    )

    model.learn(total_timesteps=args.extra_steps,
                callback=checkpoint_cb,
                reset_num_timesteps=False,
                progress_bar=False)

    out = os.path.join(dirp, prefix)
    model.save(out)
    vec_env.save(out + "_vecnorm.pkl")
    print(f"Burst complete. Saved to {out}.zip")


if __name__ == "__main__":
    main()
