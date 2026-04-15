"""
SAC training with NLMS behavioural-cloning warmup (Option B).

This is an alternative training entry point to ``train_sac.py``.  The
problem with vanilla SAC on the seismic noise-cancellation task is that
the policy has to *discover* cancellation via random exploration before
its replay buffer contains any useful trajectories.  In practice this
either fails (broadband-reward run plateaus below no-op) or gets stuck in
a local optimum well below NLMS (in-band-reward Option A run).

The fix: **warm-start the SAC actor by imitating the NLMS adaptive filter**
on a training episode.  After BC pretraining the actor already produces
NLMS-equivalent actions, so when SAC's reinforcement-learning phase
begins it only has to *improve* on NLMS rather than discover cancellation
from scratch.

Pipeline
--------

  Phase 1: BC dataset collection
    Run NLMS forward on N seismic episodes; record (observation, action)
    pairs at each timestep, where the observation includes the residual
    window built from NLMS's own past actions (so the obs distribution
    matches what the agent will see when it starts behaving like NLMS).

  Phase 2: BC pretraining
    Supervised MSE loss between the SAC actor's deterministic output
    (squashed mean action) and the NLMS target action, for ``--bc-epochs``
    passes over the dataset.  Trains both the bilinear feature extractor
    and the actor head end-to-end.

  Phase 3: SAC RL training
    Standard SAC ``model.learn()`` for ``--timesteps`` env steps.  The
    critic starts from random init and learns the value function from
    the (BC-warmed) policy's transitions.  Once the critic catches up,
    the actor refines the BC policy to maximise the in-band reward.

Usage
-----
    # Full Option B training (single GPU, ~2 hours)
    python train_sac_bc.py \\
        --timesteps 60000 \\
        --episode-duration 600 \\
        --conv-channels 24 --conv-layers 6 --bilinear-rank 16 \\
        --features-dim 192 --batch-size 256 \\
        --bc-episodes 8 --bc-epochs 20 \\
        --save-path models/sac_bilinear_bc

    # CPU run (slower, for laptops)
    python train_sac_bc.py --device cpu --bc-episodes 4 --timesteps 30000

Compatible with the same evaluation script as ``train_sac.py``:

    python scripts/rl_showcase.py --model-path models/sac_bilinear_bc \\
        --duration 1800 --drift-scale 2.0 --save-dir results/rl_showcase_bc
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from baselines import LMSFilter
from noise_removal import (
    NoiseCancellationEnv,
    SeismicConfig,
    SeismicSignalSimulator,
)
from noise_removal.bilinear_policy import BilinearDilatedExtractor


# ---------------------------------------------------------------------------
# Action-smoothness wrapper (duplicated from train_sac.py to keep this script
# self-contained — that file is being modified/run in parallel).
# ---------------------------------------------------------------------------


class ActionSmoothnessWrapper(gym.Wrapper):
    """Reward shaping: subtract λ·(a_t − a_{t-1})² from the env reward."""

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


# ---------------------------------------------------------------------------
# BC dataset collection
# ---------------------------------------------------------------------------


def _build_seismic_config(args) -> SeismicConfig:
    return SeismicConfig(
        drift=True,
        tilt_coupling=True,
        sensor_noise_exponent=2.0,
        sensor_noise_band=(0.05, 0.5),
        gain_drift_sigma=0.1 * args.drift_scale,
        freq_drift_sigma=0.01 * args.drift_scale,
        t2l_gain_drift_sigma=2.7 * args.drift_scale,
    )


def collect_bc_dataset(
    cfg: SeismicConfig,
    n_episodes: int,
    episode_duration: float,
    window_size: int,
    action_clip: float,
    warmup_skip: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Roll NLMS forward on ``n_episodes`` independent seismic episodes and
    return the (observation, NLMS-action) pairs in a flat array.

    The observation at time t is exactly what the SAC env would produce
    if the agent's past actions had been NLMS's actions:

        obs_t = [witness_x[t-W+1..t], witness_y[t-W+1..t], residual[t-W..t-1]]

    where ``residual[k] = main[k] - nlms_action[k]``.  Building the obs
    against NLMS's *own* residuals (rather than zero-action residuals) is
    important: it ensures the BC policy is trained on the same observation
    distribution that an NLMS-equivalent policy would induce in the env,
    so the BC-to-RL handover is in-distribution.

    The first ``warmup_skip`` samples per episode are discarded — NLMS has
    a convergence transient at the episode start that pollutes the BC
    targets if not cut.
    """
    obs_list = []
    action_list = []

    for ep in range(n_episodes):
        sim = SeismicSignalSimulator(cfg, seed=seed + ep)
        data = sim.generate_episode(duration=episode_duration, signal_amplitude=0.0)
        n = len(data["main"])

        # 1. Run NLMS forward to get its actions
        nlms = LMSFilter(filter_length=window_size, step_size=0.5, normalized=True)
        cleaned = nlms.run(
            data["witness_x"], data["main"], witness_y=data["witness_y"]
        )
        # NLMS subtracts its coupling estimate from main; the equivalent
        # "action" in the env's terminology is exactly that estimate:
        nlms_actions = data["main"] - cleaned
        nlms_actions = np.clip(nlms_actions, -action_clip, action_clip)

        # 2. For each step t >= window_size + warmup_skip, build the obs
        wx = data["witness_x"]
        wy = data["witness_y"]
        main = data["main"]

        t_start = window_size + warmup_skip
        for t in range(t_start, n):
            wx_win = wx[t - window_size + 1 : t + 1]
            wy_win = wy[t - window_size + 1 : t + 1]
            main_win = main[t - window_size : t]
            action_win = nlms_actions[t - window_size : t]
            residual_win = (main_win - action_win).astype(np.float32)

            obs = np.concatenate([wx_win, wy_win, residual_win]).astype(np.float32)
            obs_list.append(obs)
            action_list.append(nlms_actions[t])

        print(
            f"  ep {ep + 1}/{n_episodes}: collected {n - t_start} samples "
            f"(NLMS action range "
            f"[{nlms_actions.min():.2f}, {nlms_actions.max():.2f}])"
        )

    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(action_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# BC pretraining
# ---------------------------------------------------------------------------


def bc_pretrain(
    model: SAC,
    vec_env: VecNormalize,
    obs_arr: np.ndarray,
    action_arr: np.ndarray,
    args,
) -> None:
    """
    Pretrain the SAC actor by MSE-regression onto the NLMS targets.

    The actor's deterministic output is  ``tanh(mu(features))``, which is
    in [-1, 1].  We compare it against the NLMS target normalised to the
    same range  ``nlms_action / action_clip``.  Both the bilinear feature
    extractor and the actor heads are updated jointly.

    First, the VecNormalize observation statistics are fit on the BC
    dataset, so the obs scaling matches what the SAC RL phase will see.
    """
    device = model.device
    n = len(obs_arr)
    bs = args.bc_batch_size

    # ---- 1. Fit VecNormalize obs running mean/std on the BC dataset ----
    print(f"  fitting VecNormalize obs stats on {n:,} samples...")
    vec_env.training = True
    chunk = 1024
    for i in range(0, n, chunk):
        vec_env.obs_rms.update(obs_arr[i : i + chunk])
    vec_env.training = False

    # ---- 2. Supervised MSE loop on (obs, NLMS-action) ----
    print(
        f"  pretraining SAC actor for {args.bc_epochs} epochs  "
        f"(batch={bs}, lr={args.bc_learning_rate})..."
    )
    actor = model.policy.actor
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.bc_learning_rate)
    rng = np.random.default_rng(args.seed)

    losses = []
    for epoch in range(args.bc_epochs):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, bs):
            batch_idx = idx[i : i + bs]
            batch_obs = obs_arr[batch_idx]
            batch_acts = action_arr[batch_idx]

            # Normalise observations the same way SAC will at rollout time
            batch_obs_norm = vec_env.normalize_obs(batch_obs).astype(np.float32)

            obs_t = torch.from_numpy(batch_obs_norm).to(device)
            target = torch.from_numpy(batch_acts).to(device).unsqueeze(-1)

            # Deterministic actor output: tanh(mean) in [-1, 1]
            mean_actions, _log_std, _ = actor.get_action_dist_params(obs_t)
            squashed = torch.tanh(mean_actions)

            # Target NLMS action normalised to [-1, 1] (then clamped to
            # avoid tanh-saturation at the boundaries)
            target_norm = torch.clamp(target / args.action_clip, -0.999, 0.999)

            loss = F.mse_loss(squashed, target_norm)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(1, n_batches)
        losses.append(avg)
        # Print every epoch for short runs, every 5 for longer
        log_every = 1 if args.bc_epochs <= 10 else max(1, args.bc_epochs // 10)
        if epoch % log_every == 0 or epoch == args.bc_epochs - 1:
            print(f"    epoch {epoch + 1:3d}/{args.bc_epochs}  "
                  f"mse_norm={avg:.5f}  "
                  f"(rms_target_norm={float(target_norm.std()):.3f})")

    actor.eval()
    print(f"  BC done. final loss = {losses[-1]:.5f}")


# ---------------------------------------------------------------------------
# Replay-buffer prefill: critic warmup by inserting real NLMS rollouts
# ---------------------------------------------------------------------------


def prefill_replay_buffer(
    model: SAC,
    args,
    n_episodes: int,
    seed: int,
) -> int:
    """
    Roll NLMS through the actual env and push (obs, next_obs, action,
    reward, done) transitions into the SAC replay buffer.

    This solves the **critic-catch-up problem** that plagues naive BC →
    SAC handoffs: without prefill, SAC's critic starts from a random
    initialisation, so its gradient on the actor pulls the BC-warmed
    policy toward maximising a meaningless value function — typically
    degrading the BC initialisation in the first few RL episodes.  With
    prefill, the critic's very first gradient updates train against a
    buffer full of NLMS-quality transitions and the real env reward, so
    Q(obs, action) starts converging from the right basin.

    The transitions are added with **unnormalised** observations and
    rewards (matching SAC's internal convention — VecNormalize stats are
    applied at sample time, not at insertion time).
    """
    print(f"  prefilling replay buffer with {n_episodes} NLMS episodes…")
    cfg = _build_seismic_config(args)
    n_added = 0

    for ep in range(n_episodes):
        # Use a fresh raw env so the rollout matches what SAC will see at
        # inference time (no VecNormalize wrapping → unnormalised obs/reward).
        raw_env = NoiseCancellationEnv(
            config=cfg,
            window_size=args.window_size,
            episode_duration=args.episode_duration,
            action_clip=args.action_clip,
            freq_reward=args.freq_reward,
            freq_band_low=args.freq_band_low,
            freq_band_high=args.freq_band_high,
        )
        env_w = ActionSmoothnessWrapper(
            raw_env, smoothness_lambda=args.smoothness_lambda
        )
        obs, _ = env_w.reset(seed=seed + ep)

        # Pull the simulator data out of the inner env so we can compute
        # the NLMS action sequence offline up-front.
        data = raw_env._data
        nlms = LMSFilter(
            filter_length=args.window_size, step_size=0.5, normalized=True
        )
        cleaned = nlms.run(
            data["witness_x"], data["main"], witness_y=data["witness_y"]
        )
        nlms_actions = np.clip(
            data["main"] - cleaned, -args.action_clip, args.action_clip
        )

        # Step the env with the precomputed NLMS actions and store
        # transitions in the SAC replay buffer (n_envs=1 layout).
        t = args.window_size
        done = False
        ep_count = 0
        while not done:
            action = np.array([nlms_actions[t]], dtype=np.float32)
            next_obs, reward, term, trunc, _info = env_w.step(action)
            done = bool(term or trunc)

            model.replay_buffer.add(
                obs[None, :].astype(np.float32),
                next_obs[None, :].astype(np.float32),
                action[None, :],
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=bool),
                [{}],
            )
            n_added += 1
            ep_count += 1

            obs = next_obs
            t += 1

        print(f"    ep {ep + 1}/{n_episodes}: added {ep_count} transitions")

    print(f"  total transitions added: {n_added:,}")
    return n_added


# ---------------------------------------------------------------------------
# Env factory + CLI
# ---------------------------------------------------------------------------


def make_env(args, seed: int):
    def _thunk():
        cfg = _build_seismic_config(args)
        env: gym.Env = NoiseCancellationEnv(
            config=cfg,
            window_size=args.window_size,
            episode_duration=args.episode_duration,
            action_clip=args.action_clip,
            freq_reward=args.freq_reward,
            freq_band_low=args.freq_band_low,
            freq_band_high=args.freq_band_high,
        )
        env = ActionSmoothnessWrapper(env, smoothness_lambda=args.smoothness_lambda)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Training duration / env
    p.add_argument("--timesteps", type=int, default=80_000,
                   help="RL phase steps after BC warmup (default 80000)")
    p.add_argument("--window-size", type=int, default=240)
    p.add_argument("--episode-duration", type=float, default=600.0)

    # Coupling regime (matches Option A defaults — fixed drift, no DR)
    p.add_argument("--drift-scale", type=float, default=2.0)

    # Network
    p.add_argument("--conv-channels", type=int, default=24)
    p.add_argument("--conv-layers", type=int, default=6)
    p.add_argument("--bilinear-rank", type=int, default=16)
    p.add_argument("--features-dim", type=int, default=192)

    # SAC RL hyperparameters
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=256,
                   help="Smaller than train_sac.py default (1000) because the BC "
                        "warmup means we want SAC to start using its actor "
                        "immediately, not random exploration. Just enough to fill "
                        "one batch before gradient updates begin.")
    p.add_argument("--target-entropy", type=float, default=-0.2)
    p.add_argument("--smoothness-lambda", type=float, default=0.01)
    p.add_argument("--action-clip", type=float, default=15.0)

    # Reward
    p.add_argument("--freq-reward", action="store_true", default=True)
    p.add_argument("--no-freq-reward", dest="freq_reward", action="store_false")
    p.add_argument("--freq-band-low", type=float, default=0.05)
    p.add_argument("--freq-band-high", type=float, default=0.5)

    # BC-specific
    p.add_argument("--bc-episodes", type=int, default=8,
                   help="Number of NLMS rollout episodes for the BC dataset.")
    p.add_argument("--bc-episode-duration", type=float, default=600.0,
                   help="Duration (s) of each BC dataset episode.")
    p.add_argument("--bc-warmup-skip", type=int, default=600,
                   help="Drop the first N samples of each NLMS rollout to skip "
                        "the convergence transient (default 600 = 150 s @ 4 Hz).")
    p.add_argument("--bc-epochs", type=int, default=20)
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--bc-learning-rate", type=float, default=3e-4)
    p.add_argument("--skip-bc", action="store_true",
                   help="Skip BC and go straight to RL (ablation control).")

    # Replay-buffer prefill
    p.add_argument("--prefill-buffer", action="store_true", default=True,
                   help="Prefill the SAC replay buffer with NLMS rollouts so "
                        "the critic warm-starts from real transitions instead "
                        "of random init (default: on).  Critical for avoiding "
                        "the BC-warmed actor degrading in the first few RL "
                        "episodes due to a meaningless critic gradient.")
    p.add_argument("--no-prefill-buffer", dest="prefill_buffer",
                   action="store_false")
    p.add_argument("--prefill-episodes", type=int, default=4,
                   help="Number of NLMS episodes to push into the replay "
                        "buffer before RL training begins (default 4).")

    # Misc
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--save-path", default="models/sac_bilinear_bc")
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--checkpoint-freq", type=int, default=10_000,
                   help="Save an intermediate checkpoint every N env steps of "
                        "the RL phase (default 10000; set to 0 to disable).  "
                        "Each checkpoint writes <save_path>_<steps>_steps.zip "
                        "(model) and <save_path>_vecnormalize_<steps>_steps.pkl "
                        "(VecNormalize stats) so every intermediate model can "
                        "be evaluated independently with scripts/rl_showcase.py.")
    p.add_argument("--device", default="auto",
                   help="torch device ('auto', 'cpu', 'cuda', 'cuda:1', ...).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # ---- Build env + SAC ----
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
        device=args.device,
    )

    print()
    print("=" * 72)
    print("  Option B: SAC + bilinear features + NLMS BC warmup")
    print("=" * 72)
    print(f"  drift_scale     : {args.drift_scale}")
    print(f"  episode duration: {args.episode_duration:.0f} s")
    print(f"  window_size     : {args.window_size}")
    print(f"  reward          : "
          f"{'band-limited [' + str(args.freq_band_low) + ', ' + str(args.freq_band_high) + '] Hz' if args.freq_reward else 'broadband y_t^2 - e_t^2'}")
    print(f"  action_clip     : ±{args.action_clip}")
    print(f"  target entropy  : {args.target_entropy}")
    print(f"  RL timesteps    : {args.timesteps:,}")
    print(f"  device          : {model.device}")
    print(f"  save path       : {args.save_path}")
    print("-" * 72)

    # =========================================================================
    # Phase 1: BC dataset collection
    # =========================================================================
    if not args.skip_bc:
        print("\n[1/3] BC dataset collection (rolling NLMS forward)…")
        t0 = time.time()
        cfg = _build_seismic_config(args)
        obs_arr, action_arr = collect_bc_dataset(
            cfg,
            n_episodes=args.bc_episodes,
            episode_duration=args.bc_episode_duration,
            window_size=args.window_size,
            action_clip=args.action_clip,
            warmup_skip=args.bc_warmup_skip,
            seed=args.seed + 100,
        )
        print(
            f"  total: {len(obs_arr):,} (obs, action) pairs  "
            f"({time.time() - t0:.1f} s)"
        )
        print(
            f"  obs shape: {obs_arr.shape}  "
            f"action range: [{action_arr.min():.2f}, {action_arr.max():.2f}]  "
            f"std: {action_arr.std():.2f}"
        )

        # =====================================================================
        # Phase 2: BC pretraining
        # =====================================================================
        print("\n[2/3] BC pretraining (supervised MSE on actor)…")
        t0 = time.time()
        bc_pretrain(model, vec_env, obs_arr, action_arr, args)
        print(f"  BC phase: {time.time() - t0:.1f} s")
        # Free the dataset memory before RL phase
        del obs_arr, action_arr

        # Save a "post-BC, pre-RL" checkpoint so the BC contribution can be
        # evaluated on its own (ablation: "what does NLMS imitation alone
        # buy us?")
        bc_only_path = args.save_path + "_bc_only"
        model.save(bc_only_path)
        vec_env.save(bc_only_path + "_vecnorm.pkl")
        print(f"  saved BC-only checkpoint: {bc_only_path}.zip")
    else:
        print("\n[1-2/3] BC phases SKIPPED (--skip-bc)")

    # =========================================================================
    # Phase 2.5: Replay-buffer prefill (critic warmup)
    # =========================================================================
    if args.prefill_buffer:
        print("\n[2.5/3] Replay-buffer prefill (critic warmup)…")
        t0 = time.time()
        prefill_replay_buffer(
            model, args,
            n_episodes=args.prefill_episodes,
            seed=args.seed + 200,
        )
        print(f"  prefill phase: {time.time() - t0:.1f} s")

    # =========================================================================
    # Phase 3: SAC RL training
    # =========================================================================
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

    print(f"\n[3/3] SAC RL training for {args.timesteps:,} steps…")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks if callbacks else None,
        log_interval=args.log_interval,
    )

    model.save(args.save_path)
    vec_env.save(args.save_path + "_vecnorm.pkl")
    print(
        f"\nSaved final model to {args.save_path}.zip and "
        f"VecNormalize to {args.save_path}_vecnorm.pkl"
    )


if __name__ == "__main__":
    main()
