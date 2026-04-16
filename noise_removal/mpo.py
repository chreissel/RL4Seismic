"""
Maximum a Posteriori Policy Optimization (MPO) for continuous control.

Implements the MPO algorithm from Abdolmaleki et al. (2018),
"Maximum a Posteriori Policy Optimisation" (arXiv:1806.06920),
which is the RL algorithm used in the Deep Loop Shaping paper
(arXiv:2509.14016, DeepMind / Caltech, Nature 2025).

MPO alternates between two steps:

  **E-step** — find a non-parametric improved policy q(a|s) by solving
  a KL-constrained optimisation:

      q*(a|s) ∝ π_old(a|s) · exp(Q(s,a) / η)

  where η is the temperature found by minimising the convex dual:

      g(η) = η·ε + η·log( 1/K · Σ_k exp(Q(s,a_k) / η) )

  ε is the KL bound (hyperparameter), {a_k} are sampled from π_old.

  **M-step** — fit the parametric policy π_θ to approximate q* via
  weighted maximum likelihood, with separate KL constraints on the
  mean and variance to prevent premature collapse:

      max  Σ_k w_k · log π_θ(a_k | s)
      s.t. E[ KL_mean(π_old ∥ π_θ) ] ≤ ε_μ
           E[ KL_var (π_old ∥ π_θ) ] ≤ ε_σ

  The constraints are enforced via dual variables α_μ, α_σ updated
  with gradient ascent on the Lagrangian.

The critic uses twin Q-functions with soft target updates (like SAC/TD3).

This implementation accepts any ``BaseFeaturesExtractor`` from SB3
(e.g. ``BilinearDilatedExtractor``) and works with SB3's ``ReplayBuffer``
and ``VecNormalize``.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------

class GaussianActor(nn.Module):
    """Diagonal Gaussian policy: obs → features → (μ, log σ)."""

    def __init__(
        self,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        action_dim: int,
        net_arch: list[int],
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers: list[nn.Module] = []
        in_dim = features_dim
        for h in net_arch:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

    def forward(self, obs: torch.Tensor):
        features = self.features_extractor(obs)
        h = self.mlp(features)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean, log_std = self(obs)
        return Normal(mean, log_std.exp())

    def sample(self, obs: torch.Tensor, n: int = 64) -> torch.Tensor:
        """Sample n actions per observation.  Returns (batch, n, act_dim)."""
        dist = self.get_dist(obs)
        # dist.rsample((n,)) → (n, batch, act_dim)
        return dist.rsample((n,)).permute(1, 0, 2)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Log-prob of actions.  actions: (batch, [n], act_dim) → (batch, [n])."""
        dist = self.get_dist(obs)
        return dist.log_prob(actions).sum(dim=-1)


class QNetwork(nn.Module):
    """Q-function: (obs, action) → scalar Q-value."""

    def __init__(
        self,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        action_dim: int,
        net_arch: list[int],
    ):
        super().__init__()
        self.features_extractor = features_extractor

        layers: list[nn.Module] = []
        in_dim = features_dim + action_dim
        for h in net_arch:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        x = torch.cat([features, actions], dim=-1)
        return self.mlp(x).squeeze(-1)


class TwinQNetwork(nn.Module):
    """Twin Q-functions for pessimistic value estimation."""

    def __init__(
        self,
        features_extractor_class: type,
        features_extractor_kwargs: dict,
        observation_space,
        action_dim: int,
        net_arch: list[int],
    ):
        super().__init__()
        fe1 = features_extractor_class(observation_space, **features_extractor_kwargs)
        fe2 = features_extractor_class(observation_space, **features_extractor_kwargs)
        features_dim = fe1.features_dim
        self.q1 = QNetwork(fe1, features_dim, action_dim, net_arch)
        self.q2 = QNetwork(fe2, features_dim, action_dim, net_arch)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        return self.q1(obs, actions), self.q2(obs, actions)

    def min_q(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs, actions)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# MPO Algorithm
# ---------------------------------------------------------------------------

class MPO:
    """Maximum a Posteriori Policy Optimization.

    Parameters
    ----------
    observation_space, action_space : gymnasium spaces
    features_extractor_class : type
        SB3-compatible feature extractor (e.g. BilinearDilatedExtractor).
    features_extractor_kwargs : dict
        Kwargs for the feature extractor constructor.
    net_arch : list[int]
        Hidden layer sizes for actor and critic MLPs.
    lr_actor, lr_critic : float
        Learning rates.
    lr_dual : float
        Learning rate for the dual variables (η, α_μ, α_σ).
    epsilon : float
        KL bound for the E-step (non-parametric improvement).
    epsilon_mean : float
        KL bound on the policy mean in the M-step.
    epsilon_var : float
        KL bound on the policy variance in the M-step.
    n_action_samples : int
        Number of action samples K per state in the E-step.
    gamma : float
        Discount factor.
    tau : float
        Soft target update rate (Polyak averaging).
    batch_size : int
    buffer_size : int
    learning_starts : int
        Number of env steps before training begins.
    train_freq : int
        Train every N env steps.
    gradient_steps : int
        Number of gradient steps per train call.
    device : str
    """

    def __init__(
        self,
        observation_space,
        action_space,
        features_extractor_class: type,
        features_extractor_kwargs: dict,
        net_arch: list[int] = (256, 256),
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_dual: float = 1e-2,
        epsilon: float = 0.1,
        epsilon_mean: float = 0.01,
        epsilon_var: float = 1e-4,
        n_action_samples: int = 64,
        gamma: float = 0.999,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 200_000,
        learning_starts: int = 1000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.observation_space = observation_space
        self.action_space = action_space
        act_dim = action_space.shape[0]
        act_low = torch.tensor(action_space.low, dtype=torch.float32, device=self.device)
        act_high = torch.tensor(action_space.high, dtype=torch.float32, device=self.device)
        self._act_low = act_low
        self._act_high = act_high

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.n_action_samples = n_action_samples
        self.epsilon = epsilon
        self.epsilon_mean = epsilon_mean
        self.epsilon_var = epsilon_var

        net_arch = list(net_arch)

        # --- Actor ---
        actor_fe = features_extractor_class(observation_space, **features_extractor_kwargs)
        features_dim = actor_fe.features_dim
        self.actor = GaussianActor(
            actor_fe, features_dim, act_dim, net_arch,
        ).to(self.device)

        # --- Critic (twin Q) ---
        self.critic = TwinQNetwork(
            features_extractor_class, features_extractor_kwargs,
            observation_space, act_dim, net_arch,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)

        # --- Dual variables (learnable, optimised on log scale) ---
        self.log_eta = nn.Parameter(torch.tensor(1.0, device=self.device))
        self.log_alpha_mean = nn.Parameter(torch.tensor(1.0, device=self.device))
        self.log_alpha_var = nn.Parameter(torch.tensor(1.0, device=self.device))

        # --- Optimisers ---
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.dual_optimizer = torch.optim.Adam(
            [self.log_eta, self.log_alpha_mean, self.log_alpha_var], lr=lr_dual
        )

        # --- Replay buffer (SB3) ---
        from stable_baselines3.common.buffers import ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            buffer_size, observation_space, action_space,
            device=self.device, n_envs=1,
        )

        self.num_timesteps = 0
        self._n_updates = 0

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def train_step(self):
        """One gradient update: critic + E-step + M-step + dual update."""
        data = self.replay_buffer.sample(self.batch_size)
        obs = data.observations
        actions = data.actions
        rewards = data.rewards.squeeze(-1)
        next_obs = data.next_observations
        dones = data.dones.squeeze(-1)

        # ---------- Critic update (TD target with target network) ----------
        with torch.no_grad():
            next_actions = self.actor.sample(next_obs, n=1).squeeze(1)
            next_actions = next_actions.clamp(self._act_low, self._act_high)
            q_target = self.critic_target.min_q(next_obs, next_actions)
            td_target = rewards + self.gamma * (1.0 - dones) * q_target

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------- E-step: find non-parametric improved policy ----------
        with torch.no_grad():
            # Sample K actions per state from the current (old) policy
            sampled_actions = self.actor.sample(obs, n=self.n_action_samples)
            # (batch, K, act_dim)
            sampled_actions = sampled_actions.clamp(self._act_low, self._act_high)

            # Evaluate Q for each (s, a_k) pair
            B, K, A = sampled_actions.shape
            obs_rep = obs.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
            act_flat = sampled_actions.reshape(B * K, A)
            q_values = self.critic_target.min_q(obs_rep, act_flat).reshape(B, K)

        # Dual optimisation for temperature η (clamped to prevent collapse)
        eta = self.log_eta.exp().clamp(min=1e-2)
        # g(η) = η·ε + η·logsumexp(Q/η) − η·log(K)
        dual_loss_eta = (
            eta * self.epsilon
            + eta * (torch.logsumexp(q_values / eta, dim=1) - np.log(K)).mean()
        )

        # Compute normalised weights w_k = softmax(Q_k / η)
        with torch.no_grad():
            weights = F.softmax(q_values / eta.detach(), dim=1)  # (B, K)

        # ---------- M-step: weighted MLE with KL constraints ----------
        # Get old policy parameters (detached)
        with torch.no_grad():
            old_mean, old_log_std = self.actor(obs)
            old_std = old_log_std.exp()

        # Get new policy parameters
        new_mean, new_log_std = self.actor(obs)
        new_std = new_log_std.exp()

        # Weighted log-likelihood of sampled actions under new policy
        new_dist = Normal(new_mean.unsqueeze(1), new_std.unsqueeze(1))
        log_probs = new_dist.log_prob(sampled_actions).sum(dim=-1)  # (B, K)
        policy_loss = -(weights.detach() * log_probs).sum(dim=1).mean()

        # Decoupled KL constraints (mean and variance separately)
        # KL_mean = 0.5 * Σ_d (μ_new - μ_old)² / σ_old²
        kl_mean = 0.5 * (((new_mean - old_mean) / old_std) ** 2).sum(dim=-1).mean()
        # KL_var = 0.5 * Σ_d (σ_old²/σ_new² + log(σ_new²/σ_old²) - 1)
        var_ratio = (old_std / new_std) ** 2
        kl_var = 0.5 * (var_ratio + 2 * (new_log_std - old_log_std) - 1).sum(dim=-1).mean()

        alpha_mean = self.log_alpha_mean.exp()
        alpha_var = self.log_alpha_var.exp()

        # Actor loss with Lagrangian KL penalties
        actor_loss = (
            policy_loss
            + alpha_mean.detach() * kl_mean
            + alpha_var.detach() * kl_var
        )

        # Dual loss for α_μ, α_σ (enforce constraints)
        dual_loss_alpha = (
            alpha_mean * (self.epsilon_mean - kl_mean.detach())
            + alpha_var * (self.epsilon_var - kl_var.detach())
        )

        # ---------- Optimise ----------
        # Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Dual variables (η, α_μ, α_σ)
        total_dual_loss = dual_loss_eta - dual_loss_alpha
        self.dual_optimizer.zero_grad()
        total_dual_loss.backward()
        self.dual_optimizer.step()

        # ---------- Soft target update ----------
        with torch.no_grad():
            for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_tgt.data.lerp_(p.data, self.tau)

        self._n_updates += 1
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": policy_loss.item(),
            "eta": eta.item(),
            "alpha_mean": alpha_mean.item(),
            "alpha_var": alpha_var.item(),
            "kl_mean": kl_mean.item(),
            "kl_var": kl_var.item(),
        }

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Predict action for a single observation."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            if deterministic:
                mean, _ = self.actor(obs_t)
                action = mean
            else:
                dist = self.actor.get_dist(obs_t)
                action = dist.sample()
            action = action.clamp(self._act_low, self._act_high)
            return action.cpu().numpy().squeeze(0)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def learn(
        self,
        total_timesteps: int,
        env,
        vec_normalize=None,
        callback=None,
        log_interval: int = 1,
        verbose: int = 1,
        checkpoint_freq: int = 0,
        checkpoint_path: str | None = None,
    ):
        """Train for ``total_timesteps`` environment steps.

        Parameters
        ----------
        env : VecEnv (SB3-compatible, single env)
        vec_normalize : VecNormalize wrapper (for obs/reward normalisation)
        callback : optional callable(locals, globals) called each episode end
        log_interval : print episode stats every N episodes
        verbose : verbosity level (0=silent, 1=episode logs)
        checkpoint_freq : save a checkpoint every N env steps (0 = disabled)
        checkpoint_path : base path for checkpoints; files are saved as
            ``<checkpoint_path>_<steps>_steps.zip`` (model) and
            ``<checkpoint_path>_vecnormalize_<steps>_steps.pkl`` (VecNormalize).
        """
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        n_episodes = 0
        ep_rewards: list[float] = []
        next_checkpoint = checkpoint_freq if checkpoint_freq > 0 else total_timesteps + 1

        for step in range(total_timesteps):
            self.num_timesteps = step + 1

            # Select action
            if step < self.learning_starts:
                action = np.array([self.action_space.sample()])
            else:
                obs_np = obs[0] if obs.ndim > 1 else obs
                action = self.predict(obs_np, deterministic=False)
                action = np.array([action])

            # Step environment
            new_obs, reward, done, info = env.step(action)

            # Store transition (with unnormalised obs if VecNormalize is used)
            real_obs = obs
            real_new_obs = new_obs
            if vec_normalize is not None:
                real_obs = vec_normalize.get_original_obs()
                real_new_obs = vec_normalize.get_original_obs()

            self.replay_buffer.add(
                real_obs, real_new_obs, action, reward, done, [info[0]] if isinstance(info, list) else [info],
            )

            obs = new_obs
            episode_reward += float(reward[0])
            episode_length += 1

            if done[0]:
                n_episodes += 1
                ep_rewards.append(episode_reward)
                if verbose >= 1 and n_episodes % log_interval == 0:
                    recent = ep_rewards[-10:]
                    mean_r = np.mean(recent)
                    print(f"Episode {n_episodes:4d} | "
                          f"steps {self.num_timesteps:>7,} | "
                          f"ep_rew {episode_reward:>10.1f} | "
                          f"mean_rew(10) {mean_r:>10.1f} | "
                          f"eta {self.log_eta.exp().item():.3f} | "
                          f"n_updates {self._n_updates}")
                episode_reward = 0.0
                episode_length = 0
                obs = env.reset()

                if callback is not None:
                    callback(locals(), globals())

            # Train
            if step >= self.learning_starts and step % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    self.train_step()

            # Checkpoint
            if self.num_timesteps >= next_checkpoint and checkpoint_path is not None:
                ckpt = f"{checkpoint_path}_{self.num_timesteps}_steps"
                self.save(ckpt)
                if vec_normalize is not None:
                    vec_normalize.save(f"{checkpoint_path}_vecnormalize_{self.num_timesteps}_steps.pkl")
                if verbose >= 1:
                    print(f"  [checkpoint] saved {ckpt}.zip")
                next_checkpoint += checkpoint_freq

        return self

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save model parameters to ``path``.zip."""
        data = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "dual_optimizer": self.dual_optimizer.state_dict(),
            "log_eta": self.log_eta.data.cpu(),
            "log_alpha_mean": self.log_alpha_mean.data.cpu(),
            "log_alpha_var": self.log_alpha_var.data.cpu(),
            "num_timesteps": self.num_timesteps,
            "n_updates": self._n_updates,
        }
        if not path.endswith(".zip"):
            path = path + ".zip"
        torch.save(data, path)

    def load(self, path: str):
        """Load model parameters from ``path``."""
        if not path.endswith(".zip"):
            path = path + ".zip"
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(data["actor_state_dict"])
        self.critic.load_state_dict(data["critic_state_dict"])
        self.critic_target.load_state_dict(data["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        self.dual_optimizer.load_state_dict(data["dual_optimizer"])
        self.log_eta.data = data["log_eta"].to(self.device)
        self.log_alpha_mean.data = data["log_alpha_mean"].to(self.device)
        self.log_alpha_var.data = data["log_alpha_var"].to(self.device)
        self.num_timesteps = data.get("num_timesteps", 0)
        self._n_updates = data.get("n_updates", 0)
        return self
