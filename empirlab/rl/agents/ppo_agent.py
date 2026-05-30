"""
Proximal Policy Optimization (PPO) Agent
=========================================
Reference: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.
           (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
           https://arxiv.org/abs/1707.06347

Implementation: clip-based PPO with GAE advantage estimation.
Works with any Gym-compatible environment with continuous or discrete actions.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal, Categorical
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _check_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for PPOAgent. "
            "Install with: pip install torch"
        )


# ── Neural Network Architectures ──────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class _ActorContinuous(nn.Module):
    """Gaussian policy for continuous action spaces."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.mu_net  = _MLP(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mu  = self.mu_net(obs)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def act(self, obs):
        dist = self(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class _ActorDiscrete(nn.Module):
    """Categorical policy for discrete action spaces."""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = _MLP(obs_dim, n_actions, hidden)

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)

    def act(self, obs):
        dist = self(obs)
        action = dist.sample()
        return action, dist.log_prob(action)


class _Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = _MLP(obs_dim, 1, hidden)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ── PPO Rollout Buffer ─────────────────────────────────────────────────────────

class _RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones  = [], [], []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_advantages(self, last_value: float, gamma: float, lam: float):
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """Clip-based PPO agent for Gym-compatible environments.

    Parameters
    ----------
    env         : gym environment
    hidden_size : int, default 64      Hidden layer width.
    lr          : float, default 3e-4  Adam learning rate.
    gamma       : float, default 0.99  Discount factor.
    lam         : float, default 0.95  GAE lambda.
    clip_eps    : float, default 0.2   PPO clip epsilon.
    n_epochs    : int, default 4       Update epochs per rollout.
    batch_size  : int, default 64      Mini-batch size.
    rollout_len : int, default 2048    Steps per rollout.
    device      : str or None          'cpu'/'cuda'/None (auto-detect).

    Examples
    --------
    >>> from empirlab.rl.envs.stock_env import StockTradingEnv
    >>> import numpy as np
    >>> prices = 100 * np.exp(np.cumsum(np.random.default_rng(0).standard_normal(500) * 0.01))
    >>> env = StockTradingEnv(prices)
    >>> agent = PPOAgent(env, hidden_size=64)
    >>> history = agent.fit(n_steps=2000, verbose=True)
    """

    def __init__(self, env, hidden_size: int = 64, lr: float = 3e-4,
                 gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, n_epochs: int = 4,
                 batch_size: int = 64, rollout_len: int = 2048,
                 device: Optional[str] = None):
        _check_torch()
        import torch
        self.env = env
        self.gamma, self.lam = gamma, lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_len = rollout_len

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        obs_dim = int(np.prod(env.observation_space.shape))

        # Detect action space type
        try:
            from gymnasium import spaces as gym_spaces
        except ImportError:
            from gym import spaces as gym_spaces

        if isinstance(env.action_space, gym_spaces.Discrete):
            self.continuous = False
            n_act = env.action_space.n
            self.actor = _ActorDiscrete(obs_dim, n_act, hidden_size).to(self.device)
        else:
            self.continuous = True
            act_dim = int(np.prod(env.action_space.shape))
            self.actor  = _ActorContinuous(obs_dim, act_dim, hidden_size).to(self.device)

        self.critic = _Critic(obs_dim, hidden_size).to(self.device)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        self.buffer = _RolloutBuffer()

    # ------------------------------------------------------------------
    def _to_tensor(self, x):
        import torch
        return torch.FloatTensor(np.asarray(x)).to(self.device)

    @torch.no_grad()
    def _policy(self, obs_np: np.ndarray):
        obs_t = self._to_tensor(obs_np).unsqueeze(0)
        action_t, log_prob_t = self.actor.act(obs_t)
        value_t = self.critic(obs_t)
        action_np = action_t.squeeze(0).cpu().numpy()
        return action_np, float(log_prob_t.item()), float(value_t.item())

    # ------------------------------------------------------------------
    def _update(self):
        import torch
        adv, ret = self.buffer.compute_advantages(0.0, self.gamma, self.lam)

        obs_t   = self._to_tensor(np.array(self.buffer.obs))
        act_t   = self._to_tensor(np.array(self.buffer.actions))
        lp_old  = self._to_tensor(np.array(self.buffer.log_probs))
        adv_t   = self._to_tensor(adv)
        ret_t   = self._to_tensor(ret)
        adv_t   = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        T = len(obs_t)
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.n_epochs):
            idx = np.random.permutation(T)
            for start in range(0, T, self.batch_size):
                b = idx[start: start + self.batch_size]
                obs_b = obs_t[b]
                act_b = act_t[b]
                lp_b = lp_old[b]
                adv_b = adv_t[b]
                ret_b = ret_t[b]

                dist = self.actor(obs_b)
                if self.continuous:
                    new_lp = dist.log_prob(act_b).sum(-1)
                else:
                    new_lp = dist.log_prob(act_b.long().squeeze(-1))
                entropy = dist.entropy().mean()
                ratio = (new_lp - lp_b).exp()

                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                val = self.critic(obs_b)
                value_loss = 0.5 * (val - ret_b).pow(2).mean()

                loss = policy_loss + value_loss - 0.01 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
                )
                self.optimizer.step()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        self.buffer.reset()
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)

    # ------------------------------------------------------------------
    def fit(self, n_steps: int = 50_000, verbose: bool = True) -> list[dict]:
        """Train the agent for n_steps environment steps.

        Returns
        -------
        list of dict with keys: step, ep_return, policy_loss, value_loss, entropy
        """
        history = []
        obs, _ = self.env.reset()
        ep_return, ep_len = 0.0, 0
        step = 0

        while step < n_steps:
            action, log_prob, value = self._policy(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(obs, action, log_prob, reward, value, float(done))
            obs = next_obs
            ep_return += reward
            ep_len += 1
            step += 1

            if done:
                obs, _ = self.env.reset()
                if verbose:
                    print(f"Step {step:6d} | ep_return={ep_return:.4f} | ep_len={ep_len}")
                ep_return, ep_len = 0.0, 0

            if len(self.buffer.obs) >= self.rollout_len or (done and step >= n_steps):
                pl, vl, ent = self._update()
                history.append({"step": step, "ep_return": ep_return,
                                 "policy_loss": pl, "value_loss": vl, "entropy": ent})

        return history

    # ------------------------------------------------------------------
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return deterministic action for a given observation."""
        import torch
        with torch.no_grad():
            obs_t = self._to_tensor(obs).unsqueeze(0)
            dist  = self.actor(obs_t)
            if self.continuous:
                action = dist.mean
            else:
                action = dist.probs.argmax(-1)
        return action.squeeze(0).cpu().numpy()
