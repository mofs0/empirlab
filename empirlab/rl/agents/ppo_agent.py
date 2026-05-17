"""
Proximal Policy Optimization (PPO) Agent
=========================================
Reference: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.
           (2017). Proximal policy optimization algorithms. arXiv:1707.06347.

Status: API stub — use stable-baselines3.PPO for production.
"""
from __future__ import annotations


class PPOAgent:
    """PPO for financial trading environments.

    Parameters
    ----------
    env         : gym-compatible environment
    hidden_size : int, default 64
    lr          : float, default 3e-4
    gamma       : float, default 0.99
    clip_eps    : float, default 0.2
    n_epochs    : int, default 10
    """

    def __init__(self, env, hidden_size=64, lr=3e-4,
                 gamma=0.99, clip_eps=0.2, n_epochs=10):
        self.env         = env
        self.hidden_size = hidden_size
        self.lr          = lr
        self.gamma       = gamma
        self.clip_eps    = clip_eps
        self.n_epochs    = n_epochs

    def train(self, total_timesteps: int = 100_000):
        raise NotImplementedError(
            "PPOAgent coming in v0.2.\n"
            "Quick start:\n"
            "  pip install stable-baselines3\n"
            "  from stable_baselines3 import PPO\n"
            "  model = PPO('MlpPolicy', env).learn(total_timesteps)"
        )

    def predict(self, obs) -> int:
        raise NotImplementedError
