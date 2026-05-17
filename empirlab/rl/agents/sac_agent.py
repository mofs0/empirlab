"""
Soft Actor-Critic (SAC) Agent
================================
Reference: Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
           Soft actor-critic: Off-policy maximum entropy deep reinforcement
           learning with a stochastic actor. arXiv:1801.01290.

Status: API stub — use stable-baselines3.SAC for production.
"""
from __future__ import annotations


class SACAgent:
    """SAC for continuous-action portfolio environments.

    Parameters
    ----------
    env    : gym-compatible environment with continuous action space
    hidden : int, default 256
    lr     : float, default 3e-4
    alpha  : float, default 0.2   entropy temperature
    tau    : float, default 0.005  soft update coefficient
    """

    def __init__(self, env, hidden=256, lr=3e-4, alpha=0.2, tau=0.005):
        self.env    = env
        self.hidden = hidden
        self.lr     = lr
        self.alpha  = alpha
        self.tau    = tau

    def train(self, total_timesteps: int = 100_000):
        raise NotImplementedError(
            "SACAgent coming in v0.2.\n"
            "Quick start:\n"
            "  pip install stable-baselines3\n"
            "  from stable_baselines3 import SAC"
        )

    def predict(self, obs):
        raise NotImplementedError
