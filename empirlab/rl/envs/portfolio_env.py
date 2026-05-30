"""
Multi-Asset Portfolio Environment
====================================
A Gym-compatible environment for continuous portfolio rebalancing
across multiple stocks simultaneously.

Observation : [price_returns_window (T×N), current_weights (N), cash_ratio (1)]
Action      : Portfolio weights w ∈ Δ^N (simplex, including cash)
Reward      : log portfolio return − transaction_cost × turnover
"""
from __future__ import annotations

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class PortfolioEnv(gym.Env):
    """Continuous-action multi-asset portfolio environment.

    Parameters
    ----------
    price_matrix : ndarray (T, N)
        Historical price matrix. T time steps, N assets.
    window : int, default 20
        Number of past return periods in observation.
    initial_cash : float, default 1.0
        Normalised portfolio value at start.
    transaction_cost : float, default 0.001
        Proportional cost on turnover (0.1% per unit of weight change).
    reward_scaling : float, default 1.0
        Scale factor on step reward.

    Observation Space
    -----------------
    Box of shape (window * N + N + 1,):
        - Flattened normalised return matrix (window × N)
        - Current portfolio weights (N assets, not including cash)
        - Current cash ratio (1 scalar)

    Action Space
    ------------
    Box [0, 1]^(N+1): target weights over N assets + cash.
    The environment normalises them to sum to 1 (softmax-free simplex projection).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> prices = 100 * np.exp(np.cumsum(rng.standard_normal((252, 5)) * 0.01, axis=0))
    >>> env = PortfolioEnv(prices, window=10)
    >>> obs, info = env.reset()
    >>> obs.shape
    (51,)
    >>> done = False
    >>> while not done:
    ...     action = env.action_space.sample()
    ...     obs, reward, terminated, truncated, info = env.step(action)
    ...     done = terminated or truncated
    """

    metadata = {"render_modes": []}

    def __init__(self, price_matrix: np.ndarray, window: int = 20,
                 initial_cash: float = 1.0,
                 transaction_cost: float = 0.001,
                 reward_scaling: float = 1.0):
        super().__init__()
        self.prices = np.asarray(price_matrix, dtype=float)          # (T, N)
        self.T, self.N = self.prices.shape
        self.window = window
        self.initial_cash = initial_cash
        self.tc = transaction_cost
        self.reward_scaling = reward_scaling

        # Log returns (T-1, N)
        self.log_returns = np.log(self.prices[1:] / self.prices[:-1])

        obs_dim = window * self.N + self.N + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: weights for N assets + 1 cash, projected to simplex
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.N + 1,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        ret_window = self.log_returns[self.t - self.window: self.t]  # (window, N)
        # Normalise each asset's returns to zero mean, unit std
        mu = ret_window.mean(0, keepdims=True)
        sd = ret_window.std(0, keepdims=True) + 1e-8
        ret_norm = (ret_window - mu) / sd
        return np.concatenate([
            ret_norm.ravel(),
            self.weights[:-1],    # N asset weights
            self.weights[-1:]     # cash weight
        ]).astype(np.float32)

    def _project_simplex(self, raw: np.ndarray) -> np.ndarray:
        """Project raw action to probability simplex via normalisation."""
        raw = np.clip(raw, 1e-8, None)
        return raw / raw.sum()

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window        # first usable time step
        self.portfolio_value = float(self.initial_cash)
        # Start all-cash
        self.weights = np.zeros(self.N + 1, dtype=float)
        self.weights[-1] = 1.0     # 100% cash
        self.history = [self.portfolio_value]
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float)
        new_weights = self._project_simplex(action)   # (N+1,)

        # Turnover cost
        turnover = np.abs(new_weights - self.weights).sum()
        cost = self.tc * turnover

        # Asset returns for period t → t+1
        r = self.log_returns[self.t]   # (N,)
        port_return = (new_weights[:-1] * r).sum()   # cash earns 0

        # Net log return
        net_log_ret = port_return - cost
        self.portfolio_value *= np.exp(net_log_ret)
        self.history.append(self.portfolio_value)

        # Update weights by price drift then renormalise
        asset_values = new_weights[:-1] * np.exp(r)
        cash_value   = new_weights[-1]
        total = asset_values.sum() + cash_value
        self.weights = np.append(asset_values, cash_value) / total

        self.t += 1
        terminated = self.t >= self.T - 1
        reward = float(net_log_ret * self.reward_scaling)

        info = {
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "weights": self.weights.copy(),
            "step": self.t,
        }
        return self._get_obs(), reward, terminated, False, info

    def portfolio_value_series(self) -> np.ndarray:
        return np.array(self.history)

    def render(self):
        pass
