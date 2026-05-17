"""
Single-Stock Trading Environment (Gymnasium-compatible)
=========================================================
Reference: Liu, X., et al. (2021). FinRL: A Deep Reinforcement Learning
           Library for Automated Stock Trading in Quantitative Finance.
           arXiv:2011.09607. https://arxiv.org/abs/2011.09607

Actions: 0 = hold, 1 = buy all-in, 2 = sell all
State  : [cash_norm, position_norm, *features_t]
"""
from __future__ import annotations
import numpy as np


class StockTradingEnv:
    """Discrete-action single-stock trading environment.

    Parameters
    ----------
    prices     : np.ndarray  shape (T,)  daily close prices
    features   : np.ndarray  shape (T, F)  technical indicators / macro
    init_cash  : float, default 1e4
    trade_cost : float, default 1e-3   fraction of trade value

    Examples
    --------
    >>> env = StockTradingEnv(prices, features)
    >>> obs = env.reset()
    >>> obs, reward, done, info = env.step(1)   # buy
    """

    def __init__(self, prices: np.ndarray, features: np.ndarray,
                 init_cash: float = 1e4, trade_cost: float = 1e-3):
        self.prices     = np.asarray(prices,   dtype=float)
        self.features   = np.asarray(features, dtype=float)
        self.init_cash  = init_cash
        self.trade_cost = trade_cost
        self.T          = len(prices)
        self.obs_dim    = 2 + features.shape[1]
        self.action_dim = 3
        self.reset()

    def reset(self) -> np.ndarray:
        self.t      = 0
        self.cash   = self.init_cash
        self.shares = 0.0
        return self._obs()

    def _obs(self) -> np.ndarray:
        price = self.prices[self.t]
        obs = np.concatenate([
            [self.cash   / (self.init_cash + 1e-8)],
            [self.shares * price / (self.init_cash + 1e-8)],
            self.features[self.t],
        ])
        return obs.astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        price    = self.prices[self.t]
        prev_val = self.cash + self.shares * price

        if action == 1:                       # buy all-in
            n          = int(self.cash / (price * (1 + self.trade_cost)))
            cost       = n * price * (1 + self.trade_cost)
            self.cash   -= cost
            self.shares += n
        elif action == 2:                     # sell all
            proceeds    = self.shares * price * (1 - self.trade_cost)
            self.cash   += proceeds
            self.shares  = 0.0

        self.t  += 1
        done     = self.t >= self.T - 1
        new_val  = self.cash + self.shares * self.prices[self.t]
        reward   = (new_val - prev_val) / (self.init_cash + 1e-8)
        return self._obs(), reward, done, {}

    def portfolio_value(self) -> float:
        return self.cash + self.shares * self.prices[self.t]
