"""Multi-asset portfolio rebalancing environment.

Status: API stub — full implementation coming in v0.2.
"""
from __future__ import annotations
import numpy as np


class PortfolioEnv:
    """Continuous-action multi-asset portfolio environment.

    Action  : portfolio weight vector (N assets), must sum to 1.
    State   : [current_weights, *return_features]

    Status: coming in v0.2. See FinRL for production use.
    """

    def __init__(self, returns: np.ndarray, features: np.ndarray = None,
                 init_value: float = 1e4, trade_cost: float = 5e-4):
        self.returns    = np.asarray(returns, dtype=float)  # (T, N)
        self.features   = features
        self.init_value = init_value
        self.trade_cost = trade_cost
        self.N          = returns.shape[1]
        self.T          = returns.shape[0]

    def reset(self) -> np.ndarray:
        raise NotImplementedError("PortfolioEnv coming in v0.2")

    def step(self, weights: np.ndarray):
        raise NotImplementedError
