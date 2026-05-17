"""ML-enhanced portfolio construction.

Status: API stub — full implementation coming in v0.2.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class MLPortfolio:
    """Top-K long-short portfolio from ML return predictions.

    Parameters
    ----------
    k         : int, default 20   Stocks per leg (long / short).
    long_only : bool, default False
    rebalance : str, default 'monthly'  'daily' | 'weekly' | 'monthly'
    """

    def __init__(self, k: int = 20, long_only: bool = False,
                 rebalance: str = "monthly"):
        self.k         = k
        self.long_only = long_only
        self.rebalance = rebalance

    def construct(
        self,
        predictions: pd.Series,
        returns: pd.Series,
    ) -> pd.Series:
        """Build long-short portfolio returns.

        predictions : Series with MultiIndex (date, asset), predicted returns.
        returns     : Series with same index, realised returns.
        Returns     : portfolio return series indexed by date.
        """
        raise NotImplementedError("MLPortfolio.construct() coming in v0.2")
