"""
Post-LASSO Instrumental Variables
===================================
Reference: Belloni, A., Chen, D., Chernozhukov, V., & Hansen, C. (2012).
           Sparse models and methods for optimal instruments with an
           application to eminent domain.
           Econometrica, 80(6), 2369-2429.
           https://doi.org/10.3982/ECTA9626

Status: API stub — full implementation coming in v0.2.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class PostLassoIV:
    """Post-LASSO 2SLS for high-dimensional instrument selection.

    Parameters
    ----------
    penalty : str, default 'BC'
        Penalty choice: 'BC' (Belloni-Chernozhukov rigorous LASSO),
        'CV' (cross-validation).
    n_folds : int, default 5
    """

    def __init__(self, penalty: str = "BC", n_folds: int = 5):
        self.penalty = penalty
        self.n_folds = n_folds

    def fit(self, X, y, d, Z):
        """
        X : (n, p) controls
        y : (n,)   outcome
        d : (n,)   endogenous regressor
        Z : (n, q) instruments
        """
        raise NotImplementedError(
            "PostLassoIV coming in empirlab v0.2. "
            "For now: use linearmodels.iv.IV2SLS with manual LASSO first stage."
        )

    def summary(self) -> pd.DataFrame:
        raise NotImplementedError
