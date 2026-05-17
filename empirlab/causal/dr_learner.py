"""
Doubly Robust Learner for CATE
================================
Reference: Kennedy, E.H. (2023). Towards optimal doubly robust estimation of
           heterogeneous causal effects. Electronic Journal of Statistics,
           17(2), 3008–3049. https://doi.org/10.1214/23-EJS2157

Status: API stub — full implementation coming in v0.2.
"""
from __future__ import annotations
import numpy as np


class DRLearner:
    """Doubly Robust Learner for heterogeneous treatment effects.

    Parameters
    ----------
    ml_g   : sklearn estimator   Outcome model E[Y | X, D].
    ml_p   : sklearn estimator   Propensity model P(D=1 | X).
    ml_tau : sklearn estimator   Final CATE regressor.
    n_folds : int, default 5
    """

    def __init__(self, ml_g=None, ml_p=None, ml_tau=None, n_folds=5):
        self.ml_g    = ml_g
        self.ml_p    = ml_p
        self.ml_tau  = ml_tau
        self.n_folds = n_folds

    def fit(self, X, y, d):
        raise NotImplementedError(
            "DRLearner is coming in empirlab v0.2. "
            "For now, use econml.dr.DRLearner."
        )

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def confidence_interval(self, X, alpha: float = 0.05):
        raise NotImplementedError
