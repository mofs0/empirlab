"""
Causal Forest — Heterogeneous Treatment Effect Estimation
=========================================================
Reference: Wager, S., & Athey, S. (2018).
           Estimation and inference of heterogeneous treatment effects
           using random forests.
           Journal of the American Statistical Association, 113(523), 1228-1242.
           https://doi.org/10.1080/01621459.2017.1319839

Implementation note
-------------------
True honest causal forests require two independent subsamples.
This implementation approximates via separate ExtraTreesRegressors on treated /
control units and derives CIs from the infinitesimal jackknife (IJ) over trees.
For production use, prefer econml.dml.CausalForestDML or the R grf package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from scipy import stats


class CausalForest:
    """Approximate Causal Forest for heterogeneous treatment effects.

    Parameters
    ----------
    n_estimators     : int, default 2000
    min_samples_leaf : int, default 5
    max_features     : str or float, default 'sqrt'
    n_jobs           : int, default -1
    random_state     : int, default 42

    Examples
    --------
    >>> from empirlab.causal import CausalForest
    >>> from empirlab.causal.datasets import make_hte_data
    >>> X, y, d, tau = make_hte_data(n=2000, p=10)
    >>> cf = CausalForest().fit(X, y, d)
    >>> tau_hat = cf.predict(X)
    >>> lb, ub  = cf.confidence_interval(X)
    """

    def __init__(self, n_estimators=2000, min_samples_leaf=5,
                 max_features="sqrt", n_jobs=-1, random_state=42):
        self.n_estimators     = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features     = max_features
        self.n_jobs           = n_jobs
        self.random_state     = random_state

    def _base(self):
        return ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def fit(self, X, y, d):
        """Fit response surfaces mu_1(X) and mu_0(X)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        d = np.asarray(d, dtype=float).ravel()
        idx1 = d == 1
        idx0 = d == 0
        self.mu1_ = self._base().fit(X[idx1], y[idx1])
        self.mu0_ = self._base().fit(X[idx0], y[idx0])
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict CATE tau(X) = E[Y(1) - Y(0) | X]."""
        X = np.asarray(X, dtype=float)
        return self.mu1_.predict(X) - self.mu0_.predict(X)

    def confidence_interval(
        self, X, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Infinitesimal jackknife confidence intervals for CATE.

        Returns
        -------
        lb, ub : np.ndarray of shape (n,)
        """
        X = np.asarray(X, dtype=float)
        tau_trees = np.array([
            t1.predict(X) - t0.predict(X)
            for t1, t0 in zip(self.mu1_.estimators_, self.mu0_.estimators_)
        ])  # (n_estimators, n)
        mu  = tau_trees.mean(axis=0)
        se  = np.sqrt(tau_trees.var(axis=0) / self.n_estimators)
        z   = stats.norm.ppf(1 - alpha / 2)
        return mu - z * se, mu + z * se

    def summary(self, X) -> pd.DataFrame:
        """Distribution summary of estimated CATEs."""
        tau = self.predict(X)
        lb, ub = self.confidence_interval(X)
        return pd.DataFrame({
            "cate_mean":    [round(tau.mean(), 4)],
            "cate_std":     [round(tau.std(), 4)],
            "cate_p25":     [round(float(np.percentile(tau, 25)), 4)],
            "cate_p75":     [round(float(np.percentile(tau, 75)), 4)],
            "ci_width_avg": [round(float((ub - lb).mean()), 4)],
        })
