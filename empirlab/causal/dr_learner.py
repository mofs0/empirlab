"""
Doubly Robust Learner for CATE
================================
Reference: Kennedy, E.H. (2023). Towards optimal doubly robust estimation of
           heterogeneous causal effects. Electronic Journal of Statistics,
           17(2), 3008-3049. https://doi.org/10.1214/23-EJS2157

Algorithm (cross-fitted DR-Learner):
  1. Cross-fitting: in each fold estimate outcome models E[Y|X,D=d] and
     propensity score P(D=1|X).
  2. Construct doubly-robust pseudo-outcomes:
       φ_i = μ_1(X_i) − μ_0(X_i)
            + D_i(Y_i − μ_1(X_i))/e(X_i)
            − (1-D_i)(Y_i − μ_0(X_i))/(1-e(X_i))
  3. Regress φ on X with ml_tau → τ̂(x).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.base import clone, is_classifier


class DRLearner:
    """Doubly Robust Learner for heterogeneous treatment effects (Kennedy 2023).

    Parameters
    ----------
    ml_g : sklearn estimator or None
        Outcome model for E[Y | X, D=d].
        Default: GradientBoostingRegressor(n_estimators=200).
    ml_p : sklearn estimator or None
        Propensity model P(D=1 | X).
        Default: GradientBoostingClassifier(n_estimators=200).
    ml_tau : sklearn estimator or None
        Final CATE regressor τ̂(x).
        Default: Ridge(alpha=1.0).
    n_folds : int, default 5
        Cross-fitting folds.
    clip_propensity : float, default 0.01
        Clip propensity scores to [clip, 1-clip] to avoid extreme weights.

    Examples
    --------
    >>> import numpy as np
    >>> from empirlab.causal.datasets import make_hte_data
    >>> X, y, d, tau_true = make_hte_data(n=1000, p=5, seed=0)
    >>> dr = DRLearner(n_folds=3)
    >>> dr.fit(X, y, d)
    >>> tau_hat = dr.predict(X)
    >>> print(f"Corr with truth: {np.corrcoef(tau_hat, tau_true)[0,1]:.3f}")
    """

    def __init__(self, ml_g=None, ml_p=None, ml_tau=None,
                 n_folds: int = 5, clip_propensity: float = 0.01):
        self.ml_g   = ml_g
        self.ml_p   = ml_p
        self.ml_tau = ml_tau
        self.n_folds = n_folds
        self.clip_propensity = clip_propensity

    def _default_g(self):
        return GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, random_state=42)

    def _default_p(self):
        return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                          learning_rate=0.05, random_state=42)

    def _default_tau(self):
        return Ridge(alpha=1.0)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, d: np.ndarray) -> "DRLearner":
        """Fit the DR-Learner.

        Parameters
        ----------
        X : (n, p)  covariates
        y : (n,)    outcome
        d : (n,)    binary treatment (0/1)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        d = np.asarray(d, dtype=float).ravel()
        n = len(y)

        ml_g1  = clone(self.ml_g)   if self.ml_g   is not None else self._default_g()
        ml_g0  = clone(self.ml_g)   if self.ml_g   is not None else self._default_g()
        ml_p   = clone(self.ml_p)   if self.ml_p   is not None else self._default_p()

        mu1  = np.zeros(n)
        mu0  = np.zeros(n)
        prop = np.zeros(n)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, d_tr  = y[train_idx], d[train_idx]

            # Outcome models
            g1 = clone(ml_g1)
            g0 = clone(ml_g0)
            g1.fit(X_tr[d_tr == 1], y_tr[d_tr == 1])
            g0.fit(X_tr[d_tr == 0], y_tr[d_tr == 0])
            mu1[val_idx] = g1.predict(X_val)
            mu0[val_idx] = g0.predict(X_val)

            # Propensity model
            p_mod = clone(ml_p)
            p_mod.fit(X_tr, d_tr)
            if hasattr(p_mod, "predict_proba"):
                prop[val_idx] = p_mod.predict_proba(X_val)[:, 1]
            else:
                prop[val_idx] = p_mod.predict(X_val)

        # Clip propensity
        prop = np.clip(prop, self.clip_propensity, 1 - self.clip_propensity)

        # DR pseudo-outcomes
        phi = (mu1 - mu0
               + d * (y - mu1) / prop
               - (1 - d) * (y - mu0) / (1 - prop))

        # Final CATE regression
        tau_model = (clone(self.ml_tau) if self.ml_tau is not None
                     else self._default_tau())
        tau_model.fit(X, phi)
        self.tau_model_  = tau_model
        self.phi_        = phi        # DR pseudo-outcomes (for diagnostics)
        self.mu1_        = mu1
        self.mu0_        = mu0
        self.prop_       = prop
        self._n          = n
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE τ̂(x) for new observations."""
        if not hasattr(self, "tau_model_"):
            raise RuntimeError("Call fit() first.")
        return self.tau_model_.predict(np.asarray(X, dtype=float))

    # ------------------------------------------------------------------
    def confidence_interval(self, X: np.ndarray,
                             alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Approximate pointwise CI via bootstrap of pseudo-outcome regression.

        Note: This is a simplified interval around the CATE prediction using
        the variance of phi residuals — not a full honest CI.
        """
        tau_hat = self.predict(X)
        resid   = self.phi_ - self.predict(self._X_train if hasattr(self, "_X_train") else X[:self._n])
        sigma   = np.std(resid)
        z = 1.96 if alpha == 0.05 else float(np.abs(np.percentile(
            np.random.standard_normal(100000), 100 * alpha / 2)))
        return tau_hat - z * sigma, tau_hat + z * sigma

    # ------------------------------------------------------------------
    def summary(self, X: np.ndarray | None = None) -> pd.DataFrame:
        """Summary statistics of the estimated CATE distribution."""
        tau = self.predict(X) if X is not None else self.phi_
        return pd.DataFrame({
            "mean_cate":   [tau.mean()],
            "std_cate":    [tau.std()],
            "p10_cate":    [np.percentile(tau, 10)],
            "p50_cate":    [np.percentile(tau, 50)],
            "p90_cate":    [np.percentile(tau, 90)],
        })
