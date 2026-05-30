"""
ML Factor Model
===============
Reference: Gu, S., Kelly, B., & Xiu, D. (2020).
           Empirical asset pricing via machine learning.
           Review of Financial Studies, 33(5), 2223-2273.
           https://doi.org/10.1093/rfs/hhaa009

Model:  r_{i,t+1} = g(z_{i,t}) + epsilon_{i,t+1}
where z_{i,t} is a vector of stock characteristics.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

_MODELS = {
    "enet":   ElasticNetCV(cv=5, max_iter=10000, l1_ratio=[0.1, 0.5, 0.9, 0.99]),
    "lasso":  LassoCV(cv=5, max_iter=10000),
    "ridge":  RidgeCV(cv=5),
    "forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                    max_features=0.3, n_jobs=-1),
    "gbm":    GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,
                                        max_depth=3, subsample=0.5),
}


class MLFactorModel:
    """ML-based cross-sectional return predictor (Gu, Kelly & Xiu 2020 style).

    Parameters
    ----------
    method      : str, default 'enet'
        One of 'enet', 'lasso', 'ridge', 'forest', 'gbm'.
    standardize : bool, default True

    Examples
    --------
    >>> model = MLFactorModel(method='ridge')
    >>> model.fit(chars_train, returns_train)
    >>> r_hat = model.predict(chars_test)
    >>> print(f'OOS R² = {model.r2_oos(chars_test, returns_test):.4f}')
    """

    def __init__(self, method: str = "enet", standardize: bool = True):
        if method not in _MODELS:
            raise ValueError(f"method must be one of {list(_MODELS)}, got '{method}'")
        self.method      = method
        self.standardize = standardize

    def fit(self, chars: np.ndarray, returns: np.ndarray):
        """Train on (N*T, P) characteristic matrix and (N*T,) return vector."""
        X = np.asarray(chars,   dtype=float)
        y = np.asarray(returns, dtype=float).ravel()
        if self.standardize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        self.model_ = clone(_MODELS[self.method])
        self.model_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, chars: np.ndarray) -> np.ndarray:
        """Predict expected returns for new characteristic matrix."""
        X = np.asarray(chars, dtype=float)
        if self.standardize:
            X = self.scaler_.transform(X)
        return self.model_.predict(X)

    @property
    def feature_importance_(self) -> np.ndarray | None:
        """Feature importances (available for tree methods; |coef| for linear)."""
        if hasattr(self.model_, "feature_importances_"):
            return self.model_.feature_importances_
        if hasattr(self.model_, "coef_"):
            return np.abs(self.model_.coef_)
        return None

    def r2_oos(self, chars_test: np.ndarray, returns_test: np.ndarray) -> float:
        """Out-of-sample R² (Campbell & Thompson 2008): 1 - SS_res / SS_tot,
        where SS_tot uses zero expected return as benchmark.
        """
        y    = np.asarray(returns_test, dtype=float).ravel()
        yhat = self.predict(chars_test)
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = (y ** 2).sum()
        return float(1 - ss_res / ss_tot)
