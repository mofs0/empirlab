"""
Double/Debiased Machine Learning (DML)
Reference: Chernozhukov et al. (2018). The Econometrics Journal, 21(1), C1-C68.
           https://doi.org/10.1111/ectj.12097

Model (Partially Linear Regression):
    Y = theta*D + g(X) + eps    E[eps|D,X]=0
    D = m(X) + v                E[v|X]=0

Estimator (Eq. 2.3):
    1. Cross-fit residuals: eps_Y = Y - E[Y|X],  eps_D = D - E[D|X]
    2. theta_hat = (eps_D @ eps_D)^{-1} (eps_D @ eps_Y)
    3. Std err via influence function (Eq. 3.2)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

_LEARNERS = {
    "lasso":  LassoCV(cv=5, max_iter=5000),
    "ridge":  RidgeCV(cv=5),
    "forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                    max_features=0.5, n_jobs=-1),
}

class DoubleML:
    """Double/Debiased ML for the Partially Linear Regression model.

    Parameters
    ----------
    ml_l : str or sklearn estimator   Learner for E[Y|X]. Default "lasso".
    ml_m : str or sklearn estimator   Learner for E[D|X]. Default "lasso".
    n_folds : int                     Cross-fitting folds. Default 5.
    n_rep   : int                     Repeated cross-fitting. Default 1.
    random_state : int                Default 42.

    Examples
    --------
    >>> from empirlab.causal import DoubleML
    >>> from empirlab.causal.datasets import make_plr_data
    >>> X, y, d = make_plr_data(n=2000, theta=1.2)
    >>> dml = DoubleML().fit(X, y, d)
    >>> print(dml.summary())
    """
    def __init__(self, ml_l="lasso", ml_m="lasso",
                 n_folds=5, n_rep=1, random_state=42):
        self.ml_l = ml_l; self.ml_m = ml_m
        self.n_folds = n_folds; self.n_rep = n_rep
        self.random_state = random_state

    def _get(self, spec):
        return clone(_LEARNERS[spec]) if isinstance(spec, str) else clone(spec)

    def _rep(self, X, y, d, seed):
        n = len(y); ey = np.zeros(n); ed = np.zeros(n)
        for tr, te in KFold(self.n_folds, shuffle=True, random_state=seed).split(X):
            ey[te] = y[te] - self._get(self.ml_l).fit(X[tr], y[tr]).predict(X[te])
            ed[te] = d[te] - self._get(self.ml_m).fit(X[tr], d[tr]).predict(X[te])
        th = (ed @ ey) / (ed @ ed)
        psi = ed * (ey - th * ed)
        se  = np.sqrt(np.mean(psi**2) / n) / abs(-(ed @ ed) / n)
        return th, se

    def fit(self, X, y, d):
        X,y,d = np.asarray(X,float), np.asarray(y,float).ravel(), np.asarray(d,float).ravel()
        rs = [self._rep(X,y,d, self.random_state+i) for i in range(self.n_rep)]
        self.coef_    = float(np.median([r[0] for r in rs]))
        self.std_err_ = float(np.median([r[1] for r in rs]))
        self.t_stat_  = self.coef_ / self.std_err_
        self.p_value_ = 2 * stats.norm.sf(abs(self.t_stat_))
        z = stats.norm.ppf(0.975)
        self.ci_lower_ = self.coef_ - z * self.std_err_
        self.ci_upper_ = self.coef_ + z * self.std_err_
        self._fitted = True
        return self

    def summary(self) -> pd.DataFrame:
        check_is_fitted(self, "_fitted")
        sig = "***" if self.p_value_<0.01 else "**" if self.p_value_<0.05 else "*" if self.p_value_<0.1 else ""
        return pd.DataFrame({
            "coef":[round(self.coef_,4)], "std_err":[round(self.std_err_,4)],
            "t_stat":[round(self.t_stat_,4)], "p_value":[round(self.p_value_,4)],
            "ci_lower":[round(self.ci_lower_,4)], "ci_upper":[round(self.ci_upper_,4)],
            "sig":[sig]}, index=["treatment"])
