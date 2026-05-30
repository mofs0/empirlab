"""
Post-LASSO Instrumental Variables
===================================
Reference: Belloni, A., Chen, D., Chernozhukov, V., & Hansen, C. (2012).
           Sparse models and methods for optimal instruments with an
           application to eminent domain.
           Econometrica, 80(6), 2369-2429.
           https://doi.org/10.3982/ECTA9626

Algorithm (Post-LASSO 2SLS):
  1. LASSO first stage: D ~ Z + X  →  selected instruments Z_hat
  2. OLS first stage on Z_hat + X  →  D_hat  (Post-LASSO prediction)
  3. 2SLS second stage: Y ~ D_hat + X  →  ATT with HC3 SEs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler


class PostLassoIV:
    """Post-LASSO 2SLS for high-dimensional instrument selection.

    Parameters
    ----------
    penalty : {'CV', 'BC'}
        'CV'  — select LASSO penalty via 5-fold CV (default).
        'BC'  — Belloni-Chernozhukov rigorous penalty  λ = 2.2 * σ * sqrt(n * log(2*q/δ))
                with δ=0.05 (requires homoscedastic first stage).
    n_folds : int, default 5
        Folds for CV penalty selection.
    alpha_bc : float, default 0.05
        Significance level for BC penalty.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n, p, q = 500, 10, 30
    >>> Z = rng.standard_normal((n, q))   # instruments
    >>> X = rng.standard_normal((n, p))   # controls
    >>> pi = np.zeros(q); pi[:3] = 1.0
    >>> D = Z @ pi + rng.standard_normal(n) * 0.5   # endogenous regressor
    >>> Y = 2.0 * D + X[:, 0] + rng.standard_normal(n)
    >>> iv = PostLassoIV()
    >>> iv.fit(X, Y, D, Z)
    >>> print(iv.summary())
    """

    def __init__(self, penalty: str = "CV", n_folds: int = 5,
                 alpha_bc: float = 0.05):
        self.penalty  = penalty
        self.n_folds  = n_folds
        self.alpha_bc = alpha_bc

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray,
            d: np.ndarray, Z: np.ndarray) -> "PostLassoIV":
        """Fit Post-LASSO 2SLS.

        Parameters
        ----------
        X : (n, p)  exogenous controls
        y : (n,)    outcome
        d : (n,)    endogenous regressor
        Z : (n, q)  candidate instruments (excluded)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        d = np.asarray(d, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float)
        n, p = X.shape
        q = Z.shape[1]

        # Combine instruments + controls for first stage
        W = np.hstack([Z, X])   # (n, q+p)
        scaler = StandardScaler()
        W_sc = scaler.fit_transform(W)

        # ---- First stage: LASSO  D ~ W --------------------------------
        if self.penalty == "CV":
            lasso = LassoCV(cv=self.n_folds, n_jobs=-1, max_iter=10000)
            lasso.fit(W_sc, d)
            selected = np.where(lasso.coef_[:q] != 0)[0]  # selected instrument indices
        else:  # BC penalty
            sigma_d = np.std(d - W_sc @ np.linalg.lstsq(W_sc, d, rcond=None)[0], ddof=W_sc.shape[1])
            lam = 2.2 * sigma_d * np.sqrt(n * np.log(2 * q / self.alpha_bc)) / n
            lasso = Lasso(alpha=lam, max_iter=10000)
            lasso.fit(W_sc, d)
            selected = np.where(lasso.coef_[:q] != 0)[0]

        self.selected_instruments_ = selected
        self.n_selected_ = len(selected)

        if len(selected) == 0:
            raise ValueError(
                "LASSO selected zero instruments. Try penalty='CV' or provide "
                "stronger instruments."
            )

        # ---- Post-LASSO first stage OLS  D ~ Z_sel + X ---------------
        Z_sel  = Z[:, selected]
        W_post = np.hstack([Z_sel, X, np.ones((n, 1))])
        coef_fs, _, _, _ = np.linalg.lstsq(W_post, d, rcond=None)
        D_hat = W_post @ coef_fs
        self.first_stage_coef_ = coef_fs
        self.D_hat_ = D_hat

        # First-stage F-statistic
        resid_fs = d - D_hat
        resid_null = d - d.mean()
        self.f_stat_ = float(
            ((resid_null @ resid_null - resid_fs @ resid_fs) / len(selected))
            / (resid_fs @ resid_fs / (n - len(selected) - p - 1))
        )

        # ---- Second stage OLS  Y ~ D_hat + X --------------------------
        W2 = np.hstack([D_hat[:, None], X, np.ones((n, 1))])
        coef_2s, _, _, _ = np.linalg.lstsq(W2, y, rcond=None)
        self.coef_    = coef_2s[0]          # structural coefficient on D
        self.coef_X_  = coef_2s[1:-1]
        self.intercept_ = coef_2s[-1]

        # HC3 robust SE for the structural coefficient
        resid2 = y - W2 @ coef_2s
        X2TX2  = W2.T @ W2
        meat   = 0.0
        leverage = np.einsum("ij,jk,ik->i", W2, np.linalg.inv(X2TX2), W2)
        for i in range(n):
            e2 = resid2[i] ** 2 / (1 - leverage[i]) ** 2
            v  = W2[i][:, None]
            meat += e2 * (v @ v.T)
        V = np.linalg.inv(X2TX2) @ meat @ np.linalg.inv(X2TX2)
        self.se_ = float(np.sqrt(V[0, 0]))

        self._n = n
        return self

    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Return tidy summary DataFrame."""
        import scipy.stats as stats
        t = self.coef_ / self.se_
        p = 2 * stats.t.sf(abs(t), df=self._n - 2)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "."
        df = pd.DataFrame({
            "coef":     [self.coef_],
            "std_err":  [self.se_],
            "t_stat":   [t],
            "p_value":  [p],
            "ci_lower": [self.coef_ - 1.96 * self.se_],
            "ci_upper": [self.coef_ + 1.96 * self.se_],
            "sig":      [sig],
            "f_stat_fs":[self.f_stat_],
            "n_selected":[self.n_selected_],
        }, index=["treatment"])
        return df
