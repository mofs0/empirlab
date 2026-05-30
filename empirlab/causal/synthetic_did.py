"""
Synthetic Difference-in-Differences
=====================================
Reference: Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W., &
           Wager, S. (2021). Synthetic difference-in-differences.
           American Economic Review, 111(12), 4088-4118.
           https://doi.org/10.1257/aer.20190159

Implementation follows Algorithm 1:
  1. Estimate unit weights ω by balancing pre-treatment trends.
  2. Estimate time weights λ by balancing post-treatment means.
  3. Weighted 2x2 DiD regression → ATT + bootstrap SE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _unit_weights(Y_pre: np.ndarray, treated_mask: np.ndarray, zeta: float) -> np.ndarray:
    """Unit weights omega* (Eq. 2.3). Minimises ||Y_donor @ w - Y_treated_mean||^2 + zeta*||w||^2."""
    Y_donor = Y_pre[~treated_mask]
    y_target = Y_pre[treated_mask].mean(0)
    N0 = Y_donor.shape[0]
    w0 = np.ones(N0) / N0

    def obj(w):
        r = Y_donor.T @ w - y_target
        return r @ r + zeta * (w @ w)

    def jac(w):
        r = Y_donor.T @ w - y_target
        return 2 * Y_donor @ r + 2 * zeta * w

    res = minimize(obj, w0, jac=jac, method="SLSQP",
                   bounds=[(0, None)] * N0,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-9, "maxiter": 5000})
    return res.x


def _time_weights(Y_post_ctrl: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Time weights lambda* (Eq. 2.5). Balances post-period weighted control mean."""
    T_post = Y_post_ctrl.shape[1]
    l0 = np.ones(T_post) / T_post
    target = omega @ Y_post_ctrl.mean(1)  # scalar

    def obj(lv):
        diff = omega @ (Y_post_ctrl @ lv) - target
        return diff ** 2

    res = minimize(obj, l0, method="SLSQP",
                   bounds=[(0, None)] * T_post,
                   constraints=[{"type": "eq", "fun": lambda lv: lv.sum() - 1}],
                   options={"ftol": 1e-9, "maxiter": 5000})
    return res.x


class SyntheticDiD:
    """Synthetic Difference-in-Differences estimator (Arkhangelsky et al. 2021).

    Parameters
    ----------
    zeta : float or None
        L2 regularisation for unit weights.  None (default) auto-selects
        ζ = N_control * T_pre * σ̂² as in Appendix A of the paper.
    n_boot : int, default 200
        Bootstrap replications for SE estimation.
    random_state : int or None

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # 20 units, 12 time periods, treatment starts at period 8
    >>> Y = rng.standard_normal((20, 12))
    >>> Y[0, 8:] += 2.0   # true ATT = 2
    >>> sdid = SyntheticDiD(n_boot=100)
    >>> sdid.fit(Y, treated_units=[0], T_pre=8)
    >>> print(sdid.summary())
    """

    def __init__(self, zeta: float | None = None,
                 n_boot: int = 200,
                 random_state: int | None = 42):
        self.zeta = zeta
        self.n_boot = n_boot
        self.random_state = random_state

    def fit(self, Y: np.ndarray, treated_units: list[int], T_pre: int) -> "SyntheticDiD":
        """Fit the estimator.

        Parameters
        ----------
        Y            : ndarray (N, T)  — outcome panel
        treated_units: list[int]       — row indices of treated units
        T_pre        : int             — number of pre-treatment periods

        Returns
        -------
        self
        """
        Y = np.asarray(Y, dtype=float)
        N, T = Y.shape

        mask = np.zeros(N, dtype=bool)
        mask[list(treated_units)] = True

        Y_pre_all  = Y[:, :T_pre]
        Y_post_all = Y[:, T_pre:]
        N0 = (~mask).sum()

        # Auto-select zeta
        if self.zeta is None:
            sigma2 = float(np.var(np.diff(Y_pre_all[~mask], axis=1), ddof=1))
            zeta = N0 * T_pre * sigma2
        else:
            zeta = float(self.zeta)
        self.zeta_ = zeta

        omega = _unit_weights(Y_pre_all, mask, zeta)
        self.omega_ = omega  # (N0,)

        lam = _time_weights(Y_post_all[~mask], omega)
        self.lambda_ = lam  # (T_post,)

        # Weighted 2x2 DiD
        Y_ctrl = Y[~mask]
        Y_trt  = Y[mask]
        pre_ctrl  = Y_ctrl[:, :T_pre].mean(1) @ omega
        post_ctrl = (Y_ctrl[:, T_pre:] @ lam) @ omega
        pre_trt   = Y_trt[:, :T_pre].mean()
        post_trt  = (Y_trt[:, T_pre:] @ lam).mean()

        self.att_ = (post_trt - pre_trt) - (post_ctrl - pre_ctrl)
        self.se_  = None
        self._Y, self._mask, self._T_pre = Y, mask, T_pre
        return self

    def bootstrap_se(self, n_boot: int | None = None) -> float:
        """Placebo bootstrap SE (Algorithm 2 of the paper)."""
        if not hasattr(self, "att_"):
            raise RuntimeError("Call fit() first.")
        rng   = np.random.default_rng(self.random_state)
        nb    = n_boot or self.n_boot
        Y, mask, T_pre = self._Y, self._mask, self._T_pre
        ctrl_idx = np.where(~mask)[0]
        atts = []
        for _ in range(nb):
            bc = rng.choice(ctrl_idx, size=len(ctrl_idx), replace=True)
            placebo_trt = [int(bc[0])]
            Y_b = np.vstack([Y[bc], Y[placebo_trt]])
            try:
                est = SyntheticDiD(zeta=self.zeta_, n_boot=0, random_state=None)
                est.fit(Y_b, [len(bc)], T_pre)
                atts.append(est.att_)
            except Exception:
                continue
        self.se_ = float(np.std(atts, ddof=1))
        return self.se_

    def estimate(self) -> dict:
        """Point estimate + 95% CI."""
        if self.se_ is None:
            self.bootstrap_se()
        return {"att": self.att_, "se": self.se_,
                "ci_lower": self.att_ - 1.96 * self.se_,
                "ci_upper": self.att_ + 1.96 * self.se_}

    def summary(self) -> pd.DataFrame:
        """Return tidy summary DataFrame."""
        from scipy import stats
        res = self.estimate()
        z = res["att"] / res["se"] if res["se"] > 0 else np.nan
        p = 2 * stats.norm.sf(abs(z))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "."
        return pd.DataFrame({"att": [res["att"]], "std_err": [res["se"]],
                             "z_stat": [z], "p_value": [p],
                             "ci_lower": [res["ci_lower"]], "ci_upper": [res["ci_upper"]],
                             "sig": [sig]}, index=["treatment"])
