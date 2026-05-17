"""Bootstrap confidence intervals and multiple-testing corrections."""
from __future__ import annotations
import numpy as np
from typing import Callable


def bootstrap_ci(
    stat_fn: Callable,
    data: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for any scalar statistic.

    Parameters
    ----------
    stat_fn : callable  Takes ``data`` and returns a scalar.
    data    : array     Passed directly to stat_fn.
    n_boot  : int       Number of bootstrap draws.
    alpha   : float     Significance level (0.05 → 95% CI).

    Returns
    -------
    (lower, upper)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    boots = [stat_fn(data[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def bh_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR correction. Returns boolean array of rejections."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    threshold = (np.arange(1, n + 1) / n) * alpha
    reject = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        if p[order[i]] <= threshold[i]:
            reject[order[: i + 1]] = True
            break
    return reject


def robust_se(resid: np.ndarray, X: np.ndarray) -> np.ndarray:
    """HC3 heteroskedasticity-robust standard errors."""
    n, k = X.shape
    leverage = np.einsum("ij,ij->i", X @ np.linalg.pinv(X.T @ X), X)
    e = resid / (1 - leverage)
    meat = (X * e[:, None]).T @ (X * e[:, None])
    bread = np.linalg.pinv(X.T @ X)
    vcov = bread @ meat @ bread
    return np.sqrt(np.diag(vcov))
