"""Financial and econometric evaluation metrics."""
from __future__ import annotations
import numpy as np


def sharpe(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    """Annualised Sharpe ratio."""
    r = np.asarray(returns, dtype=float)
    excess = r - rf / periods
    std = excess.std()
    return 0.0 if np.isclose(std, 0.0) else float(excess.mean() / std * np.sqrt(periods))


def max_drawdown(prices: np.ndarray) -> float:
    """Maximum drawdown (negative, e.g. -0.25 means -25%)."""
    p = np.asarray(prices, dtype=float)
    peak = np.maximum.accumulate(p)
    dd = (p - peak) / np.where(peak == 0, 1, peak)
    return float(dd.min())


def ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Information Coefficient — Pearson correlation of predictions and realised returns."""
    from scipy.stats import pearsonr
    r, _ = pearsonr(np.asarray(y_pred).ravel(), np.asarray(y_true).ravel())
    return float(r)


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def calmar(returns: np.ndarray, prices: np.ndarray, periods: int = 252) -> float:
    """Calmar ratio = annualised return / |max drawdown|."""
    ann_ret = float(np.mean(returns) * periods)
    mdd = abs(max_drawdown(prices))
    return ann_ret / mdd if mdd != 0 else np.nan


def turnover(weights: np.ndarray) -> float:
    """Average daily turnover given a (T, N) weight matrix."""
    w = np.asarray(weights, dtype=float)
    return float(np.abs(np.diff(w, axis=0)).sum(axis=1).mean())
