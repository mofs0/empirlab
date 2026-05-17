"""Unit tests for empirlab.utils"""
import numpy as np
import pytest
from empirlab.utils.metrics import sharpe, max_drawdown, ic, rmse, mse
from empirlab.utils.inference import bootstrap_ci, bh_correction


def test_sharpe_positive_drift():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.02, 252)
    assert isinstance(sharpe(r), float)


def test_sharpe_zero_std():
    assert sharpe(np.ones(100) * 0.001) == 0.0


def test_max_drawdown():
    prices = np.array([100, 110, 90, 95, 80, 100], dtype=float)
    mdd = max_drawdown(prices)
    assert abs(mdd - (-30 / 110)) < 1e-6


def test_ic_perfect():
    y = np.arange(10, dtype=float)
    assert abs(ic(y, y) - 1.0) < 1e-10


def test_rmse_half():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert abs(rmse(y_pred, y_true) - 0.5) < 1e-10


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(42)
    data = rng.normal(5.0, 1.0, 500)
    lo, hi = bootstrap_ci(np.mean, data, n_boot=1000, alpha=0.05, seed=0)
    assert lo < 5.0 < hi


def test_bh_correction_all_null():
    p = np.ones(50) * 0.5
    assert not bh_correction(p, alpha=0.05).any()


def test_bh_correction_obvious_signal():
    p = np.concatenate([np.ones(5) * 1e-10, np.ones(45) * 0.9])
    reject = bh_correction(p, alpha=0.05)
    assert reject[:5].all()
    assert not reject[5:].any()
