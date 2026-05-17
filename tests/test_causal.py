"""Unit tests for empirlab.causal"""
import numpy as np
import pytest
from empirlab.causal import DoubleML, CausalForest
from empirlab.causal.datasets import make_plr_data, make_hte_data


def test_dml_recovers_theta():
    X, y, d = make_plr_data(n=1000, p=10, theta=1.5, seed=0)
    dml = DoubleML(ml_l="lasso", ml_m="lasso", n_folds=3)
    dml.fit(X, y, d)
    assert abs(dml.coef_ - 1.5) < 0.3, "DML theta far from truth"
    assert dml.p_value_ < 0.05


def test_dml_summary_columns():
    X, y, d = make_plr_data(n=500, p=5, theta=1.0, seed=1)
    df = DoubleML().fit(X, y, d).summary()
    assert list(df.columns) == ["coef", "std_err", "t_stat", "p_value",
                                 "ci_lower", "ci_upper", "sig"]
    assert df.index[0] == "treatment"


def test_causal_forest_predict_shape():
    X, y, d, tau = make_hte_data(n=500, p=5, seed=42)
    cf = CausalForest(n_estimators=50).fit(X, y, d)
    tau_hat = cf.predict(X)
    assert tau_hat.shape == (500,)


def test_causal_forest_ci_bounds():
    X, y, d, _ = make_hte_data(n=300, p=4, seed=7)
    cf = CausalForest(n_estimators=50).fit(X, y, d)
    lb, ub = cf.confidence_interval(X)
    tau = cf.predict(X)
    assert np.all(lb <= tau + 1e-8)
    assert np.all(ub >= tau - 1e-8)


def test_plr_data_shapes():
    X, y, d = make_plr_data(n=200, p=8)
    assert X.shape == (200, 8)
    assert y.shape == (200,)
    assert d.shape == (200,)
