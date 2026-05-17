"""Unit tests for empirlab.finance"""
import numpy as np
import pytest
from empirlab.finance import MLFactorModel


def test_fit_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))
    y = rng.standard_normal(200)
    pred = MLFactorModel(method="ridge").fit(X, y).predict(X[:20])
    assert pred.shape == (20,)


def test_r2_oos_positive_on_easy_dgp():
    rng = np.random.default_rng(1)
    X_tr = rng.standard_normal((300, 8))
    y_tr = X_tr[:, 0] + rng.standard_normal(300) * 0.1
    X_te = rng.standard_normal((100, 8))
    y_te = X_te[:, 0] + rng.standard_normal(100) * 0.1
    r2 = MLFactorModel(method="enet").fit(X_tr, y_tr).r2_oos(X_te, y_te)
    assert np.isfinite(r2) and r2 > 0.5


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        MLFactorModel(method="xgboost")


def test_feature_importance_available():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 5))
    y = rng.standard_normal(100)
    model = MLFactorModel(method="forest").fit(X, y)
    fi = model.feature_importance_
    assert fi is not None and fi.shape == (5,)
