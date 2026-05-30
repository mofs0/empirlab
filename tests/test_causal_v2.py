"""Unit tests for empirlab.causal v0.2 modules."""
import numpy as np
import pytest
from empirlab.causal import SyntheticDiD, PostLassoIV, DRLearner
from empirlab.causal.datasets import make_hte_data


# ================================================================ SyntheticDiD
class TestSyntheticDiD:
    def _make_panel(self, N=20, T=12, T_pre=8, att=2.0, seed=0):
        rng = np.random.default_rng(seed)
        Y = rng.standard_normal((N, T)) * 0.5
        Y += np.arange(N)[:, None] * 0.1     # unit FEs
        Y += np.linspace(0, 1, T)[None, :]   # time trend
        Y[0, T_pre:] += att
        return Y, T_pre, att

    def test_fit_returns_self(self):
        Y, T_pre, _ = self._make_panel()
        sdid = SyntheticDiD(n_boot=0)
        ret = sdid.fit(Y, [0], T_pre)
        assert ret is sdid

    def test_att_has_correct_sign(self):
        Y, T_pre, att = self._make_panel(att=3.0, N=25, T=16, T_pre=10)
        sdid = SyntheticDiD(n_boot=0)
        sdid.fit(Y, [0], T_pre)
        # ATT should be positive (within 2 std of truth for this small sample)
        assert sdid.att_ > 0.0

    def test_omega_sums_to_one(self):
        Y, T_pre, _ = self._make_panel()
        sdid = SyntheticDiD(n_boot=0)
        sdid.fit(Y, [0], T_pre)
        assert abs(sdid.omega_.sum() - 1.0) < 1e-6

    def test_lambda_sums_to_one(self):
        Y, T_pre, _ = self._make_panel()
        sdid = SyntheticDiD(n_boot=0)
        sdid.fit(Y, [0], T_pre)
        assert abs(sdid.lambda_.sum() - 1.0) < 1e-6

    def test_bootstrap_se_positive(self):
        Y, T_pre, _ = self._make_panel(N=15, T=10, T_pre=6)
        sdid = SyntheticDiD(n_boot=30, random_state=0)
        sdid.fit(Y, [0], T_pre)
        se = sdid.bootstrap_se(n_boot=30)
        assert se > 0

    def test_summary_columns(self):
        Y, T_pre, _ = self._make_panel(N=15, T=10, T_pre=6)
        sdid = SyntheticDiD(n_boot=20, random_state=0)
        sdid.fit(Y, [0], T_pre)
        df = sdid.summary()
        assert "att" in df.columns and "std_err" in df.columns


# ================================================================ PostLassoIV
class TestPostLassoIV:
    def _make_iv_data(self, n=300, p=5, q=15, seed=0):
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, q))
        X = rng.standard_normal((n, p))
        pi = np.zeros(q); pi[:3] = [1.5, 1.0, 0.8]
        nu = rng.standard_normal(n)
        D = Z @ pi + X[:, 0] + nu * 0.3
        beta_true = 2.0
        eps = 0.4 * nu + rng.standard_normal(n)
        Y = beta_true * D + X[:, 0] + eps
        return X, Y, D, Z, beta_true

    def test_fit_returns_self(self):
        X, Y, D, Z, _ = self._make_iv_data()
        iv = PostLassoIV()
        ret = iv.fit(X, Y, D, Z)
        assert ret is iv

    def test_selects_some_instruments(self):
        X, Y, D, Z, _ = self._make_iv_data()
        iv = PostLassoIV()
        iv.fit(X, Y, D, Z)
        assert iv.n_selected_ > 0

    def test_f_stat_strong_instruments(self):
        X, Y, D, Z, _ = self._make_iv_data()
        iv = PostLassoIV()
        iv.fit(X, Y, D, Z)
        assert iv.f_stat_ > 10, f"Weak instrument F={iv.f_stat_:.1f}"

    def test_coefficient_close_to_truth(self):
        X, Y, D, Z, beta_true = self._make_iv_data(n=500)
        iv = PostLassoIV()
        iv.fit(X, Y, D, Z)
        assert abs(iv.coef_ - beta_true) < 0.5, f"coef={iv.coef_:.3f}, truth={beta_true}"

    def test_summary_shape(self):
        X, Y, D, Z, _ = self._make_iv_data()
        iv = PostLassoIV()
        iv.fit(X, Y, D, Z)
        df = iv.summary()
        assert "coef" in df.columns and "f_stat_fs" in df.columns


# ================================================================ DRLearner
class TestDRLearner:
    def test_predict_shape(self):
        X, y, d, _ = make_hte_data(n=300, p=5, seed=0)
        dr = DRLearner(n_folds=3)
        dr.fit(X, y, d)
        tau = dr.predict(X)
        assert tau.shape == (300,)

    def test_phi_shape(self):
        X, y, d, _ = make_hte_data(n=200, p=4, seed=1)
        dr = DRLearner(n_folds=3)
        dr.fit(X, y, d)
        assert dr.phi_.shape == (200,)

    def test_corr_with_truth(self):
        X, y, d, tau_true = make_hte_data(n=800, p=5, seed=42)
        dr = DRLearner(n_folds=3)
        dr.fit(X, y, d)
        tau_hat = dr.predict(X)
        corr = np.corrcoef(tau_hat, tau_true)[0, 1]
        assert corr > 0.3, f"Low correlation with ground truth: {corr:.3f}"

    def test_propensity_clipped(self):
        X, y, d, _ = make_hte_data(n=200, p=4, seed=5)
        dr = DRLearner(n_folds=3, clip_propensity=0.1)
        dr.fit(X, y, d)
        assert dr.prop_.min() >= 0.1
        assert dr.prop_.max() <= 0.9
