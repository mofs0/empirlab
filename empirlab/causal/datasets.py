"""Benchmark data generators for causal inference methods."""
import numpy as np

def make_plr_data(n=2000, p=20, theta=1.2, sigma=1.0, seed=42):
    """Partially Linear Regression DGP.
    Y = theta*D + g(X) + eps,  D = m(X) + v
    Returns X (n,p), y (n,), d (n,)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    d = X[:, 0] + rng.standard_normal(n) * 0.3
    y = theta * d + 0.5 * X[:, 0]**2 + X[:, 1] + rng.standard_normal(n) * sigma
    return X, y, d

def make_hte_data(n=2000, p=10, seed=42):
    """Heterogeneous treatment effect DGP. True CATE = 1 + 2*X[:,0]"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    d = rng.binomial(1, 0.5, n)
    tau = 1 + 2 * X[:, 0]
    y = tau * d + X[:, 1] + rng.standard_normal(n)
    return X, y, d, tau
