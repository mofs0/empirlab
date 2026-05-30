"""Unit tests for empirlab.rl (requires gymnasium)."""
import numpy as np
import pytest

try:
    import gymnasium as gym
    _GYM = True
except ImportError:
    try:
        import gym
        _GYM = True
    except ImportError:
        _GYM = False

pytestmark = pytest.mark.skipif(not _GYM, reason="gymnasium not installed")


def _make_stock_env(n=300, n_feat=3):
    """Helper: build StockTradingEnv with synthetic data."""
    from empirlab.rl.envs.stock_env import StockTradingEnv
    rng = np.random.default_rng(42)
    prices   = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    features = rng.standard_normal((n, n_feat))
    return StockTradingEnv(prices=prices, features=features)


def _make_portfolio_env(n=200, n_assets=5, window=10):
    from empirlab.rl.envs.portfolio_env import PortfolioEnv
    rng = np.random.default_rng(42)
    prices = 100.0 * np.exp(np.cumsum(rng.standard_normal((n, n_assets)) * 0.01, axis=0))
    return PortfolioEnv(price_matrix=prices, window=window)


# ──────────────────────────────────────────────────────────────
class TestStockTradingEnv:
    def test_reset_returns_array(self):
        env = _make_stock_env()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_obs_dim(self):
        env = _make_stock_env(n_feat=4)
        obs = env.reset()
        # obs_dim = 2 (cash_ratio, shares_ratio) + n_feat
        assert obs.shape == (6,)

    def test_step_returns_4tuple(self):
        env = _make_stock_env()
        env.reset()
        result = env.step(1)   # buy
        assert len(result) == 4   # obs, reward, done, info

    def test_step_reward_finite(self):
        env = _make_stock_env()
        env.reset()
        obs, reward, done, info = env.step(1)
        assert np.isfinite(reward)

    def test_episode_terminates(self):
        env = _make_stock_env(n=50)
        env.reset()
        done = False; steps = 0
        while not done and steps < 200:
            obs, reward, done, info = env.step(env.action_space.sample()
                                               if hasattr(env, "action_space")
                                               else np.random.randint(3))
            steps += 1
        assert done, "Episode never terminated"

    def test_portfolio_value_positive(self):
        env = _make_stock_env()
        env.reset()
        for _ in range(20):
            env.step(1)   # always buy
        assert env.portfolio_value() > 0


# ──────────────────────────────────────────────────────────────
class TestPortfolioEnv:
    def test_reset_obs_shape(self):
        env = _make_portfolio_env(n_assets=4, window=10)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_action_space_includes_cash(self):
        env = _make_portfolio_env(n_assets=4)
        # 4 assets + 1 cash
        assert env.action_space.shape == (5,)

    def test_weights_sum_to_one(self):
        env = _make_portfolio_env()
        env.reset()
        for _ in range(5):
            obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            assert abs(info["weights"].sum() - 1.0) < 1e-6
            if terminated or truncated:
                break

    def test_portfolio_grows_on_all_up_market(self):
        # All assets appreciate 1% per period
        prices = 100.0 * (1.01 ** np.arange(60))[:, None] * np.ones((60, 3))
        from empirlab.rl.envs.portfolio_env import PortfolioEnv
        env = PortfolioEnv(prices, window=5, transaction_cost=0.0)
        env.reset()
        action = np.array([0.33, 0.33, 0.33, 0.01])
        done = False
        while not done:
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        vals = env.portfolio_value_series()
        assert vals[-1] > vals[0]

    def test_episode_terminates(self):
        env = _make_portfolio_env()
        env.reset()
        done = False; steps = 0
        while not done and steps < 500:
            obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated; steps += 1
        assert done
