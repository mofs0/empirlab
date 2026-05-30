"""Unit tests for empirlab.dl (requires torch)."""
import numpy as np
import pytest

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

pytestmark = pytest.mark.skipif(not _TORCH, reason="torch not installed")


@pytest.fixture
def seq_data():
    """Return (X, y) for sequence regression: X (n, window, features), y (n,)."""
    rng = np.random.default_rng(0)
    n, window, features = 200, 10, 5
    X = rng.standard_normal((n, window, features)).astype(np.float32)
    # y is sum of last step values (easy to learn)
    y = X[:, -1, :].sum(axis=1).astype(np.float32)
    return X, y


class TestLSTMForecaster:
    def test_predict_shape(self, seq_data):
        from empirlab.dl.lstm import LSTMForecaster
        X, y = seq_data
        model = LSTMForecaster(hidden_size=16, n_layers=1, epochs=2, batch_size=32)
        model.fit(X, y)
        preds = model.predict(X[:20])
        assert preds.shape == (20,)

    def test_predict_dtype_float(self, seq_data):
        from empirlab.dl.lstm import LSTMForecaster
        X, y = seq_data
        model = LSTMForecaster(hidden_size=8, epochs=1)
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert preds.dtype in (np.float32, np.float64)

    def test_fit_returns_self(self, seq_data):
        from empirlab.dl.lstm import LSTMForecaster
        X, y = seq_data
        model = LSTMForecaster(hidden_size=8, epochs=1)
        ret = model.fit(X, y)
        assert ret is model

    def test_gru_variant(self, seq_data):
        from empirlab.dl.lstm import LSTMForecaster
        X, y = seq_data
        model = LSTMForecaster(rnn_type="GRU", hidden_size=16, epochs=2)
        model.fit(X, y)
        assert model.predict(X[:10]).shape == (10,)

    def test_loss_decreases(self, seq_data):
        from empirlab.dl.lstm import LSTMForecaster
        X, y = seq_data
        model = LSTMForecaster(hidden_size=32, n_layers=1, epochs=10,
                                batch_size=32, lr=1e-2)
        model.fit(X, y, verbose=False)
        # Training loss history should be decreasing (at least last < first)
        history = model.history_
        assert history[-1] < history[0], "Loss did not decrease during training"


class TestTrainLoop:
    def test_train_loop_returns_history(self):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from empirlab.dl.trainer import train_loop

        model = nn.Linear(5, 1)
        X = torch.randn(100, 5); y = torch.randn(100)
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=20)

        history = train_loop(model, loader, val_loader=None, epochs=3, lr=1e-3)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3
