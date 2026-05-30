"""
LSTM / GRU Sequence Forecaster
================================
References
----------
- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory.
  Neural Computation, 9(8), 1735-1780.
- Cho, K., et al. (2014). Learning phrase representations using RNN
  encoder-decoder for statistical machine translation. arXiv:1406.1078.
"""
from __future__ import annotations

import numpy as np
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as _e:
    raise ImportError(
        "PyTorch is required for empirlab.dl. Install with: pip install torch"
    ) from _e


class _RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 cell_type, dropout):
        super().__init__()
        cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = cls(input_size, hidden_size, num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.0)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])          # last time step → forecast


class LSTMForecaster:
    """Scikit-learn-style wrapper for LSTM / GRU time-series forecasting.

    Parameters
    ----------
    input_size  : int   Number of input features per time step.
    hidden_size : int, default 64
    num_layers  : int, default 2
    output_size : int, default 1   Forecast horizon (steps ahead).
    cell_type   : str, default 'lstm'   'lstm' or 'gru'
    dropout     : float, default 0.1
    lr          : float, default 1e-3
    epochs      : int,  default 50
    batch_size  : int,  default 64

    Examples
    --------
    >>> model = LSTMForecaster(input_size=5, hidden_size=64, epochs=30)
    >>> model.fit(X_train, y_train)   # X: (N, seq_len, features)
    >>> y_hat = model.predict(X_test)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1,
                 cell_type: str = "lstm", dropout: float = 0.1,
                 lr: float = 1e-3, epochs: int = 50,
                 batch_size: int = 64, device: str = None):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.output_size = output_size
        self.cell_type   = cell_type
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Train on (N, seq_len, input_size) input and (N,) target."""
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(1)

        self.net_ = _RNNNet(
            self.input_size, self.hidden_size, self.num_layers,
            self.output_size, self.cell_type, self.dropout,
        ).to(self.device)

        opt  = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        crit = nn.MSELoss()
        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=self.batch_size, shuffle=True)
        self.history_ = []

        for ep in range(1, self.epochs + 1):
            self.net_.train()
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(self.net_(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(xb)
            avg = total / len(X_t)
            self.history_.append(avg)
            if verbose and ep % 10 == 0:
                print(f"Epoch {ep:3d}/{self.epochs} | Loss {avg:.6f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on (N, seq_len, input_size) input."""
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.FloatTensor(X).to(self.device)).cpu().numpy()
        return out.squeeze()
