"""
Temporal Fusion Transformer (TFT)
====================================
Reference: Lim, B., Arik, S.O., Loeff, N., & Pfister, T. (2021).
           Temporal fusion transformers for interpretable multi-horizon
           time series forecasting. International Journal of Forecasting,
           37(4), 1748-1764. https://doi.org/10.1016/j.ijforecast.2021.03.012

Status: API stub — full implementation coming in v0.2.
For now, use pytorch-forecasting.TemporalFusionTransformer.
"""
from __future__ import annotations
import numpy as np


class TemporalFusionTransformer:
    """TFT for multi-horizon interpretable forecasting.

    Parameters
    ----------
    d_model       : int, default 64
    n_heads       : int, default 4
    n_lstm_layers : int, default 2
    dropout       : float, default 0.1
    horizon       : int   Forecast horizon.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 n_lstm_layers: int = 2, dropout: float = 0.1,
                 horizon: int = 1):
        self.d_model       = d_model
        self.n_heads       = n_heads
        self.n_lstm_layers = n_lstm_layers
        self.dropout       = dropout
        self.horizon       = horizon

    def fit(self, X, y, **kwargs):
        raise NotImplementedError(
            "TFT coming in empirlab v0.2. "
            "Quick start: pip install pytorch-forecasting"
        )

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def plot_attention(self):
        raise NotImplementedError
