"""
Synthetic Difference-in-Differences
=====================================
Reference: Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W., &
           Wager, S. (2021). Synthetic difference-in-differences.
           American Economic Review, 111(12), 4088-4118.
           https://doi.org/10.1257/aer.20190159

Status: API stub — full implementation coming in v0.2.
"""
from __future__ import annotations
import numpy as np


class SyntheticDiD:
    """Synthetic DiD estimator.

    Parameters
    ----------
    zeta : float or None   Regularisation weight (None = auto-select).
    """

    def __init__(self, zeta: float = None):
        self.zeta = zeta

    def fit(self, Y: np.ndarray, treated_units: list, T_pre: int):
        """
        Y            : (N, T) outcome panel matrix
        treated_units: list of treated unit indices
        T_pre        : number of pre-treatment periods
        """
        raise NotImplementedError(
            "SyntheticDiD coming in empirlab v0.2. "
            "Reference R implementation: https://github.com/synth-inference/synthdid"
        )

    def estimate(self) -> dict:
        raise NotImplementedError

    def bootstrap_se(self, n_boot: int = 200) -> float:
        raise NotImplementedError
