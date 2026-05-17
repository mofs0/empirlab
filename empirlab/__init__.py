"""empirlab — Causal ML, Deep Learning, RL and LLMs for Economics & Finance."""
__version__ = "0.1.0"
__author__  = "mofs0"

from empirlab.causal.dml import DoubleML
from empirlab.causal.datasets import make_plr_data, make_hte_data

__all__ = ["DoubleML", "make_plr_data", "make_hte_data"]
