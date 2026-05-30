"""empirlab — Causal ML, Deep Learning, RL and LLMs for Economics & Finance."""
__version__ = "0.1.0"
__author__  = "mofs0"

# Lazy imports: submodules are imported on demand to avoid hard dependency
# failures when optional extras (torch, gymnasium, sklearn…) are absent.
def __getattr__(name):
    if name == "DoubleML":
        from empirlab.causal.dml import DoubleML
        return DoubleML
    if name in ("make_plr_data", "make_hte_data"):
        from empirlab.causal import datasets as _ds
        return getattr(_ds, name)
    raise AttributeError(f"module 'empirlab' has no attribute {name!r}")

__all__ = ["DoubleML", "make_plr_data", "make_hte_data"]
