"""empirlab — Causal ML, Deep Learning, RL and LLMs for Economics & Finance."""
__version__ = "0.1.0"
__author__  = "mofs0"

# Lazy imports: submodules are imported on demand to avoid hard dependency
# failures when optional extras (torch, gymnasium, sklearn…) are absent.
def __getattr__