"""LLM-assisted data annotation for economic text.

Status: API stub — full implementation coming in v0.3.
"""
from __future__ import annotations


class LLMAnnotator:
    """Label economic text via LLM few-shot or zero-shot annotation.

    Parameters
    ----------
    model       : str, default 'gpt-4o'
    labels      : list[str]   Possible annotation categories.
    examples    : list[dict]  Few-shot examples: [{'text':..., 'label':...}]
    temperature : float, default 0.0
    """

    def __init__(self, model="gpt-4o", labels=None,
                 examples=None, temperature=0.0):
        self.model       = model
        self.labels      = labels or []
        self.examples    = examples or []
        self.temperature = temperature

    def annotate(self, texts: list[str]) -> list[str]:
        """Return predicted labels for each text."""
        raise NotImplementedError("LLMAnnotator coming in v0.3")

    def annotate_df(self, df, text_col: str, label_col: str = "label"):
        raise NotImplementedError
