"""LLM agent for cleaning messy financial data.

Status: API stub — full implementation coming in v0.3.
"""
from __future__ import annotations
import pandas as pd


class LLMDataCleaner:
    """Use an LLM to identify and fix issues in tabular financial data.

    Capabilities (coming in v0.3)
    -----------------------------
    - Detect and impute outliers with explanation
    - Standardise company name variants (e.g. 'Tencent', '腾讯', 'TCEHY')
    - Parse mixed date / number formats
    - Flag suspicious rows for human review

    Parameters
    ----------
    model : str, default 'gpt-4o'
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("LLMDataCleaner coming in v0.3")

    def audit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of flagged issues without modifying the data."""
        raise NotImplementedError
