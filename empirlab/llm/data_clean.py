"""
LLM-Based Data Cleaner for Economics Research
===============================================
Uses GPT-4o-mini to detect and fix common data quality issues
in tabular / panel datasets: outlier flags, OCR errors in text fields,
inconsistent company name normalisation, unit mismatches, etc.

This is not a replacement for rule-based cleaning — it is designed
for the hard residual cases that regex cannot handle.
"""
from __future__ import annotations

import json
import os
from typing import Optional


class LLMDataCleaner:
    """LLM-assisted data cleaning for economic datasets.

    Parameters
    ----------
    model : str, default 'gpt-4o-mini'
    openai_api_key : str or None

    Examples
    --------
    >>> from empirlab.llm import LLMDataCleaner
    >>> cleaner = LLMDataCleaner()
    >>>
    >>> # Normalise company names
    >>> names = ["Apple Inc.", "APPLE INC", "Apple Incorporated", "Google LLC", "Alphabet Inc"]
    >>> print(cleaner.normalise_entities(names, entity_type="company"))
    >>> # ["Apple Inc.", "Apple Inc.", "Apple Inc.", "Google LLC", "Alphabet Inc."]
    >>>
    >>> # Detect likely data entry errors in a numeric column
    >>> import pandas as pd
    >>> df = pd.DataFrame({"revenue": [1200, 1350, 13500, 1400, 1250]})  # 13500 likely 1350
    >>> flags = cleaner.flag_outliers_llm(df, "revenue", context="quarterly revenue in $M")
    >>> print(flags)
    """

    def __init__(self, model: str = "gpt-4o-mini",
                 openai_api_key: Optional[str] = None):
        self.model   = model
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    def _call(self, system: str, user: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    def normalise_entities(self, names: list[str],
                           entity_type: str = "company") -> list[str]:
        """Cluster and normalise entity name variants.

        Parameters
        ----------
        names       : list of raw name strings (may contain duplicates/variants)
        entity_type : str  e.g. 'company', 'country', 'journal', 'author'

        Returns
        -------
        list of str  Canonical form for each input name.
        """
        system = (
            f"You are a data normalisation expert. "
            f"Given a list of {entity_type} name variants, return the canonical (most formal) "
            f"name for each entry. Preserve order. Return a JSON array of strings only."
        )
        user = json.dumps(names, ensure_ascii=False)
        raw  = self._call(system, user)
        return json.loads(raw)

    # ------------------------------------------------------------------
    def flag_outliers_llm(self, df, col: str,
                          context: str = "",
                          n_examples: int = 10) -> list[bool]:
        """Ask the LLM to flag likely data-entry errors in a numeric column.

        Returns a boolean list: True = likely error, False = plausible value.
        """
        values = df[col].tolist()
        sample = values[:n_examples] if len(values) > n_examples else values
        system = (
            "You are a data quality expert for economics datasets. "
            "For each value in the list, decide if it is a likely data-entry error "
            "(e.g. off by factor of 10, unit mismatch, OCR error). "
            f"Context: {context or col}. "
            "Return a JSON array of booleans (true = likely error) matching the input length."
        )
        raw   = self._call(system, json.dumps(sample))
        flags = json.loads(raw)
        # Pad with False if sample was truncated
        return flags + [False] * (len(values) - len(flags))

    # ------------------------------------------------------------------
    def standardise_units(self, values: list[str],
                          target_unit: str) -> list[float | None]:
        """Convert heterogeneous textual values to a target numeric unit.

        Parameters
        ----------
        values      : list of strings like ["$1.2B", "1200M", "€950 million"]
        target_unit : str  e.g. 'USD millions'

        Returns
        -------
        list of float or None (None = could not convert)
        """
        system = (
            f"Convert each value to {target_unit}. "
            "Return a JSON array of numbers (null if conversion is impossible). "
            "No explanation, JSON only."
        )
        raw = self._call(system, json.dumps(values, ensure_ascii=False))
        return json.loads(raw)

    # ------------------------------------------------------------------
    def classify_missing(self, df, col: str,
                         context: str = "") -> list[str]:
        """Classify missing values by missingness mechanism.

        Returns one of: 'MCAR', 'MAR', 'MNAR', 'structural' per row.
        """
        import pandas as pd
        descriptions = []
        for _, row in df.iterrows():
            is_missing = pd.isna(row[col])
            other_vals = {k: v for k, v in row.items() if k != col}
            descriptions.append({"missing": is_missing, "other_cols": other_vals})

        system = (
            f"For each observation, classify the missingness of '{col}' as one of: "
            "'MCAR' (missing completely at random), 'MAR' (missing at random given other covariates), "
            "'MNAR' (missing not at random / informative), or 'structural' (always missing by design). "
            f"Context: {context}. Return a JSON array of strings."
        )
        raw = self._call(system, json.dumps(descriptions[:20], default=str))
        return json.loads(raw)
