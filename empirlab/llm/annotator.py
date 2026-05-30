"""
LLM-based Data Annotator for Economics
========================================
Structured annotation of text data using GPT-4o with JSON schema enforcement.

Use cases in economics/finance research:
  - Classify news articles by topic (monetary policy, fiscal policy, ...)
  - Label qualitative survey responses
  - Extract structured information from earnings call transcripts
  - Code open-ended interview data with custom codebooks
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional


class LLMAnnotator:
    """Batch text annotator backed by GPT-4o structured outputs.

    Parameters
    ----------
    task_description : str
        Plain-language description of the annotation task.
    labels : list of str or None
        For classification: valid label strings.
        None = free-form extraction (use schema instead).
    schema : dict or None
        JSON schema for structured extraction (overrides labels).
    model : str, default 'gpt-4o-mini'
        OpenAI model. Use 'gpt-4o' for best accuracy.
    examples : list of dict or None
        Few-shot examples: [{"text": ..., "label": ...}, ...].
    batch_size : int, default 10
        Number of texts per API call (cost efficiency).
    openai_api_key : str or None

    Examples
    --------
    >>> from empirlab.llm import LLMAnnotator
    >>>
    >>> # Classification
    >>> ann = LLMAnnotator(
    ...     task_description="Classify the monetary policy stance of each sentence.",
    ...     labels=["hawkish", "dovish", "neutral"],
    ...     model="gpt-4o-mini",
    ... )
    >>> texts = [
    ...     "The Fed signalled aggressive rate hikes to combat inflation.",
    ...     "The committee voted to maintain rates at current levels.",
    ...     "Policymakers remain accommodative to support growth.",
    ... ]
    >>> results = ann.annotate(texts)
    >>> print(results)  # ["hawkish", "neutral", "dovish"]
    >>>
    >>> # Structured extraction
    >>> ext = LLMAnnotator(
    ...     task_description="Extract key info from earnings call sentences.",
    ...     schema={"guidance": "str", "sentiment": "positive|negative|neutral", "quarter": "str"},
    ... )
    >>> results = ext.annotate(["Q3 EPS guidance raised to $2.50, strong outlook."])
    """

    def __init__(self, task_description: str,
                 labels: Optional[list[str]] = None,
                 schema: Optional[dict] = None,
                 model: str = "gpt-4o-mini",
                 examples: Optional[list[dict]] = None,
                 batch_size: int = 10,
                 openai_api_key: Optional[str] = None):
        self.task_description = task_description
        self.labels      = labels
        self.schema      = schema
        self.model       = model
        self.examples    = examples or []
        self.batch_size  = batch_size
        self.api_key     = openai_api_key or os.environ.get("OPENAI_API_KEY", "")

        if labels is None and schema is None:
            raise ValueError("Provide either labels (classification) or schema (extraction).")

    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        lines = [
            f"Task: {self.task_description}",
            "",
        ]
        if self.labels:
            lines += [
                f"Valid labels: {self.labels}",
                "Return a JSON array where each element is EXACTLY one of the valid labels.",
                "Array length must equal the number of input texts.",
            ]
        else:
            lines += [
                f"Output schema per text: {json.dumps(self.schema)}",
                "Return a JSON array of objects, one per input text, matching the schema.",
            ]
        if self.examples:
            lines += ["", "Examples:"]
            for ex in self.examples:
                lines.append(json.dumps(ex))
        lines += ["", "Respond with ONLY a valid JSON array. No explanation."]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _annotate_batch(self, texts: list[str]) -> list[Any]:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        client  = OpenAI(api_key=self.api_key)
        system  = self._build_system_prompt()
        user_msg = json.dumps({f"text_{i}": t for i, t in enumerate(texts)}, ensure_ascii=False)

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content
                parsed = json.loads(raw)
                # Extract list from response (may be wrapped in a key)
                if isinstance(parsed, list):
                    return parsed
                # Try to find a list value
                for v in parsed.values():
                    if isinstance(v, list):
                        return v
                raise ValueError(f"Expected list in response, got: {parsed}")
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return []

    # ------------------------------------------------------------------
    def annotate(self, texts: list[str],
                 verbose: bool = False) -> list[Any]:
        """Annotate a list of texts.

        Parameters
        ----------
        texts   : list of str
        verbose : bool, default False

        Returns
        -------
        list of labels (str) or dicts (for schema extraction)
        """
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            if verbose:
                print(f"  Annotating batch {i//self.batch_size + 1} "
                      f"({len(batch)} texts) …")
            results.extend(self._annotate_batch(batch))
        return results

    # ------------------------------------------------------------------
    def annotate_df(self, df, text_col: str,
                    output_col: str = "label",
                    verbose: bool = False):
        """Annotate a DataFrame column in-place.

        Parameters
        ----------
        df         : pandas DataFrame
        text_col   : str   Column containing texts to annotate.
        output_col : str   Name for the new annotation column.
        verbose    : bool

        Returns
        -------
        DataFrame with new column added.
        """
        import pandas as pd
        texts  = df[text_col].tolist()
        labels = self.annotate(texts, verbose=verbose)
        df     = df.copy()
        df[output_col] = labels
        return df
