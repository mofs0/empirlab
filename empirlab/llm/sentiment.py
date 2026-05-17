"""
Financial Sentiment Analysis
=============================
Reference: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P.
           (2014). Good debt or bad debt: Detecting semantic orientations
           in economic texts. JASIST, 65(4), 782-796.
           https://doi.org/10.1002/asi.23062

Backends
--------
- 'finbert' : HuggingFace ProsusAI/finbert (requires: pip install transformers torch)
- 'gpt-4o'  : OpenAI API         (requires: pip install openai, OPENAI_API_KEY env var)
"""
from __future__ import annotations


class FinSentiment:
    """Sentiment scorer for financial text.

    Parameters
    ----------
    model      : str, default 'finbert'
    batch_size : int, default 32
    device     : str or None   'cpu' | 'cuda' | None (auto)

    Examples
    --------
    >>> pipe = FinSentiment(model='finbert')
    >>> scores = pipe.score(['Earnings beat expectations', 'Revenue missed targets'])
    >>> print(scores)   # [0.82, -0.71]
    """

    _LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    def __init__(self, model: str = "finbert", batch_size: int = 32,
                 device: str = None):
        self.model_name = model
        self.batch_size = batch_size
        self.device     = device
        self._pipe      = None

    # ── FinBERT ──────────────────────────────────────────────────────────
    def _load_finbert(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("pip install transformers torch")
        import torch
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=0 if dev == "cuda" else -1,
            truncation=True,
            max_length=512,
        )

    def _score_finbert(self, texts: list[str]) -> list[float]:
        if self._pipe is None:
            self._load_finbert()
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out   = self._pipe(batch)
            results.extend(
                self._LABEL_MAP.get(r["label"].lower(), 0.0) * r["score"]
                for r in out
            )
        return results

    # ── GPT-4o ───────────────────────────────────────────────────────────
    def _score_gpt(self, texts: list[str]) -> list[float]:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        client = OpenAI()
        scores = []
        for text in texts:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial sentiment analyser. "
                            "Respond with a SINGLE float in [-1, 1] where "
                            "-1 = very negative, 0 = neutral, 1 = very positive. "
                            "No other text."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=10,
            )
            try:
                scores.append(float(resp.choices[0].message.content.strip()))
            except ValueError:
                scores.append(0.0)
        return scores

    # ── Public API ────────────────────────────────────────────────────────
    def score(self, texts: list[str] | str) -> list[float]:
        """Score texts. Returns a list of floats in [-1, 1]."""
        if isinstance(texts, str):
            texts = [texts]
        if self.model_name == "finbert":
            return self._score_finbert(texts)
        if self.model_name == "gpt-4o":
            return self._score_gpt(texts)
        raise ValueError(
            f"Unknown model '{self.model_name}'. Choose 'finbert' or 'gpt-4o'."
        )

    def score_df(self, texts: list[str]):
        """Return a DataFrame with columns: text, sentiment."""
        import pandas as pd
        return pd.DataFrame({"text": texts, "sentiment": self.score(texts)})
