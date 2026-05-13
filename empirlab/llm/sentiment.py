"""Financial Sentiment — FinBERT (Malo et al. 2014) / GPT-4o. Stub — coming v0.3."""
class FinSentiment:
    def __init__(self, model="finbert", batch_size=32):
        self.model=model; self.batch_size=batch_size
    def score(self, texts: list) -> list:
        raise NotImplementedError("v0.3")
