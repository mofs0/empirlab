"""ML Factor Model — Gu, Kelly & Xiu (2020). RFS 33(5). Stub — coming v0.2."""
class MLFactorModel:
    def __init__(self, method="enet", n_factors=5):
        self.method=method; self.n_factors=n_factors
    def fit(self, returns, chars): raise NotImplementedError("v0.2")
    def predict(self, chars):      raise NotImplementedError("v0.2")
    def factor_decompose(self):    raise NotImplementedError("v0.2")
