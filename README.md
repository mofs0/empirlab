# empirlab

**Causal ML · Deep Learning · RL · LLMs for Economics & Finance Research**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![sklearn-compatible](https://img.shields.io/badge/sklearn-compatible-orange)](https://scikit-learn.org/)

> A research-grade Python toolkit for modern empirical methods in economics and finance.  
> Scikit-learn compatible API · Paper-linked notebooks · Statistical inference built-in

---

## Why empirlab?

Most causal ML packages (EconML, DoubleML, CausalML) are general-purpose.  
**empirlab** is built specifically for **economics & finance research**, with:

- Notebooks organised by **research question**, not method name
- Every estimator links to its **original paper** with exact equation numbers
- **Confidence intervals and standard errors by default** — not an afterthought
- Finance-native data loaders (A-share via AKShare, FRED macro, Yahoo Finance)
- LLM tooling for sentiment analysis, literature review, and data annotation

---

## Module Overview

| Module | What it solves | Key classes |
|--------|---------------|-------------|
| `empirlab.causal` | Treatment effect estimation | `DoubleML`, `CausalForest`, `DRLearner`, `SyntheticDiD`, `PostLassoIV` |
| `empirlab.finance` | Return prediction & factor models | `MLFactorModel`, `ReturnPredictor`, `MLPortfolio` |
| `empirlab.dl` | Sequence modeling for macro/finance | `LSTMForecaster`, `TemporalFusionTransformer` |
| `empirlab.rl` | Algorithmic trading & portfolio RL | `StockTradingEnv`, `PPOAgent`, `SACAgent` |
| `empirlab.llm` | Text data in economics | `FinSentiment`, `LitReviewRAG`, `LLMAnnotator` |
| `empirlab.utils` | Shared infrastructure | metrics, inference, viz, data IO |

---

## Quick Start

```bash
git clone https://github.com/mofs0/empirlab.git
cd empirlab
pip install -e .
```

### Double Machine Learning (Chernozhukov et al., 2018)

```python
from empirlab.causal import DoubleML
from empirlab.causal.datasets import make_plr_data

X, y, d = make_plr_data(n=2000, p=20, theta=1.2, seed=42)

dml = DoubleML(ml_l="lasso", ml_m="lasso", n_folds=5)
dml.fit(X, y, d)
print(dml.summary())
#            coef  std_err   t_stat  p_value  ci_lower  ci_upper  sig
# treatment  1.193    0.048   24.85   <0.001     1.099     1.287  ***
```

### Causal Forest — Heterogeneous Treatment Effects (Wager & Athey, 2018)

```python
from empirlab.causal import CausalForest
from empirlab.causal.datasets import make_hte_data

X, y, d, tau_true = make_hte_data(n=2000, p=10)

cf = CausalForest(n_estimators=2000).fit(X, y, d)
tau_hat = cf.predict(X)
lb, ub  = cf.confidence_interval(X, alpha=0.05)   # 95% CI per unit
print(cf.summary(X))
```

### ML Factor Model (Gu, Kelly & Xiu, 2020)

```python
from empirlab.finance import MLFactorModel
from empirlab.utils.metrics import ic

model = MLFactorModel(method="enet")
model.fit(chars_train, returns_train)
r_hat = model.predict(chars_test)
print(f"IC = {ic(r_hat, returns_test):.4f}")
print(f"OOS R² = {model.r2_oos(chars_test, returns_test):.4f}")
```

### Financial Sentiment (FinBERT / GPT-4o)

```python
from empirlab.llm import FinSentiment

pipe = FinSentiment(model="finbert")   # or "gpt-4o"
scores = pipe.score([
    "Earnings beat expectations by 15%",
    "Revenue missed targets amid weak demand",
])
# [0.87, -0.79]
```

### A-share Data Loader

```python
from empirlab.finance import load_ashare, load_fred

df  = load_ashare("000001", start="2018-01-01")   # Ping An Bank, forward-adjusted
gdp = load_fred("GDP", start="2000-01-01")
```

### Financial Metrics

```python
from empirlab.utils.metrics import sharpe, max_drawdown, ic

print(sharpe(daily_returns, periods=252))
print(max_drawdown(price_series))
```

---

## Notebooks

Each notebook replicates a landmark paper end-to-end:

| ID | Paper | Method | Status |
|----|-------|--------|--------|
| C01 | Chernozhukov et al. (2018) | DoubleML — PLR | ✅ Ready |
| C02 | Wager & Athey (2018) | Causal Forest — HTE | ✅ Ready |
| C03 | Arkhangelsky et al. (2021) | Synthetic DiD | 🔄 v0.2 |
| C04 | Belloni et al. (2012) | Post-LASSO IV | 🔄 v0.2 |
| F01 | Gu, Kelly & Xiu (2020) | ML Factor Model | ✅ Ready |
| F02 | Kozak, Nagel & Santosh (2020) | Walk-Forward Return Prediction | ✅ Ready |
| DL01 | He et al. (2016) | ResNet — Image Classification | ✅ Ready |
| DL02 | Vaswani et al. (2017) | Transformer — Seq2Seq | ✅ Ready |
| RL01 | Liu et al. (2021) | PPO Portfolio Rebalancing | ✅ Ready |
| L01 | Malo et al. (2014) | FinBERT Sentiment | ✅ Ready |
| L02 | — | RAG for Literature Review | 🔄 v0.3 |

---

## Package Structure

```text
empirlab/
├── empirlab/
│   ├── causal/
│   │   ├── dml.py              # DoubleML — Chernozhukov et al. 2018  ✅
│   │   ├── causal_forest.py    # CausalForest — Wager & Athey 2018   ✅
│   │   ├── dr_learner.py       # DRLearner — Kennedy 2023             🔄 v0.2
│   │   ├── synthetic_did.py    # SyntheticDiD — Arkhangelsky 2021     🔄 v0.2
│   │   ├── high_dim_iv.py      # PostLassoIV — Belloni et al. 2012    🔄 v0.2
│   │   └── datasets.py         # Benchmark DGPs                       ✅
│   │
│   ├── finance/
│   │   ├── factor_model.py     # MLFactorModel — Gu, Kelly & Xiu 2020 ✅
│   │   ├── return_pred.py      # Walk-forward ReturnPredictor          ✅
│   │   ├── portfolio.py        # MLPortfolio long-short                🔄 v0.2
│   │   └── data_loaders.py     # AKShare / FRED / yfinance             ✅
│   │
│   ├── dl/
│   │   ├── lstm.py             # LSTMForecaster (LSTM / GRU)           ✅
│   │   ├── tft.py              # TemporalFusionTransformer             🔄 v0.2
│   │   └── trainer.py          # Generic PyTorch training loop         ✅
│   │
│   ├── rl/
│   │   ├── envs/
│   │   │   ├── stock_env.py    # StockTradingEnv (Gym-compatible)      ✅
│   │   │   └── portfolio_env.py                                        🔄 v0.2
│   │   └── agents/
│   │       ├── ppo_agent.py    # PPOAgent                              🔄 v0.2
│   │       └── sac_agent.py    # SACAgent                             🔄 v0.2
│   │
│   ├── llm/
│   │   ├── sentiment.py        # FinSentiment (FinBERT + GPT-4o)       ✅
│   │   ├── rag.py              # LitReviewRAG                          🔄 v0.3
│   │   ├── annotator.py        # LLMAnnotator                          🔄 v0.3
│   │   └── data_clean.py       # LLMDataCleaner                        🔄 v0.3
│   │
│   └── utils/
│       ├── metrics.py          # sharpe, max_drawdown, ic, rmse …      ✅
│       ├── inference.py        # bootstrap_ci, bh_correction, HC3      ✅
│       ├── viz.py              # coef plot, event study, SHAP bar       ✅
│       └── data_io.py          # read/write panel + disk cache          ✅
│
├── notebooks/
│   ├── causal/    C01–C04
│   ├── finance/   F01–F02
│   ├── dl/        DL01–DL02 + DL_TEMPLATE
│   ├── rl/        RL01
│   └── llm/       L01–L02
│
├── tests/
│   ├── test_causal.py
│   ├── test_utils.py
│   └── test_finance.py
│
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
# Base (causal + finance + utils)
pip install -e .

# With deep learning support
pip install -e ".[dl]"

# With RL support
pip install -e ".[rl]"

# With LLM support
pip install -e ".[llm]"

# Everything
pip install -e ".[full]"
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Design Principles

**1. Scikit-learn API everywhere**  
Every estimator: `fit(X, y, ...) → self`, `predict(X) → array`, `summary() → DataFrame`.

**2. Statistics first**  
Standard errors, confidence intervals, and p-values always computed.  
Influence-function SEs, bootstrap CIs, and BH multiple-testing correction built-in.

**3. Paper-linked code**  
Every class docstring cites the exact paper, equation numbers, and key assumptions.

**4. Finance-native**  
Data loaders for A-share (AKShare), FRED macro, and Yahoo Finance with disk caching.  
Metrics that matter for finance: Sharpe ratio, max drawdown, IC, Calmar ratio.

---

## Roadmap

- [x] `causal`: DoubleML ✅, CausalForest ✅
- [ ] `causal`: DRLearner, SyntheticDiD, PostLassoIV — v0.2
- [x] `finance`: MLFactorModel ✅, ReturnPredictor ✅, data_loaders ✅
- [ ] `finance`: MLPortfolio walk-forward backtest — v0.2
- [x] `dl`: LSTMForecaster ✅, trainer ✅, ResNet/Transformer notebooks ✅
- [ ] `dl`: TemporalFusionTransformer — v0.2
- [x] `rl`: StockTradingEnv ✅
- [ ] `rl`: PortfolioEnv, PPO, SAC — v0.2
- [x] `llm`: FinSentiment (FinBERT + GPT-4o) ✅
- [ ] `llm`: LitReviewRAG, LLMAnnotator — v0.3
- [ ] PyPI release `pip install empirlab`
- [ ] Sphinx documentation site

---

## Related Projects

| Project | Description |
|---------|-------------|
| [EconML](https://github.com/py-why/EconML) | Microsoft's causal ML — general-purpose |
| [DoubleML](https://github.com/DoubleML/doubleml-for-py) | Gold-standard DML implementation |
| [CausalML](https://github.com/uber/causalml) | Uber's uplift modelling toolkit |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | RL for quantitative finance |
| [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | Open-source financial LLMs |

---

## Citation

```bibtex
@software{empirlab2025,
  author = {mofs0},
  title  = {empirlab: Causal ML and Modern AI for Economics \& Finance},
  year   = {2025},
  url    = {https://github.com/mofs0/empirlab}
}
```

---

MIT License · © 2025 mofs0
