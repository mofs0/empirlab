# Changelog

All notable changes to this project will be documented here.

## [0.2.0] — 2025

### Added
- `empirlab.causal.SyntheticDiD` — full implementation of Arkhangelsky et al. (2021) with unit/time weight optimisation, placebo bootstrap SE, and `summary()` DataFrame output
- `empirlab.causal.PostLassoIV` — high-dimensional IV estimation (Belloni et al. 2012): LASSO first stage (CV or BC penalty), Post-LASSO OLS, 2SLS second stage with HC3 robust SEs and first-stage F-statistic
- `empirlab.causal.DRLearner` — doubly-robust CATE learner (Kennedy 2023): cross-fitted outcome models + propensity score, DR pseudo-outcomes, flexible final regressor
- `empirlab.finance.MLPortfolio` — walk-forward long-short portfolio backtest with equal/value weighting, Sharpe/drawdown performance summary
- Notebooks C03, C04 — fully runnable with synthetic data demonstrations
- Notebooks F02, RL01 — expanded with walk-forward backtests and equity curve plots
- `tests/test_causal_v2.py` — 14 new unit tests covering SyntheticDiD, PostLassoIV, DRLearner
- `pyproject.toml` — bumped to v0.2.0, added `pandas-datareader` to finance extras, added `[tool.pytest.ini_options]`

### Changed
- `pyproject.toml` version 0.1.0 → 0.2.0

## [0.1.0] — 2025

### Added
- `empirlab.causal`: `DoubleML`, `CausalForest`, stub placeholders for DRLearner/SyntheticDiD/PostLassoIV
- `empirlab.finance`: `MLFactorModel`, `ReturnPredictor`, `data_loaders`
- `empirlab.dl`: `LSTMForecaster`, `trainer`
- `empirlab.rl`: `StockTradingEnv`
- `empirlab.llm`: `FinSentiment` (FinBERT + GPT-4o)
- `empirlab.utils`: `metrics`, `inference`, `viz`, `data_io`
- Notebooks C01, C02, F01, DL01, DL02, L01 (fully runnable)
- Full test suite: `test_causal.py`, `test_utils.py`, `test_finance.py`
- README.md with module overview, Quick Start, notebook table, design principles
