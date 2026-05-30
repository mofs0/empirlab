# Contributing to empirlab

Thank you for your interest in contributing! This guide covers the development workflow.

## Setup

```bash
git clone https://github.com/mofs0/empirlab.git
cd empirlab
pip install -e ".[dev]"      # installs pytest, ruff, black
```

## Running Tests

```bash
pytest tests/ -v             # core tests (no torch / gymnasium required)
pytest tests/test_dl.py -v   # requires: pip install torch
pytest tests/test_rl.py -v   # requires: pip install gymnasium
```

All tests must pass before submitting a PR. New estimators must come with tests.

## Code Style

We use **ruff** for linting and **black** for formatting (line length 100):

```bash
ruff check empirlab/
black empirlab/ tests/
```

The CI pipeline runs both checks automatically on every push.

## Adding a New Estimator

1. Create `empirlab/<module>/your_estimator.py`
2. Follow the scikit-learn API:
   - `fit(X, y, ...) → self`
   - `predict(X) → np.ndarray`
   - `summary() → pd.DataFrame`
3. Add a class docstring with:
   - Paper reference (authors, year, journal, DOI)
   - Key equation numbers
   - A minimal `Examples` block that runs without external data
4. Add tests in `tests/test_<module>.py`
5. Export from `empirlab/<module>/__init__.py`
6. Add a notebook in `notebooks/<module>/`
7. Update `README.md` module overview table and roadmap

## Notebook Standards

Each notebook must:
- Have a clear paper reference in the first cell
- Work with synthetic data (no proprietary datasets required)
- Include at least one visualisation
- Run top-to-bottom in a clean kernel

## Versioning

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Breaking API changes → MAJOR
- New estimators / features → MINOR
- Bug fixes → PATCH

## Roadmap Modules (v0.2 targets)

| Module | Status |
|--------|--------|
| `causal.DRLearner` | ✅ Done |
| `causal.SyntheticDiD` | ✅ Done |
| `causal.PostLassoIV` | ✅ Done |
| `finance.MLPortfolio` | ✅ Done |
| `rl.PortfolioEnv` | ✅ Done |
| `rl.PPOAgent` | ✅ Done |
| `llm.LitReviewRAG` | ✅ Done |
| `llm.LLMAnnotator` | ✅ Done |
| `dl.TemporalFusionTransformer` | 🔄 v0.3 |
| `rl.SACAgent` | 🔄 v0.3 |

## Questions

Open a GitHub Issue or start a Discussion. PRs are welcome!
