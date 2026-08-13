# Contributing

## Setup

```bash
git clone <repo-url>
cd spotify-music-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running the pipeline

```bash
python main.py              # full pipeline: ingest -> validate -> clean -> features -> EDA -> train -> evaluate -> score
python main.py --skip-eda   # skip figure generation for faster iteration
```

Individual stages can be run standalone, e.g.:

```bash
python -m src.preprocessing.preprocessing_pipeline
python -m src.features.feature_engineering
python -m src.training.mlflow_tracking
python -m src.evaluation.evaluate_model
```

## Tests

```bash
pytest tests/ -v
```

All new features or bug fixes should include a corresponding test. Tests use
small synthetic fixtures (see `tests/conftest.py`), not the real catalog, so
the suite runs fast and doesn't depend on `data/raw/` being populated.

## Code style

- Every function has a docstring explaining *why*, not just *what* — especially
  around leakage-safety decisions in `src/features/` and `src/training/`.
- No bare `except:` — catch specific exceptions.
- Run `python -m pytest tests/` before opening a PR; all tests must pass.

## Adding a new feature to the model

1. Add the computation to `src/features/feature_engineering.py`.
2. Register it in `src/training/feature_target_spec.py`'s `NUMERIC_FEATURES`
   or `CATEGORICAL_FEATURES` — this is the single source of truth consumed by
   training, evaluation, and prediction, so nothing else needs to change.
3. Add a test in `tests/test_features.py`, especially if the feature touches
   `popularity` in any way — leakage-safety must be verified explicitly.
4. Re-run `python main.py` and confirm the leaderboard/evaluation numbers.

## Reporting issues

Open a GitHub issue with: what you ran, what you expected, what happened
instead, and the relevant lines from `reports/logs/pipeline.log`.
