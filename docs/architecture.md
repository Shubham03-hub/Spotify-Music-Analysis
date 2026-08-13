# Solution Architecture

```
Data Sources (tracks_metadata, audio_features, artist_metadata, genre_dictionary)
  |
  v
Data Ingestion        -> loads + merges the 4 raw CSVs on track_id / artist_id / genre
  |
  v
Data Validation        -> schema checks, null/dup checks, referential integrity
  |
  v
Data Cleaning           -> dedupe track_ids (BOTH tracks AND audio_features had
  |                          6 duplicates each, independently — see below),
  |                          genre-aware imputation
  v
Feature Engineering    -> leakage-safe artist history (prior-release only, via
  |                          shift+expanding sorted by release_date), genre-year
  |                          aggregates, interaction features
  v
EDA
  |
  v
Model Training          -> Linear Regression / RF / XGBoost / LightGBM (regression)
  |                          RF / XGBoost / LightGBM (classification)
  v
Experiment Tracking    -> MLflow logs params/metrics/artifacts per run
  |
  v
Model Registry         -> best regression + classification model versioned & registered
  |
  v
Prediction Pipeline    -> loads registered models (via joblib for fast dashboard access),
  |                          scores new/held-out tracks
  v
Dashboard              -> Streamlit: KPIs, track scorer, genre trend explorer, driver importance
  |
  v
Deployment             -> Dockerized, deployable to Streamlit Cloud / Render
```

## Why each component exists

- **Ingestion** is isolated from validation so raw data is never mutated —
  traceability if a source file changes.
- **Validation** runs before cleaning so structural problems are caught
  before they silently propagate into features. In practice this caught a
  real issue: both `tracks_metadata.csv` and `audio_features.csv`
  independently contained 6 duplicate `track_id`s. The initial
  implementation deduplicated only `tracks` and merged before deduplicating
  `audio_features`, which failed a strict 1:1 merge validation — the fix was
  to dedupe both tables *before* merging, not after.
- **Feature engineering** is a separate module from preprocessing so the
  same feature functions run identically at training time and prediction
  time (`src/prediction/predict.py` imports the same functions), avoiding
  train/serve skew. The single most important property here is leakage
  safety: artist history features (`artist_prior_track_popularity`,
  `artist_prior_avg_popularity`, etc.) are computed only from a given
  artist's releases strictly *before* the current track's `release_date`,
  via `shift()` + expanding statistics on data sorted by release date — not
  a same-artist average that would let a track "see" its own future
  sibling releases.
- **Two numeric features were found to be near-perfectly collinear**
  (`days_since_release` and `track_age_years`, r=0.999999) during
  evaluation — this was inflating Linear Regression's coefficients to
  implausible magnitudes (~150) that swamped every other driver in the
  importance chart. Fixed by dropping the redundant feature
  (`src/training/feature_target_spec.py`).
- **MLflow** sits between training and the registry so every experiment is
  reproducible and comparable. This project's MLflow version deprecated the
  plain filesystem tracking backend; the working configuration uses
  `sqlite:///mlruns/mlflow.db` per MLflow's own migration guidance.
- **Prediction pipeline** only ever talks to joblib-persisted "best" models
  (`models/*.joblib`), not directly to training code or a live MLflow
  connection — this is what lets the dashboard run standalone without
  needing the MLflow tracking store available at serve time. MLflow remains
  the system of record for experiment history and versioning.
- **Dashboard** is a thin consumption layer — it contains no modeling logic,
  only calls into `src/prediction`.

## Handling severe class imbalance

The raw catalog's `popularity_tier` distribution: Flop 466 / Niche 811 /
Mid 350 / Hit 65 / **Viral Hit 4** (post-cleaning, 1,696 tracks). With 4
examples, 5-fold stratified cross-validation cannot place at least one
member of that class in every fold, so `popularity_tier_for_modeling`
merges Viral Hit into Hit for training and evaluation purposes only — the
original 5-tier `popularity_tier` is preserved untouched for dashboard
display and business reporting (`src/training/feature_target_spec.py`).
