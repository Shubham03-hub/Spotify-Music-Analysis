"""
Single source of truth for which columns are features vs targets vs
identifiers, used identically by training, evaluation, and prediction. This
is the piece that prevents train/serve skew and, just as importantly,
prevents target leakage: every column excluded here is excluded because it
either IS a target, IDENTIFIES a row (no predictive content), or is a raw
text field not yet vectorized.
"""

from typing import List, Tuple

import pandas as pd

# Columns that identify a row rather than describe it — never used as features.
ID_COLUMNS = [
    "track_id", "artist_id", "track_name", "album_name",
    "release_date", "genre_meta_description", "artist_artist_name",
    "genre_meta_genre_name",
]

# The three target-related columns. popularity is the regression target;
# popularity_tier (and its modeling-safe merge) is the classification target.
# Both are excluded from the feature set for the OTHER task, and — critically
# — popularity is excluded from the classification feature set too, since
# popularity_tier is deterministically derived from popularity and including
# both would let the classifier "cheat" by reading the regression target.
TARGET_COLUMNS = ["popularity", "popularity_tier", "popularity_tier_for_modeling"]

CATEGORICAL_FEATURES = [
    "genre", "artist_country", "artist_career_stage",
    "key", "mode", "time_signature", "genre_meta_category",
]

NUMERIC_FEATURES = [
    "duration_ms", "explicit", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "release_year", "release_month", "release_quarter",
    "release_day_of_week", "is_weekend_release", "track_age_years",
    # NOTE: days_since_release is deliberately excluded — it is r=0.999999
    # collinear with track_age_years (same signal, different units) and was
    # inflating linear regression coefficients on both to implausible
    # magnitudes (~150) that swamped every other driver. Keeping one.
    "artist_prior_track_popularity", "artist_prior_avg_popularity",
    "artist_prior_track_count", "artist_prior_popularity_std", "is_artist_debut_track",
    "artist_career_years_at_release", "artist_momentum_score", "artist_followers_millions",
    "artist_artist_star_power", "duration_minutes", "energy_danceability",
    "valence_energy", "acoustic_electronic_balance", "is_explicit_int",
    "genre_is_rising", "genre_is_declining", "tempo_fits_genre_bpm_band",
]


def prepare_classification_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the extremely rare 'Viral Hit' class (4 of 1,696 tracks) into
    'Hit' for MODELING purposes only. With 5-fold CV, a 4-member class
    cannot be stratified (fewer members than folds), and any model's
    "performance" on it would be noise, not signal. The original 5-tier
    popularity_tier is preserved untouched for dashboard display and
    business reporting — only popularity_tier_for_modeling collapses the two.
    """
    df = df.copy()
    df["popularity_tier_for_modeling"] = df["popularity_tier"].replace({"Viral Hit": "Hit"})
    return df


def get_feature_target_columns(
    df: pd.DataFrame, task: str
) -> Tuple[List[str], List[str], str]:
    """
    Returns (categorical_features, numeric_features, target_column) for the
    given task ('regression' or 'classification'), filtered to columns that
    actually exist in df (defensive against schema drift).
    """
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]

    if task == "regression":
        target = "popularity"
    elif task == "classification":
        target = "popularity_tier_for_modeling"
    else:
        raise ValueError(f"Unknown task '{task}'. Expected 'regression' or 'classification'.")

    return categorical, numeric, target
