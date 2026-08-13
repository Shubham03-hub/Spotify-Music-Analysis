"""
Feature engineering for the Spotify Track Performance Intelligence pipeline.

Two distinct outputs come out of this module, kept deliberately separate:

1. `engineer_track_features(df)` — track-level features used to predict a
   SINGLE track's popularity score / success tier. Any feature derived from
   `popularity` at this level is computed as a LEAKAGE-SAFE, backward-looking
   statistic (e.g. "this artist's average popularity on tracks released
   BEFORE this one"), never including the row's own popularity. This is the
   single most important correctness property of this module — a feature
   like "genre's average popularity including this track" would let the
   model see the answer.

2. `build_genre_year_aggregates(df)` — a separate, coarser genre x year
   table used ONLY for the genre trend-forecasting task. Aggregating across
   many tracks per genre-year is legitimate here because the task is
   explicitly about explaining/forecasting aggregate trend, not scoring an
   individual track — the same reason "national retail sales last quarter"
   is a legitimate input to a demand forecast even though it's an aggregate
   of the same kind of number you're trying to predict going forward.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.utils.helper import load_config, resolve_path, write_csv_safe
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Track-level features
# ---------------------------------------------------------------------------

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features derived from release_date."""
    df = df.copy()
    release_date = pd.to_datetime(df["release_date"])

    df["release_month"] = release_date.dt.month
    df["release_quarter"] = release_date.dt.quarter
    df["release_day_of_week"] = release_date.dt.dayofweek
    df["is_weekend_release"] = df["release_day_of_week"].isin([4, 5]).astype(int)  # Fri/Sat drops are common in music

    reference_date = release_date.max()
    df["days_since_release"] = (reference_date - release_date).dt.days
    df["track_age_years"] = (df["days_since_release"] / 365.25).round(2)

    return df


def add_artist_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-looking, leakage-safe artist history features: for each track,
    only statistics from that artist's PRIOR releases (by release_date) are
    used — never the current track's own popularity. The first track from
    any artist has no history, so these are NaN-filled with catalog-wide
    priors (median / zero), which is a fair prior for a debut release.
    """
    df = df.copy()
    df = df.sort_values(["artist_id", "release_date"]).reset_index(drop=True)

    grouped = df.groupby("artist_id")["popularity"]

    # shift(1) before expanding/rolling ensures the current row's own
    # popularity is never included in its own feature value.
    shifted = grouped.shift(1)
    df["artist_prior_track_popularity"] = shifted
    df["artist_prior_avg_popularity"] = shifted.groupby(df["artist_id"]).expanding().mean().reset_index(level=0, drop=True)
    df["artist_prior_track_count"] = shifted.groupby(df["artist_id"]).cumcount()
    df["artist_prior_popularity_std"] = (
        shifted.groupby(df["artist_id"]).expanding().std().reset_index(level=0, drop=True)
    )

    catalog_median_popularity = df["popularity"].median()
    df["artist_prior_track_popularity"] = df["artist_prior_track_popularity"].fillna(catalog_median_popularity)
    df["artist_prior_avg_popularity"] = df["artist_prior_avg_popularity"].fillna(catalog_median_popularity)
    df["artist_prior_popularity_std"] = df["artist_prior_popularity_std"].fillna(0.0)
    df["is_artist_debut_track"] = (df["artist_prior_track_count"] == 0).astype(int)

    return df


def add_artist_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derived from artist metadata (already joined in ingestion)."""
    df = df.copy()
    df["artist_career_years_at_release"] = (df["release_year"] - df["artist_debut_year"]).clip(lower=0)
    df["artist_momentum_score"] = (
        df["artist_artist_star_power"] * np.log1p(df["artist_followers_millions"])
    )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Audio-feature interactions with known relevance to perceived 'catchiness'."""
    df = df.copy()
    df["duration_minutes"] = df["duration_ms"] / 60000.0
    df["energy_danceability"] = df["energy"] * df["danceability"]
    df["valence_energy"] = df["valence"] * df["energy"]
    df["acoustic_electronic_balance"] = df["acousticness"] - df["energy"]
    df["is_explicit_int"] = df["explicit"].astype(int)
    return df


def add_genre_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genre-level CONTEXT features that do not leak target information: these
    come from genre_dictionary metadata (category, trend label, typical BPM
    band fit), not from aggregating this dataset's own popularity values.
    """
    df = df.copy()
    df["genre_is_rising"] = (df["genre_meta_trend_direction_2020_2026"] == "rising").astype(int)
    df["genre_is_declining"] = (df["genre_meta_trend_direction_2020_2026"] == "declining").astype(int)

    def _bpm_fit(row) -> float:
        try:
            low_str, high_str = str(row["genre_meta_typical_bpm_range"]).split("-")
            low, high = float(low_str), float(high_str)
        except (ValueError, AttributeError):
            return np.nan
        if low <= row["tempo"] <= high:
            return 1.0
        span = max(high - low, 1.0)
        distance = min(abs(row["tempo"] - low), abs(row["tempo"] - high))
        return max(0.0, 1.0 - (distance / span))

    df["tempo_fits_genre_bpm_band"] = df.apply(_bpm_fit, axis=1)
    return df


def engineer_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full track-level feature engineering sequence."""
    df = add_date_features(df)
    df = add_artist_history_features(df)
    df = add_artist_profile_features(df)
    df = add_interaction_features(df)
    df = add_genre_context_features(df)

    logger.info("Track-level feature engineering complete: %s rows x %s cols.", df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Genre x year aggregates — for genre trend forecasting only
# ---------------------------------------------------------------------------

def build_genre_year_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a genre x release_year aggregate table: average popularity, track
    volume, and average audio characteristics per genre per year. This is
    the input to genre trend forecasting — a fundamentally different,
    coarser-grained task than scoring one track, so aggregation here is
    appropriate rather than a leakage risk.
    """
    agg = (
        df.groupby(["genre", "release_year"])
        .agg(
            track_count=("track_id", "count"),
            avg_popularity=("popularity", "mean"),
            median_popularity=("popularity", "median"),
            avg_danceability=("danceability", "mean"),
            avg_energy=("energy", "mean"),
            avg_valence=("valence", "mean"),
            hit_rate=("popularity_tier", lambda s: (s.isin(["Hit", "Viral Hit"])).mean()),
        )
        .reset_index()
        .sort_values(["genre", "release_year"])
    )

    # Year-over-year growth rate in average popularity, per genre
    agg["popularity_yoy_growth"] = agg.groupby("genre")["avg_popularity"].pct_change()

    # Rolling volatility of popularity per genre (std of last 2 years available)
    agg["popularity_volatility"] = (
        agg.groupby("genre")["avg_popularity"].rolling(window=2, min_periods=1).std().reset_index(level=0, drop=True)
    )

    # Simple linear trend slope of avg_popularity vs release_year, per genre,
    # fit across all years available for that genre (used as the model-based
    # trend signal, compared against genre_dictionary's own trend label as a
    # sanity check rather than a training target).
    def _trend_slope(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return 0.0
        coeffs = np.polyfit(group["release_year"], group["avg_popularity"], deg=1)
        return float(coeffs[0])

    slopes = agg.groupby("genre").apply(_trend_slope, include_groups=False).rename("popularity_trend_slope")
    agg = agg.merge(slopes, on="genre", how="left")

    logger.info("Built genre x year aggregate table: %s rows across %s genres.", len(agg), agg["genre"].nunique())
    return agg


def run_feature_engineering(config_path: str = "config/config.yaml") -> Dict[str, pd.DataFrame]:
    config = load_config(config_path)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    cleaned_path = processed_dir / "cleaned_dataset.csv"

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"{cleaned_path} not found. Run src.preprocessing.preprocessing_pipeline first."
        )

    df = pd.read_csv(cleaned_path, parse_dates=["release_date"])
    df["release_date"] = df["release_date"].astype(str)  # normalize back for downstream date parsing

    track_features = engineer_track_features(df)
    genre_year_agg = build_genre_year_aggregates(df)

    write_csv_safe(track_features, processed_dir / config["processed_files"]["model_ready"])
    write_csv_safe(genre_year_agg, processed_dir / config["processed_files"]["genre_year_agg"])

    return {"track_features": track_features, "genre_year_agg": genre_year_agg}


if __name__ == "__main__":
    run_feature_engineering()
