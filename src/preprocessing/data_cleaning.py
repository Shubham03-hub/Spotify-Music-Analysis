"""
Data cleaning for the merged Spotify dataset.

Handles exactly the issues found during profiling of this dataset:
  - 6 duplicate track_ids in tracks_metadata (kept: first occurrence, by
    release_date, on the assumption the earliest record is the canonical one;
    dropped duplicates are logged for manual review)
  - 16 null album_name values -> imputed as "Unknown Album"
  - 10 null duration_ms values -> imputed with the genre-level median duration
  - 22 null values scattered across audio feature columns -> imputed with the
    genre-level median for that feature (audio character is genre-dependent,
    so a genre-level median is a far better fill than a global median)

All imputations are logged with counts so cleaning is auditable, not silent.
"""

from typing import Dict

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

AUDIO_FEATURE_COLUMNS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "time_signature",
]


def drop_duplicate_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate track_ids from tracks_metadata, keeping the earliest
    release_date record."""
    before = len(df)
    df = df.sort_values("release_date").drop_duplicates(subset="track_id", keep="first")
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Dropped %s duplicate track_id row(s) from tracks_metadata (kept earliest release_date).", dropped)
    return df.reset_index(drop=True)


def drop_duplicate_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate track_ids from audio_features, keeping the first
    occurrence. Audio features carry no release_date to break ties on, so
    'first occurrence in file order' is the tiebreak — logged for visibility."""
    before = len(df)
    df = df.drop_duplicates(subset="track_id", keep="first")
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Dropped %s duplicate track_id row(s) from audio_features (kept first occurrence).", dropped)
    return df.reset_index(drop=True)


def impute_album_name(df: pd.DataFrame) -> pd.DataFrame:
    null_count = df["album_name"].isna().sum()
    if null_count > 0:
        df["album_name"] = df["album_name"].fillna("Unknown Album")
        logger.info("Imputed %s null album_name value(s) with 'Unknown Album'.", null_count)
    return df


def impute_duration(df: pd.DataFrame) -> pd.DataFrame:
    null_count = df["duration_ms"].isna().sum()
    if null_count > 0:
        genre_median = df.groupby("genre")["duration_ms"].transform("median")
        df["duration_ms"] = df["duration_ms"].fillna(genre_median)
        # Fallback for the rare case an entire genre group is null
        df["duration_ms"] = df["duration_ms"].fillna(df["duration_ms"].median())
        logger.info("Imputed %s null duration_ms value(s) with genre-level median.", null_count)
    return df


def impute_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in AUDIO_FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        if null_count > 0:
            genre_median = df.groupby("genre")[col].transform("median")
            df[col] = df[col].fillna(genre_median)
            df[col] = df[col].fillna(df[col].median())
            logger.info("Imputed %s null %s value(s) with genre-level median.", null_count, col)
    return df


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Final dtype pass after imputation (fillna can upcast ints to float)."""
    df["release_year"] = df["release_year"].astype(int)
    df["explicit"] = df["explicit"].astype(bool)
    df["duration_ms"] = df["duration_ms"].astype(float)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the post-merge cleaning sequence: impute, then enforce final dtypes.

    NOTE: track_id / audio_features deduplication happens BEFORE the merge
    (in preprocessing_pipeline.run_preprocessing), not here — both raw tables
    have duplicate track_ids independently, and merging first would multiply
    duplicates across the join (6 dup tracks x 6 dup audio rows in the worst
    case) instead of cleanly resolving them. This function assumes it is
    receiving an already-deduplicated, already-merged dataset.
    """
    df = impute_album_name(df)
    df = impute_duration(df)
    df = impute_audio_features(df)
    df = enforce_dtypes(df)

    remaining_nulls = df.isna().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if len(remaining_nulls) > 0:
        logger.warning("Nulls remain after cleaning:\n%s", remaining_nulls.to_string())
    else:
        logger.info("Cleaning complete. No remaining nulls in the dataset.")

    return df
