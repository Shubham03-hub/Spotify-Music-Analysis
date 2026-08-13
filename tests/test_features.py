"""
Tests for src.features.feature_engineering.

The most important property this suite checks is LEAKAGE SAFETY: artist
history features must never include a track's own popularity, and must only
reflect releases strictly before it. This is checked directly, not just
inferred from a metric looking reasonable.
"""

import pandas as pd

from src.features.feature_engineering import (
    add_artist_history_features,
    add_date_features,
    add_interaction_features,
    build_genre_year_aggregates,
    engineer_track_features,
)
from src.ingestion.data_ingestion import merge_tables


def _merged(sample_tables):
    return merge_tables(sample_tables)


def test_date_features_extract_correct_calendar_parts(sample_tables):
    df = add_date_features(_merged(sample_tables))
    row = df[df["track_id"] == "T1"].iloc[0]  # release_date = 2022-01-15
    assert row["release_month"] == 1
    assert row["release_quarter"] == 1


def test_artist_debut_track_has_no_prior_history(sample_tables):
    """Artist A1's EARLIEST track (T1, released first) should have no prior
    history — is_artist_debut_track must be 1 and prior_track_count 0."""
    df = add_artist_history_features(_merged(sample_tables))
    t1 = df[df["track_id"] == "T1"].iloc[0]
    assert t1["is_artist_debut_track"] == 1
    assert t1["artist_prior_track_count"] == 0


def test_artist_history_never_includes_own_popularity(sample_tables):
    """
    A1 has two tracks: T1 (2022-01-15, popularity=45) and T2 (2023-06-01,
    popularity=60). T2's artist_prior_track_popularity must equal T1's
    popularity (45), NOT include T2's own value (60), and must NOT be an
    average that blends both.
    """
    df = add_artist_history_features(_merged(sample_tables))
    t2 = df[df["track_id"] == "T2"].iloc[0]
    assert t2["artist_prior_track_popularity"] == 45.0
    assert t2["artist_prior_avg_popularity"] == 45.0
    assert t2["is_artist_debut_track"] == 0


def test_artist_history_is_chronologically_ordered_not_index_ordered(sample_tables):
    """Feed rows out of chronological order and confirm the feature still
    reflects true release-date order, not DataFrame row order."""
    tables = sample_tables
    # Shuffle tracks so T2 (later release) appears BEFORE T1 (earlier release) in the frame
    tables["tracks"] = tables["tracks"].iloc[[1, 0, 2, 3]].reset_index(drop=True)
    df = add_artist_history_features(merge_tables(tables))
    t2 = df[df["track_id"] == "T2"].iloc[0]
    # T2 was released AFTER T1 regardless of row order, so it must still see T1 as history
    assert t2["artist_prior_track_popularity"] == 45.0


def test_interaction_features_are_computed_correctly(sample_tables):
    df = add_interaction_features(_merged(sample_tables))
    row = df[df["track_id"] == "T1"].iloc[0]
    assert abs(row["energy_danceability"] - (0.6 * 0.7)) < 1e-9
    assert row["duration_minutes"] == 180000 / 60000.0


def test_genre_year_aggregates_group_correctly(sample_tables):
    agg = build_genre_year_aggregates(_merged(sample_tables))
    pop_2022 = agg[(agg["genre"] == "pop") & (agg["release_year"] == 2022)]
    assert len(pop_2022) == 1
    assert pop_2022.iloc[0]["track_count"] == 1


def test_engineer_track_features_produces_no_nulls_on_clean_input(sample_tables):
    df = _merged(sample_tables)
    df["album_name"] = df["album_name"].fillna("Unknown Album")
    df["duration_ms"] = df["duration_ms"].fillna(df["duration_ms"].median())
    result = engineer_track_features(df)
    assert result.isna().sum().sum() == 0
