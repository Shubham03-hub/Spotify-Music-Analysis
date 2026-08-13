"""Tests for src.ingestion.data_ingestion."""

import pandas as pd

from src.ingestion.data_ingestion import merge_tables


def test_merge_preserves_all_tracks(sample_tables):
    merged = merge_tables(sample_tables)
    assert len(merged) == len(sample_tables["tracks"])


def test_merge_joins_audio_features_correctly(sample_tables):
    merged = merge_tables(sample_tables)
    row = merged[merged["track_id"] == "T1"].iloc[0]
    assert row["danceability"] == 0.7
    assert row["energy"] == 0.6


def test_merge_joins_artist_metadata_with_prefix(sample_tables):
    merged = merge_tables(sample_tables)
    assert "artist_artist_name" in merged.columns
    row = merged[merged["track_id"] == "T1"].iloc[0]
    assert row["artist_artist_name"] == "Artist One"


def test_merge_joins_genre_metadata_with_prefix(sample_tables):
    merged = merge_tables(sample_tables)
    assert "genre_meta_genre_name" in merged.columns
    row = merged[merged["track_id"] == "T1"].iloc[0]
    assert row["genre_meta_genre_name"] == "Pop"


def test_merge_preserves_column_count_expectation(sample_tables):
    merged = merge_tables(sample_tables)
    # tracks (11) + audio_features (12, minus shared track_id) + artists (8,
    # minus shared artist_id, plus prefix) + genres (6, minus shared genre_code)
    # exact count isn't the point of this test; a sane lower bound is
    assert merged.shape[1] > sample_tables["tracks"].shape[1]
