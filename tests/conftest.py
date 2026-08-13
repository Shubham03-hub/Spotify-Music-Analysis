"""
Shared pytest fixtures. Tests use small SYNTHETIC datasets that match the
real schema (not the actual uploaded catalog) so the suite is fast, doesn't
depend on data/raw/ being present, and can deliberately construct edge cases
(nulls, duplicates, orphan foreign keys) that may not exist in the current
real data but should still be caught if they ever appear.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_tracks_df() -> pd.DataFrame:
    return pd.DataFrame({
        "track_id": ["T1", "T2", "T3", "T4"],
        "artist_id": ["A1", "A1", "A2", "A3"],
        "track_name": ["Song A", "Song B", "Song C", "Song D"],
        "album_name": ["Album X", None, "Album Y", "Album Z"],
        "genre": ["pop", "pop", "rock", "jazz"],
        "release_date": ["2022-01-15", "2023-06-01", "2021-03-10", "2024-11-20"],
        "release_year": [2022, 2023, 2021, 2024],
        "duration_ms": [180000, 200000, None, 210000],
        "explicit": [False, True, False, False],
        "popularity": [45.0, 60.0, 20.0, 80.0],
        "popularity_tier": ["Mid", "Hit", "Flop", "Hit"],
    })


@pytest.fixture
def sample_audio_features_df() -> pd.DataFrame:
    return pd.DataFrame({
        "track_id": ["T1", "T2", "T3", "T4"],
        "danceability": [0.7, 0.6, 0.3, 0.8],
        "energy": [0.6, 0.7, 0.4, 0.9],
        "key": [1, 5, 3, 7],
        "loudness": [-6.0, -5.0, -10.0, -4.0],
        "mode": [1, 0, 1, 1],
        "speechiness": [0.05, 0.1, 0.03, 0.2],
        "acousticness": [0.2, 0.1, 0.6, 0.05],
        "instrumentalness": [0.0, 0.0, 0.1, 0.0],
        "liveness": [0.1, 0.15, 0.2, 0.1],
        "valence": [0.5, 0.6, 0.3, 0.7],
        "tempo": [120.0, 128.0, 90.0, 140.0],
        "time_signature": [4, 4, 3, 4],
    })


@pytest.fixture
def sample_artists_df() -> pd.DataFrame:
    return pd.DataFrame({
        "artist_id": ["A1", "A2", "A3"],
        "artist_name": ["Artist One", "Artist Two", "Artist Three"],
        "primary_genre": ["pop", "rock", "jazz"],
        "country": ["US", "GB", "FR"],
        "career_stage": ["Established", "Rising", "Veteran"],
        "followers_millions": [5.0, 0.5, 2.0],
        "artist_star_power": [70.0, 30.0, 55.0],
        "debut_year": [2015, 2020, 2005],
    })


@pytest.fixture
def sample_genres_df() -> pd.DataFrame:
    return pd.DataFrame({
        "genre_code": ["pop", "rock", "jazz"],
        "genre_name": ["Pop", "Rock", "Jazz"],
        "category": ["Mainstream", "Mainstream", "Niche"],
        "typical_bpm_range": ["100-130", "110-140", "60-120"],
        "trend_direction_2020_2026": ["rising", "declining", "rising"],
        "description": ["Mainstream pop music", "Classic rock", "Jazz standards"],
    })


@pytest.fixture
def sample_tables(sample_tracks_df, sample_audio_features_df, sample_artists_df, sample_genres_df):
    return {
        "tracks": sample_tracks_df,
        "audio_features": sample_audio_features_df,
        "artists": sample_artists_df,
        "genres": sample_genres_df,
    }
