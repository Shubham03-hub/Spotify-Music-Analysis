"""Tests for src.training.feature_target_spec and a smoke test of src.training.train_model."""

import numpy as np
import pandas as pd
import pytest

from src.training.feature_target_spec import get_feature_target_columns, prepare_classification_target


def test_prepare_classification_target_merges_viral_hit_into_hit():
    df = pd.DataFrame({"popularity_tier": ["Flop", "Hit", "Viral Hit", "Niche", "Mid"]})
    result = prepare_classification_target(df)
    assert "Viral Hit" not in result["popularity_tier_for_modeling"].values
    assert result.loc[result["popularity_tier"] == "Viral Hit", "popularity_tier_for_modeling"].iloc[0] == "Hit"


def test_prepare_classification_target_does_not_mutate_original_column():
    df = pd.DataFrame({"popularity_tier": ["Flop", "Viral Hit"]})
    result = prepare_classification_target(df)
    # Original 5-tier column must be preserved untouched for dashboard display
    assert list(result["popularity_tier"]) == ["Flop", "Viral Hit"]


def test_get_feature_target_columns_regression():
    df = pd.DataFrame({"genre": ["pop"], "danceability": [0.5], "popularity": [50.0], "extra_col": [1]})
    categorical, numeric, target = get_feature_target_columns(df, task="regression")
    assert target == "popularity"
    assert "genre" in categorical
    assert "danceability" in numeric


def test_get_feature_target_columns_classification():
    df = pd.DataFrame({"genre": ["pop"], "danceability": [0.5], "popularity_tier_for_modeling": ["Mid"]})
    categorical, numeric, target = get_feature_target_columns(df, task="classification")
    assert target == "popularity_tier_for_modeling"


def test_get_feature_target_columns_rejects_unknown_task():
    df = pd.DataFrame({"popularity": [1.0]})
    with pytest.raises(ValueError):
        get_feature_target_columns(df, task="not_a_real_task")


def test_get_feature_target_columns_filters_to_existing_columns_only():
    """If a configured feature column doesn't exist in df (e.g. a smaller
    test dataset), it should be silently excluded rather than raising."""
    df = pd.DataFrame({"popularity": [50.0], "danceability": [0.5]})  # missing most numeric features
    categorical, numeric, target = get_feature_target_columns(df, task="regression")
    assert numeric == ["danceability"]
    assert categorical == []


@pytest.fixture
def synthetic_training_df():
    """A larger synthetic dataset (30 rows) sized so 2-fold CV is meaningful,
    used for a genuine end-to-end smoke test of the training functions
    rather than mocking them."""
    rng = np.random.RandomState(42)
    n = 30
    return pd.DataFrame({
        "genre": rng.choice(["pop", "rock"], n),
        "artist_country": rng.choice(["US", "GB"], n),
        "artist_career_stage": rng.choice(["Rising", "Established"], n),
        "key": rng.randint(0, 12, n),
        "mode": rng.randint(0, 2, n),
        "time_signature": rng.choice([3, 4], n),
        "genre_meta_category": rng.choice(["Mainstream", "Niche"], n),
        "duration_ms": rng.randint(120000, 300000, n),
        "explicit": rng.choice([0, 1], n),
        "danceability": rng.rand(n),
        "energy": rng.rand(n),
        "loudness": rng.uniform(-20, 0, n),
        "speechiness": rng.rand(n),
        "acousticness": rng.rand(n),
        "instrumentalness": rng.rand(n),
        "liveness": rng.rand(n),
        "valence": rng.rand(n),
        "tempo": rng.uniform(60, 180, n),
        "release_year": rng.choice(range(2020, 2027), n),
        "release_month": rng.randint(1, 13, n),
        "release_quarter": rng.randint(1, 5, n),
        "release_day_of_week": rng.randint(0, 7, n),
        "is_weekend_release": rng.choice([0, 1], n),
        "track_age_years": rng.uniform(0, 6, n),
        "artist_prior_track_popularity": rng.uniform(0, 100, n),
        "artist_prior_avg_popularity": rng.uniform(0, 100, n),
        "artist_prior_track_count": rng.randint(0, 10, n),
        "artist_prior_popularity_std": rng.uniform(0, 20, n),
        "is_artist_debut_track": rng.choice([0, 1], n),
        "artist_career_years_at_release": rng.uniform(0, 20, n),
        "artist_momentum_score": rng.uniform(0, 100, n),
        "artist_followers_millions": rng.uniform(0, 50, n),
        "artist_artist_star_power": rng.uniform(0, 100, n),
        "duration_minutes": rng.uniform(2, 5, n),
        "energy_danceability": rng.rand(n),
        "valence_energy": rng.rand(n),
        "acoustic_electronic_balance": rng.uniform(-1, 1, n),
        "is_explicit_int": rng.choice([0, 1], n),
        "genre_is_rising": rng.choice([0, 1], n),
        "genre_is_declining": rng.choice([0, 1], n),
        "tempo_fits_genre_bpm_band": rng.rand(n),
        "popularity": rng.uniform(0, 100, n),
        "popularity_tier": rng.choice(["Flop", "Niche", "Mid", "Hit"], n),
    })


def test_train_regression_models_smoke(synthetic_training_df):
    from src.training.train_model import train_regression_models
    config = {"modeling": {"cv_folds": 2, "regression_models": ["linear_regression", "random_forest"]}}
    results = train_regression_models(synthetic_training_df, config)
    assert set(results.keys()) == {"linear_regression", "random_forest"}
    for result in results.values():
        assert "r2" in result.cv_scores
        assert result.fitted_pipeline is not None


def test_train_classification_models_smoke(synthetic_training_df):
    from src.training.train_model import train_classification_models
    config = {"modeling": {"cv_folds": 2, "classification_models": ["random_forest"]}}
    results, label_encoder = train_classification_models(synthetic_training_df, config)
    assert "random_forest" in results
    assert "f1_macro" in results["random_forest"].cv_scores
    assert label_encoder is not None
