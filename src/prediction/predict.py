"""
Prediction pipeline for the Spotify Track Performance Intelligence system.

This is the ONLY module the dashboard talks to for scoring — it owns no
modeling logic itself, it just loads the trained pipelines via
src.utils.model_loader and applies them. Feature engineering is re-run
through src.features.feature_engineering so a single track scored here goes
through EXACTLY the same transformation as training data did.
"""

from typing import Dict, Optional

import pandas as pd

from src.training.feature_target_spec import get_feature_target_columns
from src.utils.helper import load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.model_loader import (
    load_classification_pipeline, load_label_encoder, load_model_metadata, load_regression_pipeline,
)

logger = get_logger(__name__)


def predict_popularity(df: pd.DataFrame, config: Dict = None) -> pd.Series:
    """Predict continuous popularity score for each row in a model-ready
    (already feature-engineered) DataFrame."""
    config = config or load_config()
    pipeline = load_regression_pipeline(config)
    categorical, numeric, _ = get_feature_target_columns(df, task="regression")
    preds = pipeline.predict(df[categorical + numeric])
    return pd.Series(preds, index=df.index, name="predicted_popularity").clip(0, 100)


def predict_tier(df: pd.DataFrame, config: Dict = None) -> pd.Series:
    """Predict success tier (Flop/Niche/Mid/Hit — Viral Hit is merged into
    Hit at modeling time, see feature_target_spec.prepare_classification_target)
    for each row in a model-ready DataFrame."""
    config = config or load_config()
    pipeline = load_classification_pipeline(config)
    label_encoder = load_label_encoder(config)
    categorical, numeric, _ = get_feature_target_columns(df, task="classification")
    encoded_preds = pipeline.predict(df[categorical + numeric])
    decoded = label_encoder.inverse_transform(encoded_preds)
    return pd.Series(decoded, index=df.index, name="predicted_tier")


def predict_tier_probabilities(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Full class probability distribution — used by the dashboard to show
    e.g. '62% Mid, 24% Hit, 10% Niche, 4% Flop' rather than only the winner."""
    config = config or load_config()
    pipeline = load_classification_pipeline(config)
    label_encoder = load_label_encoder(config)
    categorical, numeric, _ = get_feature_target_columns(df, task="classification")
    probs = pipeline.predict_proba(df[categorical + numeric])
    return pd.DataFrame(probs, columns=label_encoder.classes_, index=df.index)


def score_tracks(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Full scoring pass: adds predicted_popularity, predicted_tier, and
    per-class probabilities to a copy of the input DataFrame."""
    config = config or load_config()
    result = df.copy()
    result["predicted_popularity"] = predict_popularity(df, config)
    result["predicted_tier"] = predict_tier(df, config)
    probs = predict_tier_probabilities(df, config)
    probs.columns = [f"prob_{c.replace(' ', '_')}" for c in probs.columns]
    result = pd.concat([result, probs], axis=1)
    logger.info("Scored %s tracks.", len(result))
    return result


def score_full_catalog(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Convenience entrypoint: score the entire processed model-ready
    dataset, used by main.py and the dashboard's 'Download Predictions'."""
    config = load_config(config_path)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    df = pd.read_csv(processed_dir / config["processed_files"]["model_ready"])
    return score_tracks(df, config)


if __name__ == "__main__":
    scored = score_full_catalog()
    config = load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    out_path = processed_dir / "scored_catalog.csv"
    scored.to_csv(out_path, index=False)
    logger.info("Wrote scored catalog to %s", out_path)
