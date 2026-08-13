"""
Power BI data export for the Spotify Track Performance Intelligence system.

Reshapes the pipeline's flat scored_catalog.csv into a star schema (one
fact table + dimension tables) that Power BI's data model handles far
better than one wide denormalized table — smaller file, faster refresh,
and relationships you can actually build slicers/filters on.

Run with: python -m src.export.powerbi_export
(after `python main.py` has produced data/processed/scored_catalog.csv)
"""

import pandas as pd

from src.utils.helper import load_config, resolve_path, write_csv_safe
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_dim_artist(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "artist_id": "artist_id",
        "artist_artist_name": "artist_name",
        "artist_country": "artist_country",
        "artist_career_stage": "career_stage",
        "artist_followers_millions": "followers_millions",
        "artist_artist_star_power": "star_power",
        "artist_debut_year": "debut_year",
    }
    dim = df[list(cols.keys())].drop_duplicates(subset="artist_id").rename(columns=cols)
    return dim.sort_values("artist_id").reset_index(drop=True)


def build_dim_genre(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "genre": "genre_code",
        "genre_meta_genre_name": "genre_name",
        "genre_meta_category": "category",
        "genre_meta_typical_bpm_range": "typical_bpm_range",
        "genre_meta_trend_direction_2020_2026": "labeled_trend_direction",
    }
    dim = df[list(cols.keys())].drop_duplicates(subset="genre").rename(columns=cols)
    return dim.sort_values("genre_code").reset_index(drop=True)


def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    """One row per release_year present in the data — simple, since this
    dataset has no need for a full day-grain calendar table."""
    years = sorted(df["release_year"].unique())
    return pd.DataFrame({
        "release_year": years,
        "decade": [f"{(y // 10) * 10}s" for y in years],
        "is_current_year": [y == max(years) for y in years],
    })


def build_fact_tracks(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "track_id", "artist_id", "genre", "release_year", "release_date",
        "track_name", "album_name", "duration_minutes", "is_explicit_int",
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo",
        "popularity", "popularity_tier",
        "predicted_popularity", "predicted_tier",
        "prob_Flop", "prob_Hit", "prob_Mid", "prob_Niche",
        "track_age_years", "is_artist_debut_track",
    ]
    keep = [c for c in keep if c in df.columns]
    fact = df[keep].rename(columns={
        "genre": "genre_code",
        "is_explicit_int": "is_explicit",
        "popularity": "actual_popularity",
        "popularity_tier": "actual_tier",
    })
    fact["prediction_error"] = fact["actual_popularity"] - fact["predicted_popularity"]
    return fact


def build_genre_trend_fact(genre_year: pd.DataFrame) -> pd.DataFrame:
    return genre_year.rename(columns={"genre": "genre_code"})


def build_model_performance(reg_leaderboard: pd.DataFrame, clf_leaderboard: pd.DataFrame) -> pd.DataFrame:
    reg = reg_leaderboard.copy()
    reg["task"] = "regression"
    reg["primary_metric"] = "r2_mean"
    reg["primary_metric_value"] = reg["r2_mean"]

    clf = clf_leaderboard.copy()
    clf["task"] = "classification"
    clf["primary_metric"] = "f1_macro_mean"
    clf["primary_metric_value"] = clf["f1_macro_mean"]

    combined = pd.concat([
        reg[["model", "task", "primary_metric", "primary_metric_value"]],
        clf[["model", "task", "primary_metric", "primary_metric_value"]],
    ], ignore_index=True)
    return combined


def build_driver_importance(reg_importance: pd.DataFrame, clf_importance: pd.DataFrame) -> pd.DataFrame:
    def clean(df: pd.DataFrame, task: str) -> pd.DataFrame:
        out = df.copy()
        out["feature"] = out["feature"].str.split("__").str[-1].str.replace("_", " ").str.title()
        out["task"] = task
        return out

    reg = clean(reg_importance, "regression")
    clf = clean(clf_importance, "classification")
    return pd.concat([reg, clf], ignore_index=True)


def run_powerbi_export(config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    reports_dir = resolve_path(config["paths"]["reports_dir"])

    scored_path = processed_dir / "scored_catalog.csv"
    if not scored_path.exists():
        raise FileNotFoundError(
            f"{scored_path} not found. Run `python main.py` first to generate scored predictions."
        )

    df = pd.read_csv(scored_path)
    genre_year = pd.read_csv(processed_dir / "genre_year_aggregates.csv")
    reg_leaderboard = pd.read_csv(reports_dir / "regression_leaderboard.csv")
    clf_leaderboard = pd.read_csv(reports_dir / "classification_leaderboard.csv")
    reg_importance = pd.read_csv(reports_dir / "driver_importance_regression.csv")
    clf_importance = pd.read_csv(reports_dir / "driver_importance_classification.csv")

    powerbi_dir = resolve_path("powerbi_data")
    powerbi_dir.mkdir(parents=True, exist_ok=True)

    write_csv_safe(build_fact_tracks(df), powerbi_dir / "fact_tracks.csv")
    write_csv_safe(build_dim_artist(df), powerbi_dir / "dim_artist.csv")
    write_csv_safe(build_dim_genre(df), powerbi_dir / "dim_genre.csv")
    write_csv_safe(build_dim_date(df), powerbi_dir / "dim_date.csv")
    write_csv_safe(build_genre_trend_fact(genre_year), powerbi_dir / "fact_genre_year_trend.csv")
    write_csv_safe(build_model_performance(reg_leaderboard, clf_leaderboard), powerbi_dir / "model_performance.csv")
    write_csv_safe(build_driver_importance(reg_importance, clf_importance), powerbi_dir / "driver_importance.csv")

    logger.info("Power BI export complete. Files written to %s", powerbi_dir)


if __name__ == "__main__":
    run_powerbi_export()
