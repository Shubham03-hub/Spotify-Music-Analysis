"""
Exploratory Data Analysis for the Spotify Track Performance Intelligence
project.

Run as a script (not a notebook) so it's reproducible in CI/CD and reusable
by the dashboard's "Trend Analysis" tab. Every plot is saved to
reports/figures/; a plain-text summary of key stats is printed to stdout and
written to reports/eda_summary.txt so the numbers behind PHASE 5's business
interpretations are generated from real output, not asserted from memory.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — this runs outside a notebook/UI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.helper import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]


def load_data(config):
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    tracks = pd.read_csv(processed_dir / "model_ready_dataset.csv")
    genre_year = pd.read_csv(processed_dir / "genre_year_aggregates.csv")
    return tracks, genre_year


def dataset_overview(df: pd.DataFrame, summary_lines: list) -> None:
    summary_lines.append("=== DATASET OVERVIEW ===")
    summary_lines.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    summary_lines.append(f"Date range: {df['release_year'].min()}-{df['release_year'].max()}")
    summary_lines.append(f"Genres: {df['genre'].nunique()}, Artists: {df['artist_id'].nunique()}")
    summary_lines.append(f"Popularity: mean={df['popularity'].mean():.1f}, "
                          f"median={df['popularity'].median():.1f}, std={df['popularity'].std():.1f}")


def missing_and_duplicate_check(df: pd.DataFrame, summary_lines: list) -> None:
    summary_lines.append("\n=== MISSING VALUE / DUPLICATE CHECK (post-cleaning) ===")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    summary_lines.append(f"Remaining nulls: {len(nulls)} columns affected" if len(nulls) else "No remaining nulls.")
    summary_lines.append(f"Duplicate track_ids: {df['track_id'].duplicated().sum()}")


def outlier_analysis(df: pd.DataFrame, figures_dir: Path, summary_lines: list) -> None:
    summary_lines.append("\n=== OUTLIER ANALYSIS (IQR method, popularity + tempo + duration) ===")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["popularity", "tempo", "duration_ms"]):
        sns.boxplot(y=df[col], ax=ax, color="#1DB954")
        ax.set_title(f"{col} distribution")
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        summary_lines.append(f"{col}: {len(outliers)} outlier row(s) by IQR rule ({len(outliers)/len(df):.1%} of data)")
    plt.tight_layout()
    plt.savefig(figures_dir / "outlier_boxplots.png", dpi=120)
    plt.close()


def univariate_analysis(df: pd.DataFrame, figures_dir: Path, summary_lines: list) -> None:
    summary_lines.append("\n=== UNIVARIATE ANALYSIS ===")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, col in zip(axes.flat, AUDIO_FEATURES):
        sns.histplot(df[col], kde=True, ax=ax, color="#1DB954")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(figures_dir / "univariate_audio_features.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    df["popularity_tier"].value_counts().reindex(
        ["Flop", "Niche", "Mid", "Hit", "Viral Hit"]
    ).plot(kind="bar", ax=ax, color="#1DB954")
    ax.set_title("Popularity Tier Distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "popularity_tier_distribution.png", dpi=120)
    plt.close()

    tier_counts = df["popularity_tier"].value_counts()
    summary_lines.append(f"Popularity tier counts: {tier_counts.to_dict()}")
    summary_lines.append(
        f"Class imbalance ratio (Flop:Viral Hit) = "
        f"{tier_counts.get('Flop', 0)}:{tier_counts.get('Viral Hit', 0)}"
    )


def bivariate_analysis(df: pd.DataFrame, figures_dir: Path, summary_lines: list) -> None:
    summary_lines.append("\n=== BIVARIATE ANALYSIS ===")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    pairs = [
        ("danceability", "popularity"), ("energy", "popularity"), ("valence", "popularity"),
        ("acousticness", "popularity"), ("tempo", "popularity"), ("track_age_years", "popularity"),
    ]
    for ax, (x, y) in zip(axes.flat, pairs):
        sns.scatterplot(data=df, x=x, y=y, hue="genre", ax=ax, legend=False, alpha=0.5, s=15)
        corr = df[x].corr(df[y])
        ax.set_title(f"{x} vs {y} (r={corr:.2f})")
    plt.tight_layout()
    plt.savefig(figures_dir / "bivariate_popularity_relationships.png", dpi=120)
    plt.close()

    for x, y in pairs:
        corr = df[x].corr(df[y])
        summary_lines.append(f"corr({x}, popularity) = {corr:.3f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="genre", y="popularity", ax=ax)
    ax.set_title("Popularity by Genre")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "popularity_by_genre.png", dpi=120)
    plt.close()

    genre_medians = df.groupby("genre")["popularity"].median().sort_values(ascending=False)
    summary_lines.append(f"\nTop 3 genres by median popularity: {genre_medians.head(3).to_dict()}")
    summary_lines.append(f"Bottom 3 genres by median popularity: {genre_medians.tail(3).to_dict()}")


def correlation_analysis(df: pd.DataFrame, figures_dir: Path, summary_lines: list) -> None:
    summary_lines.append("\n=== CORRELATION ANALYSIS ===")
    numeric_cols = AUDIO_FEATURES + ["popularity", "track_age_years", "artist_followers_millions",
                                       "artist_artist_star_power", "duration_ms"]
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Correlation Matrix — Audio Features, Popularity, Artist Signals")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=120)
    plt.close()

    pop_corr = corr_matrix["popularity"].drop("popularity").sort_values(key=abs, ascending=False)
    summary_lines.append(f"Features most correlated with popularity (by |r|): {pop_corr.head(5).to_dict()}")


def multivariate_analysis(df: pd.DataFrame, figures_dir: Path, summary_lines: list) -> None:
    summary_lines.append("\n=== MULTIVARIATE ANALYSIS ===")
    fig, ax = plt.subplots(figsize=(9, 6))
    sample = df.sample(min(500, len(df)), random_state=42)
    scatter = ax.scatter(
        sample["danceability"], sample["energy"], c=sample["popularity"],
        cmap="viridis", alpha=0.7, s=25,
    )
    plt.colorbar(scatter, label="popularity")
    ax.set_xlabel("danceability")
    ax.set_ylabel("energy")
    ax.set_title("Danceability x Energy, colored by Popularity")
    plt.tight_layout()
    plt.savefig(figures_dir / "multivariate_dance_energy_popularity.png", dpi=120)
    plt.close()

    star_pop_corr_by_stage = df.groupby("artist_career_stage").apply(
        lambda g: g["artist_artist_star_power"].corr(g["popularity"]), include_groups=False
    )
    summary_lines.append(f"corr(star_power, popularity) by career stage: {star_pop_corr_by_stage.to_dict()}")


def genre_trend_analysis(genre_year: pd.DataFrame, genres_dict_path: Path, figures_dir: Path, summary_lines: list) -> None:
    summary_lines.append("\n=== GENRE TREND ANALYSIS (cross-check against genre_dictionary labels) ===")
    genre_dict = pd.read_csv(genres_dict_path)

    fig, ax = plt.subplots(figsize=(11, 6))
    for genre in genre_year["genre"].unique():
        sub = genre_year[genre_year["genre"] == genre]
        ax.plot(sub["release_year"], sub["avg_popularity"], marker="o", label=genre, alpha=0.7)
    ax.set_title("Average Popularity by Genre, 2020-2026")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Average Popularity")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "genre_trend_lines.png", dpi=120)
    plt.close()

    computed_slopes = genre_year.groupby("genre")["popularity_trend_slope"].first()
    merged = genre_dict.set_index("genre_code")[["trend_direction_2020_2026"]].join(
        computed_slopes.rename("computed_slope")
    )
    merged["slope_agrees_with_label"] = (
        ((merged["trend_direction_2020_2026"] == "rising") & (merged["computed_slope"] > 0)) |
        ((merged["trend_direction_2020_2026"] == "declining") & (merged["computed_slope"] < 0))
    )
    agreement_rate = merged["slope_agrees_with_label"].mean()
    summary_lines.append(f"Computed trend slope agrees with genre_dictionary label for "
                          f"{merged['slope_agrees_with_label'].sum()}/{len(merged)} genres "
                          f"({agreement_rate:.0%})")
    summary_lines.append(merged.sort_values("computed_slope", ascending=False).to_string())


def country_risk_style_analysis(df: pd.DataFrame, figures_dir: Path, summary_lines: list) -> None:
    """
    Analog of the original template's 'country risk analysis' — here, an
    artist-country x genre-mix breakdown, since this dataset has no
    macroeconomic country data. Shows which countries' catalogs skew toward
    which genres and how that relates to average popularity.
    """
    summary_lines.append("\n=== ARTIST-COUNTRY MARKET ANALYSIS ===")
    country_stats = df.groupby("artist_country").agg(
        track_count=("track_id", "count"),
        avg_popularity=("popularity", "mean"),
        dominant_genre=("genre", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
    ).sort_values("avg_popularity", ascending=False)
    summary_lines.append(country_stats.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    country_stats["avg_popularity"].plot(kind="bar", ax=ax, color="#1DB954")
    ax.set_title("Average Popularity by Artist Country")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "popularity_by_country.png", dpi=120)
    plt.close()


def run_eda(config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)
    tracks, genre_year = load_data(config)
    figures_dir = resolve_path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = resolve_path(config["paths"]["raw_dir"])

    summary_lines = []
    dataset_overview(tracks, summary_lines)
    missing_and_duplicate_check(tracks, summary_lines)
    outlier_analysis(tracks, figures_dir, summary_lines)
    univariate_analysis(tracks, figures_dir, summary_lines)
    bivariate_analysis(tracks, figures_dir, summary_lines)
    correlation_analysis(tracks, figures_dir, summary_lines)
    multivariate_analysis(tracks, figures_dir, summary_lines)
    genre_trend_analysis(genre_year, raw_dir / config["raw_files"]["genres"], figures_dir, summary_lines)
    country_risk_style_analysis(tracks, figures_dir, summary_lines)

    summary_text = "\n".join(summary_lines)
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    (reports_dir / "eda_summary.txt").write_text(summary_text, encoding="utf-8")
    logger.info("EDA complete. Figures in %s, summary in %s", figures_dir, reports_dir / "eda_summary.txt")
    print(summary_text)


if __name__ == "__main__":
    run_eda()
