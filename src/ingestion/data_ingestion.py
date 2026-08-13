"""
Data ingestion for the Spotify Track Performance Intelligence pipeline.

Loads the four raw source files (tracks, audio features, artists, genres),
performs the join into a single wide table, and writes the merged result to
data/interim/. This module does NOT clean or validate data — it only loads
and joins. Validation happens in src/validation, cleaning in
src/preprocessing, so each concern stays testable in isolation.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.helper import load_config, read_csv_safe, resolve_path, write_csv_safe
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_tables(config: Dict) -> Dict[str, pd.DataFrame]:
    """Load the four raw CSVs declared in config into a dict of DataFrames."""
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    raw_files = config["raw_files"]

    tables = {
        "tracks": read_csv_safe(raw_dir / raw_files["tracks"]),
        "audio_features": read_csv_safe(raw_dir / raw_files["audio_features"]),
        "artists": read_csv_safe(raw_dir / raw_files["artists"]),
        "genres": read_csv_safe(raw_dir / raw_files["genres"]),
    }
    return tables


def merge_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join tracks + audio_features + artists + genres into one wide table.

    Join keys:
      tracks.track_id      <-> audio_features.track_id   (1:1)
      tracks.artist_id     <-> artists.artist_id          (many:1)
      tracks.genre         <-> genres.genre_code           (many:1)

    Left joins are used from `tracks` outward so that no track is silently
    dropped if a downstream table is missing a match — any such gaps get
    caught explicitly by the validation layer, not swallowed here.
    """
    tracks = tables["tracks"]
    audio = tables["audio_features"]
    artists = tables["artists"]
    genres = tables["genres"]

    # Audio feature columns are suffixed-safe because tracks/audio share no
    # overlapping column names other than track_id (the join key).
    merged = tracks.merge(audio, on="track_id", how="left", validate="one_to_one")

    merged = merged.merge(
        artists.add_prefix("artist_"),
        left_on="artist_id",
        right_on="artist_artist_id",
        how="left",
    ).drop(columns=["artist_artist_id"])

    merged = merged.merge(
        genres.add_prefix("genre_meta_"),
        left_on="genre",
        right_on="genre_meta_genre_code",
        how="left",
    ).drop(columns=["genre_meta_genre_code"])

    logger.info(
        "Merged dataset shape: %s rows x %s cols (from %s tracks)",
        merged.shape[0],
        merged.shape[1],
        tracks.shape[0],
    )
    return merged


def run_ingestion(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Full ingestion entrypoint: load raw tables, merge, persist to interim/."""
    config = load_config(config_path)
    tables = load_raw_tables(config)
    merged = merge_tables(tables)

    interim_dir = resolve_path(config["paths"]["interim_dir"])
    out_path = interim_dir / config["interim_files"]["merged"]
    write_csv_safe(merged, out_path)

    logger.info("Ingestion complete. Merged dataset written to %s", out_path)
    return merged


if __name__ == "__main__":
    run_ingestion()
