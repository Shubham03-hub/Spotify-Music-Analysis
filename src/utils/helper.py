"""
Shared helper functions used across the pipeline: config loading, path
resolution, and small IO utilities. Keeping these in one place avoids every
module reinventing its own yaml-loading or path-joining logic.
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load the project's central YAML config into a dict."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found at {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from %s", full_path)
    return config


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root, regardless of cwd."""
    return PROJECT_ROOT / relative_path


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV with a clear error message if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file at {path} but it does not exist. "
            f"Confirm raw data has been placed in data/raw/."
        )
    df = pd.read_csv(path, **kwargs)
    logger.info("Read %s rows x %s cols from %s", df.shape[0], df.shape[1], path)
    return df


def write_csv_safe(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write a CSV, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    logger.info("Wrote %s rows x %s cols to %s", df.shape[0], df.shape[1], path)


def ensure_dirs(config: Dict[str, Any]) -> None:
    """Create all standard project directories declared in config['paths']."""
    for key, rel_path in config["paths"].items():
        resolve_path(rel_path).mkdir(parents=True, exist_ok=True)
    logger.info("Verified/created all project directories from config.")
