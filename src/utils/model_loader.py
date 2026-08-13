"""
Model loading utility. The dashboard and prediction pipeline both load
models through this single module so there is exactly one place that knows
where trained artifacts live on disk.
"""

from pathlib import Path
from typing import Any, Dict

from joblib import load

from src.utils.helper import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_regression_pipeline(config: Dict = None):
    config = config or load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])
    path = models_dir / "best_regression_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained regression model found at {path}. "
            f"Run `python -m src.training.mlflow_tracking` first."
        )
    return load(path)


def load_classification_pipeline(config: Dict = None):
    config = config or load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])
    path = models_dir / "best_classification_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained classification model found at {path}. "
            f"Run `python -m src.training.mlflow_tracking` first."
        )
    return load(path)


def load_label_encoder(config: Dict = None):
    config = config or load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])
    return load(models_dir / "classification_label_encoder.joblib")


def load_model_metadata(config: Dict = None) -> Dict[str, Any]:
    config = config or load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])
    return load(models_dir / "model_metadata.joblib")


def models_are_available(config: Dict = None) -> bool:
    config = config or load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])
    return (models_dir / "best_regression_pipeline.joblib").exists() and (
        models_dir / "best_classification_pipeline.joblib"
    ).exists()
