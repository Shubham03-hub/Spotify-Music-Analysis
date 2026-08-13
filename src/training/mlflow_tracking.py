"""
MLflow experiment tracking and model registry for the Spotify Track
Performance Intelligence system (PHASE 9).

Wraps the training already implemented in train_model.py — this module does
NOT reimplement model fitting, it adds tracking around it: every model in
the leaderboard becomes one MLflow run (params + CV metrics + the fitted
pipeline as an artifact), and the two models train_model.select_best_model()
identifies as best get registered to the Model Registry under stable names
so the prediction pipeline can load "the current production model" without
knowing which algorithm won.

To view results: run `mlflow ui --backend-store-uri mlruns` from the project
root and open http://localhost:5000
"""

from typing import Dict

import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump

from src.training.train_model import (
    build_leaderboard, select_best_model, train_classification_models, train_regression_models,
)
from src.utils.helper import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

REGRESSION_MODEL_REGISTRY_NAME = "spotify-popularity-regressor"
CLASSIFICATION_MODEL_REGISTRY_NAME = "spotify-tier-classifier"


def _log_regression_run(model_name: str, result, config: Dict) -> None:
    with mlflow.start_run(run_name=f"regression_{model_name}"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("task", "regression")
        mlflow.log_param("cv_folds", config["modeling"]["cv_folds"])

        summary = result.summary()
        for metric_name, value in summary.items():
            mlflow.log_metric(metric_name, value)

        mlflow.sklearn.log_model(
            result.fitted_pipeline, artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        logger.info("Logged MLflow run for regression/%s", model_name)


def _log_classification_run(model_name: str, result, config: Dict) -> None:
    with mlflow.start_run(run_name=f"classification_{model_name}"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("task", "classification")
        mlflow.log_param("cv_folds", config["modeling"]["cv_folds"])
        mlflow.log_param("note", "Viral Hit merged into Hit for CV stratification validity")

        summary = result.summary()
        for metric_name, value in summary.items():
            mlflow.log_metric(metric_name, value)

        mlflow.sklearn.log_model(
            result.fitted_pipeline, artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        logger.info("Logged MLflow run for classification/%s", model_name)


def run_training_with_tracking(config_path: str = "config/config.yaml") -> Dict:
    config = load_config(config_path)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    df = pd.read_csv(processed_dir / config["processed_files"]["model_ready"])

    logger.info("Training + logging regression models to MLflow...")
    reg_results = train_regression_models(df, config)
    for model_name, result in reg_results.items():
        _log_regression_run(model_name, result, config)

    logger.info("Training + logging classification models to MLflow...")
    clf_results, clf_label_encoder = train_classification_models(df, config)
    for model_name, result in clf_results.items():
        _log_classification_run(model_name, result, config)

    best_reg_name = select_best_model(reg_results, "regression")
    best_clf_name = select_best_model(clf_results, "classification")

    # Register the best of each task as a new version under a stable name.
    with mlflow.start_run(run_name=f"register_{best_reg_name}"):
        mlflow.sklearn.log_model(
            reg_results[best_reg_name].fitted_pipeline,
            artifact_path="model",
            registered_model_name=REGRESSION_MODEL_REGISTRY_NAME,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    with mlflow.start_run(run_name=f"register_{best_clf_name}"):
        mlflow.sklearn.log_model(
            clf_results[best_clf_name].fitted_pipeline,
            artifact_path="model",
            registered_model_name=CLASSIFICATION_MODEL_REGISTRY_NAME,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    logger.info(
        "Registered '%s' as %s and '%s' as %s.",
        best_reg_name, REGRESSION_MODEL_REGISTRY_NAME, best_clf_name, CLASSIFICATION_MODEL_REGISTRY_NAME,
    )

    # Also persist lightweight joblib copies. MLflow's registry is the
    # governance/versioning system of record; the dashboard and prediction
    # pipeline load from these joblib files directly so they don't need a
    # live MLflow tracking connection to serve predictions.
    models_dir = resolve_path(config["paths"]["models_dir"])
    dump(reg_results[best_reg_name].fitted_pipeline, models_dir / "best_regression_pipeline.joblib")
    dump(clf_results[best_clf_name].fitted_pipeline, models_dir / "best_classification_pipeline.joblib")
    dump(clf_label_encoder, models_dir / "classification_label_encoder.joblib")
    dump({"regression_model": best_reg_name, "classification_model": best_clf_name}, models_dir / "model_metadata.joblib")
    logger.info("Saved joblib artifacts to %s for dashboard/prediction use.", models_dir)

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    build_leaderboard(reg_results).to_csv(reports_dir / "regression_leaderboard.csv", index=False)
    build_leaderboard(clf_results).to_csv(reports_dir / "classification_leaderboard.csv", index=False)

    return {
        "best_regression_model": best_reg_name,
        "best_classification_model": best_clf_name,
        "classification_label_encoder": clf_label_encoder,
    }


if __name__ == "__main__":
    run_training_with_tracking()
