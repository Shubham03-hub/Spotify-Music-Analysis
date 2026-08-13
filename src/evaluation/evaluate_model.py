"""
Model evaluation for the Spotify Track Performance Intelligence system.

Produces PHASE 8 deliverables: regression metrics (MAE/MSE/RMSE/R2/MAPE),
residual plot, prediction-vs-actual plot, error distribution; classification
metrics (accuracy, precision/recall/F1 per class, confusion matrix); and
driver importance (feature importance) for both tasks. All figures are
computed from actual held-out predictions, not simulated.
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix,
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split

from src.training.feature_target_spec import get_feature_target_columns, prepare_classification_target
from src.utils.helper import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")
RANDOM_SEED = 42


def evaluate_regression(pipeline, df: pd.DataFrame, figures_dir: Path, model_name: str) -> Dict:
    categorical, numeric, target = get_feature_target_columns(df, task="regression")
    X, y = df[categorical + numeric], df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test.clip(lower=1), np.clip(y_pred, 1, None)),
    }
    logger.info(
        "[%s] Held-out test set: MAE=%.2f MSE=%.2f RMSE=%.2f R2=%.3f MAPE=%.1f%%",
        model_name, metrics["mae"], metrics["mse"], metrics["rmse"], metrics["r2"], metrics["mape"] * 100,
    )

    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.5, color="#1DB954")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted popularity")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residual Plot")

    axes[1].scatter(y_test, y_pred, alpha=0.5, color="#1DB954")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--")
    axes[1].set_xlabel("Actual popularity")
    axes[1].set_ylabel("Predicted popularity")
    axes[1].set_title("Prediction vs Actual")

    sns.histplot(residuals, kde=True, ax=axes[2], color="#1DB954")
    axes[2].set_title("Error Distribution")
    axes[2].set_xlabel("Residual")

    plt.tight_layout()
    plt.savefig(figures_dir / f"regression_evaluation_{model_name}.png", dpi=120)
    plt.close()

    return metrics


def evaluate_classification(pipeline, label_encoder, df: pd.DataFrame, figures_dir: Path, model_name: str) -> Dict:
    df = prepare_classification_target(df)
    categorical, numeric, target = get_feature_target_columns(df, task="classification")
    X, y_raw = df[categorical + numeric], df[target]
    y = pd.Series(label_encoder.transform(y_raw), index=y_raw.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    class_names = label_encoder.classes_
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    logger.info(
        "[%s] Held-out classification report:\n%s",
        model_name, classification_report(y_test, y_pred, target_names=class_names, zero_division=0),
    )

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"confusion_matrix_{model_name}.png", dpi=120)
    plt.close()

    return report


def extract_feature_importance(pipeline, model_name: str, figures_dir: Path, top_n: int = 15) -> pd.DataFrame:
    """Extract and plot driver importance from a fitted pipeline's final
    estimator. Works for tree-based models (feature_importances_) and linear
    models (coef_) — the two families used in this project."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        logger.warning("Model %s has no feature_importances_ or coef_; skipping driver importance.", model_name)
        return pd.DataFrame()

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax, color="#1DB954")
    ax.set_title(f"Driver Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"feature_importance_{model_name}.png", dpi=120)
    plt.close()

    return importance_df


def run_evaluation(config_path: str = "config/config.yaml") -> None:
    from src.training.train_model import (
        CLASSIFICATION_MODELS, REGRESSION_MODELS, build_preprocessor, train_classification_models, train_regression_models,
    )
    from sklearn.pipeline import Pipeline

    config = load_config(config_path)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    df = pd.read_csv(processed_dir / config["processed_files"]["model_ready"])
    figures_dir = resolve_path(config["paths"]["figures_dir"])
    reports_dir = resolve_path(config["paths"]["reports_dir"])

    # Best models identified in train_model.run_training(): linear_regression, random_forest
    categorical, numeric, _ = get_feature_target_columns(df, task="regression")
    reg_pipeline = Pipeline([
        ("preprocess", build_preprocessor(categorical, numeric)),
        ("model", REGRESSION_MODELS["linear_regression"]()),
    ])
    reg_metrics = evaluate_regression(reg_pipeline, df, figures_dir, "linear_regression")

    from sklearn.preprocessing import LabelEncoder
    df_clf = prepare_classification_target(df)
    label_encoder = LabelEncoder().fit(df_clf["popularity_tier_for_modeling"])
    categorical_c, numeric_c, _ = get_feature_target_columns(df_clf, task="classification")
    clf_pipeline = Pipeline([
        ("preprocess", build_preprocessor(categorical_c, numeric_c)),
        ("model", CLASSIFICATION_MODELS["random_forest"]()),
    ])
    clf_report = evaluate_classification(clf_pipeline, label_encoder, df, figures_dir, "random_forest")

    reg_importance = extract_feature_importance(reg_pipeline, "linear_regression_popularity", figures_dir)
    clf_importance = extract_feature_importance(clf_pipeline, "random_forest_tier", figures_dir)

    reg_importance.to_csv(reports_dir / "driver_importance_regression.csv", index=False)
    clf_importance.to_csv(reports_dir / "driver_importance_classification.csv", index=False)

    logger.info("Evaluation complete. Regression test metrics: %s", reg_metrics)
    logger.info("Top regression drivers:\n%s", reg_importance.head(8).to_string())
    logger.info("Top classification drivers:\n%s", clf_importance.head(8).to_string())

    return {
        "regression_metrics": reg_metrics,
        "classification_report": clf_report,
        "regression_importance": reg_importance,
        "classification_importance": clf_importance,
    }


if __name__ == "__main__":
    run_evaluation()
