"""
Model training for the Spotify Track Performance Intelligence system.

Trains and compares, per PHASE 7 of the spec:
  Regression (predicting `popularity`):     Linear Regression, Random Forest, XGBoost, LightGBM
  Classification (predicting success tier): Random Forest, XGBoost, LightGBM

Linear Regression is intentionally NOT used for classification — there is no
direct linear-classifier analog in the requested model list, and swapping in
Logistic Regression silently would misrepresent what was actually run. If a
linear classification baseline is wanted, add it explicitly.

Given the dataset's realistic constraints — 1,696 rows, severe class
imbalance on tier (see feature_target_spec.prepare_classification_target) —
every model here uses 5-fold cross-validation and, for classification,
class-balanced weighting. Results are reported as CV mean +/- std, not a
single lucky train/test split.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.training.feature_target_spec import (
    get_feature_target_columns,
    prepare_classification_target,
)
from src.utils.helper import load_config, resolve_path, write_csv_safe
from src.utils.logger import get_logger

logger = get_logger(__name__)

RANDOM_SEED = 42


def build_preprocessor(categorical: List[str], numeric: List[str]) -> ColumnTransformer:
    """Shared preprocessing: one-hot encode categoricals, scale numerics.
    Used identically across all model families for a fair comparison."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )


@dataclass
class ModelResult:
    model_name: str
    task: str
    cv_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    fitted_pipeline: Pipeline = None

    def summary(self) -> Dict[str, float]:
        out = {}
        for metric, scores in self.cv_scores.items():
            out[f"{metric}_mean"] = float(np.mean(scores))
            out[f"{metric}_std"] = float(np.std(scores))
        return out


REGRESSION_MODELS = {
    "linear_regression": lambda: LinearRegression(),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1),
    "xgboost": lambda: XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1,
    ),
    "lightgbm": lambda: LGBMRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=RANDOM_SEED, verbosity=-1,
    ),
}

CLASSIFICATION_MODELS = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1, eval_metric="mlogloss",
    ),
    "lightgbm": lambda: LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, class_weight="balanced", random_state=RANDOM_SEED, verbosity=-1,
    ),
}


def train_regression_models(df: pd.DataFrame, config: Dict) -> Dict[str, ModelResult]:
    categorical, numeric, target = get_feature_target_columns(df, task="regression")
    X, y = df[categorical + numeric], df[target]

    cv = KFold(n_splits=config["modeling"]["cv_folds"], shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    results = {}
    for name in config["modeling"]["regression_models"]:
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(categorical, numeric)),
            ("model", REGRESSION_MODELS[name]()),
        ])
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        cv_scores = {
            "rmse": -cv_results["test_rmse"],
            "mae": -cv_results["test_mae"],
            "r2": cv_results["test_r2"],
        }
        pipeline.fit(X, y)  # final fit on full data for the registered artifact
        result = ModelResult(model_name=name, task="regression", cv_scores=cv_scores, fitted_pipeline=pipeline)
        results[name] = result
        logger.info(
            "[regression/%s] RMSE=%.2f+/-%.2f  MAE=%.2f+/-%.2f  R2=%.3f+/-%.3f",
            name, *[v for pair in [
                (np.mean(cv_scores["rmse"]), np.std(cv_scores["rmse"])),
                (np.mean(cv_scores["mae"]), np.std(cv_scores["mae"])),
                (np.mean(cv_scores["r2"]), np.std(cv_scores["r2"])),
            ] for v in pair],
        )
    return results


def train_classification_models(df: pd.DataFrame, config: Dict) -> Dict[str, ModelResult]:
    df = prepare_classification_target(df)
    categorical, numeric, target = get_feature_target_columns(df, task="classification")
    X, y_raw = df[categorical + numeric], df[target]

    # XGBoost requires 0-indexed integer class labels; RandomForest/LightGBM
    # accept strings directly, but label-encoding once and using it for all
    # three keeps every model in the leaderboard trained on an identical
    # target representation, which matters for a fair comparison.
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_raw), index=y_raw.index, name=target)
    logger.info("Classification classes (encoded 0..%d): %s", len(label_encoder.classes_) - 1, list(label_encoder.classes_))

    cv = StratifiedKFold(n_splits=config["modeling"]["cv_folds"], shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
    }

    results = {}
    for name in config["modeling"]["classification_models"]:
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(categorical, numeric)),
            ("model", CLASSIFICATION_MODELS[name]()),
        ])
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        cv_scores = {
            "f1_macro": cv_results["test_f1_macro"],
            "accuracy": cv_results["test_accuracy"],
            "f1_weighted": cv_results["test_f1_weighted"],
        }
        pipeline.fit(X, y)
        result = ModelResult(model_name=name, task="classification", cv_scores=cv_scores, fitted_pipeline=pipeline)
        results[name] = result
        logger.info(
            "[classification/%s] macro-F1=%.3f+/-%.3f  accuracy=%.3f+/-%.3f  weighted-F1=%.3f+/-%.3f",
            name, *[v for pair in [
                (np.mean(cv_scores["f1_macro"]), np.std(cv_scores["f1_macro"])),
                (np.mean(cv_scores["accuracy"]), np.std(cv_scores["accuracy"])),
                (np.mean(cv_scores["f1_weighted"]), np.std(cv_scores["f1_weighted"])),
            ] for v in pair],
        )
    return results, label_encoder


def select_best_model(results: Dict[str, ModelResult], task: str) -> str:
    """Pick the best model by CV mean of the primary metric (R2 for
    regression, macro-F1 for classification — chosen for imbalance)."""
    metric = "r2" if task == "regression" else "f1_macro"
    best_name = max(results, key=lambda name: np.mean(results[name].cv_scores[metric]))
    logger.info("Best %s model by %s: %s", task, metric, best_name)
    return best_name


def build_leaderboard(results: Dict[str, ModelResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        row = {"model": name}
        row.update(result.summary())
        rows.append(row)
    return pd.DataFrame(rows)


def run_training(config_path: str = "config/config.yaml") -> Dict:
    config = load_config(config_path)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    df = pd.read_csv(processed_dir / config["processed_files"]["model_ready"])

    logger.info("Training regression models on popularity (%s rows)...", len(df))
    reg_results = train_regression_models(df, config)

    logger.info("Training classification models on popularity_tier (%s rows)...", len(df))
    clf_results, clf_label_encoder = train_classification_models(df, config)

    reg_leaderboard = build_leaderboard(reg_results)
    clf_leaderboard = build_leaderboard(clf_results)

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    write_csv_safe(reg_leaderboard, reports_dir / "regression_leaderboard.csv")
    write_csv_safe(clf_leaderboard, reports_dir / "classification_leaderboard.csv")

    best_reg = select_best_model(reg_results, "regression")
    best_clf = select_best_model(clf_results, "classification")

    return {
        "regression_results": reg_results,
        "classification_results": clf_results,
        "classification_label_encoder": clf_label_encoder,
        "best_regression_model": best_reg,
        "best_classification_model": best_clf,
        "regression_leaderboard": reg_leaderboard,
        "classification_leaderboard": clf_leaderboard,
    }


if __name__ == "__main__":
    run_training()
