"""
Hyperparameter tuning for the regression and classification models.

Kept separate from train_model.py (which trains with sensible fixed defaults
for the leaderboard comparison) so the leaderboard stays fast and tuning is
an explicit, opt-in step run on whichever model the leaderboard identified
as best — tuning all 7 models with a full grid on every run would be wasted
compute for models that aren't going to be selected anyway.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from src.training.feature_target_spec import get_feature_target_columns, prepare_classification_target
from src.training.train_model import build_preprocessor
from src.utils.helper import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)
RANDOM_SEED = 42

REGRESSION_PARAM_GRIDS = {
    "random_forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
    },
    "lightgbm": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [3, 4, 5, -1],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__num_leaves": [15, 31, 63],
    },
}

CLASSIFICATION_PARAM_GRIDS = {
    "random_forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.03, 0.05, 0.1],
    },
    "lightgbm": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [3, 4, 5, -1],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__num_leaves": [15, 31, 63],
    },
}

MODEL_BUILDERS = {
    "regression": {
        "random_forest": lambda: RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        "xgboost": lambda: XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        "lightgbm": lambda: LGBMRegressor(random_state=RANDOM_SEED, verbosity=-1),
    },
    "classification": {
        "random_forest": lambda: RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        "xgboost": lambda: XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1, eval_metric="mlogloss"),
        "lightgbm": lambda: LGBMClassifier(class_weight="balanced", random_state=RANDOM_SEED, verbosity=-1),
    },
}


def tune_model(
    df: pd.DataFrame,
    task: str,
    model_name: str,
    config: Dict,
    search_type: str = "random",
    n_iter: int = 20,
) -> Tuple[Pipeline, Dict, float]:
    """
    Tune one model with GridSearchCV or RandomizedSearchCV.

    search_type="random" is the default and recommended choice here: with
    1,696 rows and grids of 27-64 combinations x 5-fold CV, a full grid
    search is affordable but a randomized search over the same space gets
    ~90% of the benefit in a fraction of the fits, which matters more once
    this scales to a larger catalog. Use search_type="grid" for a smaller,
    already-narrowed grid where exhaustiveness is worth the cost.

    Returns (best_pipeline, best_params, best_cv_score).
    """
    if task == "regression":
        categorical, numeric, target = get_feature_target_columns(df, task="regression")
        X, y = df[categorical + numeric], df[target]
        cv = KFold(n_splits=config["modeling"]["cv_folds"], shuffle=True, random_state=RANDOM_SEED)
        scoring = "r2"
        param_grid = REGRESSION_PARAM_GRIDS[model_name]
    else:
        df = prepare_classification_target(df)
        categorical, numeric, target = get_feature_target_columns(df, task="classification")
        X, y_raw = df[categorical + numeric], df[target]
        y = pd.Series(LabelEncoder().fit_transform(y_raw), index=y_raw.index)
        cv = StratifiedKFold(n_splits=config["modeling"]["cv_folds"], shuffle=True, random_state=RANDOM_SEED)
        scoring = "f1_macro"
        param_grid = CLASSIFICATION_PARAM_GRIDS[model_name]

    pipeline = Pipeline([
        ("preprocess", build_preprocessor(categorical, numeric)),
        ("model", MODEL_BUILDERS[task][model_name]()),
    ])

    if search_type == "grid":
        search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1,
            n_iter=n_iter, random_state=RANDOM_SEED,
        )

    search.fit(X, y)
    logger.info(
        "[tuning/%s/%s] best %s=%.4f with params=%s",
        task, model_name, scoring, search.best_score_, search.best_params_,
    )
    return search.best_estimator_, search.best_params_, search.best_score_


def run_hyperparameter_tuning(config_path: str = "config/config.yaml") -> Dict:
    """Tune the models the leaderboard identified as strongest candidates —
    not the plain linear regression (nothing to tune) but its tree-based
    competitors, to see if tuning closes the gap."""
    config = load_config(config_path)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    df = pd.read_csv(processed_dir / config["processed_files"]["model_ready"])

    results = {}
    for model_name in ["random_forest", "xgboost", "lightgbm"]:
        pipeline, params, score = tune_model(df, "regression", model_name, config)
        results[f"regression_{model_name}"] = {"pipeline": pipeline, "params": params, "score": score}

    for model_name in ["random_forest", "xgboost", "lightgbm"]:
        pipeline, params, score = tune_model(df, "classification", model_name, config)
        results[f"classification_{model_name}"] = {"pipeline": pipeline, "params": params, "score": score}

    return results


if __name__ == "__main__":
    run_hyperparameter_tuning()
