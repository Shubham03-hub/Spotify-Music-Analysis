"""
End-to-end pipeline orchestrator for the Spotify Track Performance
Intelligence system.

Run with: python main.py [--skip-eda]

Sequence: preprocessing (ingest+validate+clean) -> feature engineering ->
EDA (optional) -> training with MLflow tracking + registry + joblib export ->
evaluation -> full-catalog prediction. Each stage is idempotent and can also
be run standalone via its own module's __main__ block, listed alongside each
call below.
"""

import argparse
import sys
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main(skip_eda: bool = False) -> None:
    start = time.time()
    logger.info("=" * 70)
    logger.info("SPOTIFY TRACK PERFORMANCE INTELLIGENCE — FULL PIPELINE RUN")
    logger.info("=" * 70)

    logger.info("[1/6] Preprocessing (ingestion + validation + cleaning)...")
    from src.preprocessing.preprocessing_pipeline import run_preprocessing
    run_preprocessing()

    logger.info("[2/6] Feature engineering...")
    from src.features.feature_engineering import run_feature_engineering
    run_feature_engineering()

    if not skip_eda:
        logger.info("[3/6] EDA...")
        from notebooks.eda_report import run_eda
        run_eda()
    else:
        logger.info("[3/6] EDA skipped (--skip-eda).")

    logger.info("[4/6] Training + MLflow tracking + model registry...")
    from src.training.mlflow_tracking import run_training_with_tracking
    run_training_with_tracking()

    logger.info("[5/6] Evaluation...")
    from src.evaluation.evaluate_model import run_evaluation
    run_evaluation()

    logger.info("[6/6] Scoring full catalog...")
    from src.prediction.predict import score_full_catalog
    from src.utils.helper import load_config, resolve_path
    scored = score_full_catalog()
    config = load_config()
    out_path = resolve_path(config["paths"]["processed_dir"]) / "scored_catalog.csv"
    scored.to_csv(out_path, index=False)
    logger.info("Scored catalog written to %s", out_path)

    elapsed = time.time() - start
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE in %.1fs. Launch the dashboard with:", elapsed)
    logger.info("  streamlit run dashboard/app.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full Spotify Track Performance Intelligence pipeline.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA figure generation (faster reruns).")
    args = parser.parse_args()
    main(skip_eda=args.skip_eda)
