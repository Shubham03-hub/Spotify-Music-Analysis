"""
Preprocessing pipeline: orchestrates raw-table validation, ingestion/merge,
and cleaning into a single callable entrypoint used by main.py.

Validation runs twice by design:
  1. On the RAW tables, before merging — catches source-data problems
     (missing columns, bad dtypes, orphan foreign keys) at the earliest
     possible point, before they get baked into a wide merged table.
  2. Duplicate-key and null-rate checks are re-verified on the CLEANED
     dataset at the end, as a safety net confirming cleaning actually
     resolved what it claimed to.
"""

from typing import Dict

import pandas as pd

from src.ingestion.data_ingestion import load_raw_tables, merge_tables
from src.preprocessing.data_cleaning import (
    clean_dataset,
    drop_duplicate_audio_features,
    drop_duplicate_tracks,
)
from src.utils.helper import load_config, resolve_path, write_csv_safe
from src.utils.logger import get_logger
from src.validation.data_validation import (
    DataValidationReport,
    check_duplicate_keys,
    check_null_rates,
    validate_business_rules,
)
from src.validation.schema_validation import validate_all_schemas

logger = get_logger(__name__)


def run_preprocessing(config_path: str = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)

    # Step 1: load raw tables
    tables = load_raw_tables(config)

    # Step 2: validate structure, then business rules, on raw tables
    validate_all_schemas(tables, config)
    raw_report = validate_business_rules(tables)
    if raw_report.errors:
        logger.warning(
            "%s error-level issue(s) found on raw data (expected: duplicate "
            "track_ids and scattered nulls — both are handled explicitly in "
            "cleaning). Proceeding to merge + clean.",
            len(raw_report.errors),
        )

    # Step 3: dedupe the two tables known to carry duplicate track_ids BEFORE
    # merging. Merging first would multiply duplicates across the join
    # rather than cleanly resolve them (validated by the one_to_one merge
    # check in merge_tables, which fails loudly if this step is skipped).
    tables["tracks"] = drop_duplicate_tracks(tables["tracks"])
    tables["audio_features"] = drop_duplicate_audio_features(tables["audio_features"])

    # Step 4: merge
    merged = merge_tables(tables)

    # Step 5: clean (impute + dtype enforcement on the merged result)
    cleaned = clean_dataset(merged)

    # Step 5: safety-net re-validation on the cleaned result
    post_report = DataValidationReport()
    check_duplicate_keys(cleaned, "track_id", "cleaned_dataset", post_report)
    check_null_rates(cleaned, "cleaned_dataset", post_report)
    if post_report.errors:
        raise ValueError(
            f"Cleaned dataset still has blocking issues — cleaning logic did "
            f"not fully resolve raw-data problems:\n{post_report.summary()}"
        )
    logger.info("Post-cleaning validation passed: no duplicate keys, no null-rate errors.")

    # Step 6: persist
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    out_path = processed_dir / "cleaned_dataset.csv"
    write_csv_safe(cleaned, out_path)

    return cleaned


if __name__ == "__main__":
    run_preprocessing()
