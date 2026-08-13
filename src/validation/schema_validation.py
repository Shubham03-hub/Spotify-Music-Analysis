"""
Schema validation for raw source tables.

Checks that each raw table has the required columns and correct-enough dtypes
before anything downstream touches it. This is deliberately separate from
data_validation.py (business-rule validation, e.g. referential integrity,
value ranges) — schema issues and business-rule issues fail for different
reasons and should be diagnosed independently.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaValidationResult:
    table_name: str
    passed: bool
    missing_columns: List[str] = field(default_factory=list)
    unexpected_dtype_issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return f"[PASS] {self.table_name}: schema OK"
        lines = [f"[FAIL] {self.table_name}: schema validation failed"]
        if self.missing_columns:
            lines.append(f"  missing columns: {self.missing_columns}")
        if self.unexpected_dtype_issues:
            lines.append(f"  dtype issues: {self.unexpected_dtype_issues}")
        return "\n".join(lines)


NUMERIC_COLUMNS_BY_TABLE = {
    "tracks": ["release_year", "duration_ms", "popularity"],
    "audio_features": [
        "danceability", "energy", "key", "loudness", "mode", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo",
        "time_signature",
    ],
    "artists": ["followers_millions", "artist_star_power", "debut_year"],
    "genres": [],
}


def validate_schema(table_name: str, df: pd.DataFrame, config: Dict) -> SchemaValidationResult:
    """Validate a single table's columns and dtypes against config['schema']."""
    schema_cfg = config["schema"][table_name]
    required_columns = schema_cfg["required_columns"]

    missing = [col for col in required_columns if col not in df.columns]

    dtype_issues = []
    for col in NUMERIC_COLUMNS_BY_TABLE.get(table_name, []):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            dtype_issues.append(f"{col} expected numeric, got {df[col].dtype}")

    passed = len(missing) == 0 and len(dtype_issues) == 0

    result = SchemaValidationResult(
        table_name=table_name,
        passed=passed,
        missing_columns=missing,
        unexpected_dtype_issues=dtype_issues,
    )
    logger.info(result.summary().replace("\n", " | "))
    return result


def validate_all_schemas(tables: Dict[str, pd.DataFrame], config: Dict) -> List[SchemaValidationResult]:
    """Run validate_schema across all four raw tables, returning all results."""
    results = [validate_schema(name, df, config) for name, df in tables.items()]

    failed = [r for r in results if not r.passed]
    if failed:
        for r in failed:
            logger.error(r.summary())
        raise ValueError(
            f"Schema validation failed for {len(failed)} table(s): "
            f"{[r.table_name for r in failed]}. See logs for details."
        )

    logger.info("All %s tables passed schema validation.", len(results))
    return results
