"""
Business-rule data validation for the Spotify Track Performance Intelligence
pipeline.

Where schema_validation.py checks structure (columns/dtypes exist), this
module checks content: duplicate keys, referential integrity across tables,
value ranges, and null-rate thresholds. Run this after schema validation and
before preprocessing/cleaning.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Audio features are Spotify-style [0, 1] normalized scores except for the
# ones listed here, which have their own natural ranges.
AUDIO_FEATURE_RANGES = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "speechiness": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "loudness": (-60.0, 5.0),
    "tempo": (0.0, 250.0),
}

MAX_ACCEPTABLE_NULL_RATE = 0.05  # 5% — above this, flag for review rather than silently imputing


@dataclass
class ValidationIssue:
    check: str
    severity: str  # "error" or "warning"
    detail: str


@dataclass
class DataValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, check: str, severity: str, detail: str) -> None:
        self.issues.append(ValidationIssue(check, severity, detail))

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        lines = [f"Validation report: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for issue in self.issues:
            lines.append(f"  [{issue.severity.upper()}] {issue.check}: {issue.detail}")
        return "\n".join(lines)


def check_duplicate_keys(df: pd.DataFrame, key: str, table_name: str, report: DataValidationReport) -> None:
    dup_count = df[key].duplicated().sum()
    if dup_count > 0:
        report.add(
            check="duplicate_keys",
            severity="error",
            detail=f"{table_name}.{key} has {dup_count} duplicate value(s); "
                    f"must be deduplicated before feature engineering.",
        )


def check_referential_integrity(
    child_df: pd.DataFrame, child_key: str, parent_df: pd.DataFrame, parent_key: str,
    relationship_name: str, report: DataValidationReport,
) -> None:
    orphans = ~child_df[child_key].isin(parent_df[parent_key])
    orphan_count = orphans.sum()
    if orphan_count > 0:
        report.add(
            check="referential_integrity",
            severity="error",
            detail=f"{relationship_name}: {orphan_count} row(s) reference a "
                    f"{parent_key} not present in the parent table.",
        )


def check_null_rates(df: pd.DataFrame, table_name: str, report: DataValidationReport) -> None:
    null_rates = df.isna().mean()
    for col, rate in null_rates.items():
        if rate == 0:
            continue
        severity = "error" if rate > MAX_ACCEPTABLE_NULL_RATE else "warning"
        report.add(
            check="null_rate",
            severity=severity,
            detail=f"{table_name}.{col} is {rate:.1%} null.",
        )


def check_value_ranges(df: pd.DataFrame, table_name: str, report: DataValidationReport) -> None:
    for col, (low, high) in AUDIO_FEATURE_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col].notna()) & ((df[col] < low) | (df[col] > high))]
        if len(out_of_range) > 0:
            report.add(
                check="value_range",
                severity="error",
                detail=f"{table_name}.{col} has {len(out_of_range)} value(s) outside "
                        f"expected range [{low}, {high}].",
            )

    if "popularity" in df.columns:
        out_of_range = df[(df["popularity"] < 0) | (df["popularity"] > 100)]
        if len(out_of_range) > 0:
            report.add(
                check="value_range",
                severity="error",
                detail=f"{table_name}.popularity has {len(out_of_range)} value(s) "
                        f"outside expected [0, 100].",
            )


def validate_business_rules(tables: Dict[str, pd.DataFrame]) -> DataValidationReport:
    """
    Run the full business-rule validation suite across the raw tables.

    Returns a DataValidationReport rather than raising on the first issue, so
    every problem in the batch is surfaced at once instead of a single
    failure blocking discovery of the rest.
    """
    report = DataValidationReport()
    tracks, audio, artists, genres = (
        tables["tracks"], tables["audio_features"], tables["artists"], tables["genres"],
    )

    check_duplicate_keys(tracks, "track_id", "tracks", report)
    check_duplicate_keys(audio, "track_id", "audio_features", report)
    check_duplicate_keys(artists, "artist_id", "artists", report)
    check_duplicate_keys(genres, "genre_code", "genres", report)

    check_referential_integrity(tracks, "artist_id", artists, "artist_id", "tracks->artists", report)
    check_referential_integrity(tracks, "genre", genres, "genre_code", "tracks->genres", report)
    check_referential_integrity(audio, "track_id", tracks, "track_id", "audio_features->tracks", report)

    for name, df in tables.items():
        check_null_rates(df, name, report)

    check_value_ranges(tracks, "tracks", report)
    check_value_ranges(audio, "audio_features", report)

    logger.info(report.summary().replace("\n", " | "))
    return report


def raise_on_errors(report: DataValidationReport) -> None:
    """Hard-stop the pipeline if any error-severity issues were found."""
    if report.errors:
        raise ValueError(
            f"Data validation found {len(report.errors)} blocking error(s). "
            f"See validation report for details:\n{report.summary()}"
        )
