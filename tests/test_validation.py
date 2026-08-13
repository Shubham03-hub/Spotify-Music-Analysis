"""Tests for src.validation.schema_validation and src.validation.data_validation."""

import pandas as pd
import pytest

from src.validation.data_validation import (
    DataValidationReport,
    check_duplicate_keys,
    check_null_rates,
    check_referential_integrity,
    check_value_ranges,
    validate_business_rules,
)
from src.validation.schema_validation import validate_schema

MINIMAL_CONFIG = {
    "schema": {
        "tracks": {"required_columns": ["track_id", "artist_id", "popularity"]},
    }
}


def test_schema_validation_passes_with_all_columns(sample_tracks_df):
    result = validate_schema("tracks", sample_tracks_df, MINIMAL_CONFIG)
    assert result.passed


def test_schema_validation_fails_with_missing_column():
    df = pd.DataFrame({"track_id": ["T1"], "artist_id": ["A1"]})  # missing popularity
    result = validate_schema("tracks", df, MINIMAL_CONFIG)
    assert not result.passed
    assert "popularity" in result.missing_columns


def test_duplicate_key_detection_finds_real_duplicates():
    df = pd.DataFrame({"track_id": ["T1", "T1", "T2"]})
    report = DataValidationReport()
    check_duplicate_keys(df, "track_id", "tracks", report)
    assert len(report.errors) == 1
    assert "1 duplicate" in report.errors[0].detail


def test_duplicate_key_detection_clean_data_has_no_errors():
    df = pd.DataFrame({"track_id": ["T1", "T2", "T3"]})
    report = DataValidationReport()
    check_duplicate_keys(df, "track_id", "tracks", report)
    assert len(report.errors) == 0


def test_referential_integrity_catches_orphan_foreign_key():
    child = pd.DataFrame({"artist_id": ["A1", "A2", "A99"]})  # A99 doesn't exist in parent
    parent = pd.DataFrame({"artist_id": ["A1", "A2"]})
    report = DataValidationReport()
    check_referential_integrity(child, "artist_id", parent, "artist_id", "tracks->artists", report)
    assert len(report.errors) == 1


def test_referential_integrity_passes_with_valid_keys():
    child = pd.DataFrame({"artist_id": ["A1", "A2"]})
    parent = pd.DataFrame({"artist_id": ["A1", "A2", "A3"]})
    report = DataValidationReport()
    check_referential_integrity(child, "artist_id", parent, "artist_id", "tracks->artists", report)
    assert len(report.errors) == 0


def test_null_rate_check_flags_high_null_rate_as_error():
    # 10 rows, 6 null (60%) — well above the 5% threshold
    df = pd.DataFrame({"col": [None] * 6 + [1.0] * 4})
    report = DataValidationReport()
    check_null_rates(df, "test_table", report)
    assert any(i.severity == "error" for i in report.issues)


def test_null_rate_check_flags_low_null_rate_as_warning_only():
    # 100 rows, 1 null (1%) — below the 5% threshold
    df = pd.DataFrame({"col": [None] + [1.0] * 99})
    report = DataValidationReport()
    check_null_rates(df, "test_table", report)
    assert len(report.errors) == 0
    assert len(report.warnings) == 1


def test_value_range_check_catches_out_of_range_popularity():
    df = pd.DataFrame({"popularity": [50.0, 150.0, -10.0]})  # two invalid
    report = DataValidationReport()
    check_value_ranges(df, "tracks", report)
    assert len(report.errors) == 1
    assert "2 value" in report.errors[0].detail


def test_value_range_check_catches_out_of_range_danceability():
    df = pd.DataFrame({"danceability": [0.5, 1.5]})  # 1.5 is invalid, max is 1.0
    report = DataValidationReport()
    check_value_ranges(df, "audio_features", report)
    assert len(report.errors) == 1


def test_validate_business_rules_end_to_end(sample_tables):
    report = validate_business_rules(sample_tables)
    # sample_tables deliberately includes a null album_name and a null
    # duration_ms (mirroring real data's quirks) at a 25% rate in this small
    # 4-row fixture — well above the 5% threshold, so these SHOULD be
    # flagged. No duplicate keys or orphan foreign keys are present, though.
    assert len(report.errors) == 2
    assert all(issue.check == "null_rate" for issue in report.errors)
    duplicate_or_referential_errors = [
        i for i in report.errors if i.check in ("duplicate_keys", "referential_integrity")
    ]
    assert len(duplicate_or_referential_errors) == 0
