"""
Unit tests for data_loader.py
Validates dataset ingestion, schema validation, data types, and null handling.
"""
import pytest
import pandas as pd
from app.config import settings
from app.data_loader import DataLoader, DataValidationError


def test_data_loader_ingestion():
    loader = DataLoader()
    df = loader.get_data()

    assert not df.empty
    assert len(df) == 500
    assert loader.record_count == 500
    assert loader.is_loaded is True


def test_required_columns_present():
    loader = DataLoader()
    df = loader.get_data()

    for col in settings.VALID_COLUMNS:
        assert col in df.columns, f"Required column '{col}' is missing"


def test_data_types_and_parsing():
    loader = DataLoader()
    df = loader.get_data()

    assert "created_at_dt" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["created_at_dt"])
    assert pd.api.types.is_float_dtype(df["response_time_hrs"])
    assert pd.api.types.is_float_dtype(df["resolution_time_hrs"])
    assert pd.api.types.is_float_dtype(df["customer_rating"])


def test_null_value_handling():
    loader = DataLoader()
    df = loader.get_data()

    # Unresolved tickets must have null resolution times and null customer ratings
    unresolved = df[df["status"].isin(["Open", "Escalated"])]
    assert len(unresolved) == 173
    assert unresolved["resolution_time_hrs"].isna().all()
    assert unresolved["customer_rating"].isna().all()

    # Resolved tickets must have valid resolution times
    resolved = df[df["status"] == "Resolved"]
    assert len(resolved) == 327
    assert resolved["resolution_time_hrs"].notna().all()


def test_categorical_values_valid():
    loader = DataLoader()
    df = loader.get_data()

    assert set(df["category"].unique()).issubset(set(settings.VALID_CATEGORIES))
    assert set(df["priority"].unique()).issubset(set(settings.VALID_PRIORITIES))
    assert set(df["status"].unique()).issubset(set(settings.VALID_STATUSES))


def test_reference_timestamp():
    loader = DataLoader()
    ref_dt = loader.get_reference_timestamp()
    df = loader.get_data()

    if settings.REFERENCE_TIME_MODE == "dataset_max":
        assert ref_dt == df["created_at_dt"].max()
        assert ref_dt.year == 2024
