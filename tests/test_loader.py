"""Tests for data loading functions."""

from io import StringIO

import numpy as np
import pandas as pd

from src.data.loader import (
    load_nasa_power_temperature,
    load_pgn_interval_data,
    load_multiple_pgn_files,
)


def test_load_pgn_interval_data(sample_pgn_csv):
    df = load_pgn_interval_data(sample_pgn_csv)

    assert len(df) == 8
    assert list(df.columns) == ["date", "start_time", "end_time", "usage_kwh", "cost"]
    assert df["usage_kwh"].dtype == float
    assert df["cost"].dtype == float
    # Check $ was stripped
    assert df["cost"].iloc[0] == 0.08


def test_load_pgn_dates_parsed(sample_pgn_csv):
    df = load_pgn_interval_data(sample_pgn_csv)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].iloc[0] == pd.Timestamp("2024-01-12")


def test_load_nasa_power_temperature(sample_nasa_csv):
    df = load_nasa_power_temperature(sample_nasa_csv)

    assert len(df) == 4
    assert list(df.columns) == ["date", "temp_c", "temp_f"]
    # Sentinel replaced with NaN
    assert np.isnan(df["temp_c"].iloc[2])
    assert np.isnan(df["temp_f"].iloc[2])
    # Normal value preserved
    assert df["temp_c"].iloc[0] == -1.98


def test_load_nasa_temp_conversion(sample_nasa_csv):
    df = load_nasa_power_temperature(sample_nasa_csv)
    # -1.98 C = 28.436 F
    assert abs(df["temp_f"].iloc[0] - 28.436) < 0.01


def test_load_multiple_pgn_deduplicates():
    csv1 = StringIO(
        "TYPE,DATE,START TIME,END TIME,USAGE (kWh),COST,NOTES\n"
        "Electric usage,2024-01-12,00:00,00:14,0.52,$0.08,\n"
        "Electric usage,2024-01-12,00:15,00:29,0.52,$0.08,\n"
    )
    csv2 = StringIO(
        "TYPE,DATE,START TIME,END TIME,USAGE (kWh),COST,NOTES\n"
        "Electric usage,2024-01-12,00:00,00:14,0.60,$0.09,\n"  # duplicate, newer value
        "Electric usage,2024-01-12,00:30,00:44,0.55,$0.08,\n"
    )
    df = load_multiple_pgn_files([csv1, csv2])

    # Should have 3 unique (date, start_time) combinations
    assert len(df) == 3
    # Duplicate should keep last (from csv2)
    row = df[(df["date"] == pd.Timestamp("2024-01-12")) & (df["start_time"] == "00:00")]
    assert row["usage_kwh"].iloc[0] == 0.60
