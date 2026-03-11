"""Tests for Open-Meteo temperature API client."""

from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd

from src.data.temperature_api import _parse_response, get_missing_temperature_dates


def test_parse_response():
    data = {
        "daily": {
            "time": ["2024-01-12", "2024-01-13"],
            "temperature_2m_mean": [-1.98, -11.64],
        }
    }
    df = _parse_response(data)

    assert len(df) == 2
    assert list(df.columns) == ["date", "temp_c", "temp_f"]
    assert df["temp_c"].iloc[0] == -1.98
    assert abs(df["temp_f"].iloc[0] - 28.436) < 0.01


def test_get_missing_temperature_dates():
    usage_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12", "2024-01-13"]),
    })
    temp_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-12", "2024-01-13"]),
        "temp_c": [5.0, 6.0],
    })

    ranges = get_missing_temperature_dates(temp_df, usage_df)

    assert len(ranges) == 1
    assert ranges[0] == (date(2024, 1, 10), date(2024, 1, 11))


def test_get_missing_temperature_dates_no_gaps():
    usage_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-12", "2024-01-13"]),
    })
    temp_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-12", "2024-01-13"]),
        "temp_c": [5.0, 6.0],
    })

    ranges = get_missing_temperature_dates(temp_df, usage_df)
    assert len(ranges) == 0


def test_get_missing_with_nan_temp():
    usage_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-12", "2024-01-13"]),
    })
    temp_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-12", "2024-01-13"]),
        "temp_c": [5.0, float("nan")],  # NaN counts as missing
    })

    ranges = get_missing_temperature_dates(temp_df, usage_df)
    assert len(ranges) == 1
    assert ranges[0] == (date(2024, 1, 13), date(2024, 1, 13))
