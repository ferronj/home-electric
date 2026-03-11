"""Tests for data cleaning and aggregation."""

from datetime import date

import numpy as np
import pandas as pd

from src.data.cleaning import (
    aggregate_to_daily,
    merge_usage_temperature,
    add_event_periods,
    interpolate_temperature_gaps,
)
from src.data.schemas import EquipmentEvent


def test_aggregate_to_daily(sample_pgn_csv):
    from src.data.loader import load_pgn_interval_data

    interval_df = load_pgn_interval_data(sample_pgn_csv)
    daily = aggregate_to_daily(interval_df)

    assert len(daily) == 2
    assert list(daily.columns) == ["date", "usage_kwh", "daily_cost"]
    # Day 1: 0.52 + 0.52 + 0.52 + 0.54 = 2.10
    assert abs(daily["usage_kwh"].iloc[0] - 2.10) < 0.01
    # Day 2: 0.30 * 4 = 1.20
    assert abs(daily["usage_kwh"].iloc[1] - 1.20) < 0.01


def test_merge_usage_temperature(sample_daily_usage, sample_temperature):
    merged = merge_usage_temperature(sample_daily_usage, sample_temperature)

    assert len(merged) == 4
    assert "temp_f" in merged.columns
    # NaN temp preserved
    assert np.isnan(merged["temp_c"].iloc[2])


def test_add_event_periods_no_events(sample_daily_usage):
    result = add_event_periods(sample_daily_usage, [])
    assert all(result["period"] == "Baseline")


def test_add_event_periods_single_event():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-11-01", "2024-11-05", "2024-11-06", "2024-11-10"
        ]),
        "usage_kwh": [30, 35, 25, 28],
    })
    events = [EquipmentEvent(name="hp", label="Heat Pump", date=date(2024, 11, 6))]
    result = add_event_periods(df, events)

    assert result["period"].iloc[0] == "Baseline"
    assert result["period"].iloc[1] == "Baseline"
    assert result["period"].iloc[2] == "After Heat Pump"
    assert result["period"].iloc[3] == "After Heat Pump"


def test_add_event_periods_two_events():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-10-01", "2024-11-10", "2025-03-20"
        ]),
        "usage_kwh": [30, 25, 20],
    })
    events = [
        EquipmentEvent(name="hp", label="Heat Pump", date=date(2024, 11, 6)),
        EquipmentEvent(name="hpwh", label="Water Heater", date=date(2025, 3, 15)),
    ]
    result = add_event_periods(df, events)

    assert result["period"].iloc[0] == "Baseline"
    assert result["period"].iloc[1] == "After Heat Pump"
    assert result["period"].iloc[2] == "After Water Heater"


def test_interpolate_temperature_small_gap():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "usage_kwh": [30, 35, 32, 28],
        "temp_c": [5.0, float("nan"), float("nan"), 11.0],
        "temp_f": [41.0, float("nan"), float("nan"), 51.8],
    })
    result = interpolate_temperature_gaps(df, max_gap=3)

    assert not np.isnan(result["temp_c"].iloc[1])
    assert not np.isnan(result["temp_c"].iloc[2])
    assert result["temp_interpolated"].iloc[1] == True
    assert result["temp_interpolated"].iloc[0] == False


def test_interpolate_temperature_large_gap():
    df = pd.DataFrame({
        "date": pd.to_datetime([f"2024-01-{d:02d}" for d in range(1, 7)]),
        "usage_kwh": [30] * 6,
        "temp_c": [5.0, float("nan"), float("nan"), float("nan"), float("nan"), 11.0],
        "temp_f": [41.0, float("nan"), float("nan"), float("nan"), float("nan"), 51.8],
    })
    result = interpolate_temperature_gaps(df, max_gap=3)

    # Gap of 4 > max_gap of 3, should NOT be interpolated
    assert np.isnan(result["temp_c"].iloc[1])
    assert result["temp_interpolated"].iloc[1] == False
