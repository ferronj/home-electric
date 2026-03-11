"""Tests for cost analysis functions."""

import pandas as pd
import numpy as np

from src.analysis.cost import (
    compute_electricity_rate,
    estimate_annual_savings,
    compute_roi_payback,
    compute_temperature_normalized_savings,
    compute_cumulative_savings,
)


def test_compute_electricity_rate():
    df = pd.DataFrame({
        "usage_kwh": [0.52, 0.52, 0.54],
        "cost": [0.08, 0.08, 0.08],
    })
    rate = compute_electricity_rate(df)
    assert abs(rate - 0.08 * 3 / (0.52 + 0.52 + 0.54)) < 0.001


def test_estimate_annual_savings():
    before = {"mean_daily_kwh": 34.9}
    after = {"mean_daily_kwh": 28.5}
    result = estimate_annual_savings(before, after, rate_per_kwh=0.154)

    assert result["daily_kwh_saved"] > 0
    assert abs(result["daily_kwh_saved"] - 6.4) < 0.1
    assert result["annual_kwh_saved"] > 2000
    assert result["annual_cost_saved"] > 0


def test_compute_roi_payback():
    result = compute_roi_payback(equipment_cost=15000, annual_savings=500)

    assert result["simple_payback_years"] == 30.0
    assert result["monthly_savings"] > 0
    assert result["total_savings_10yr"] == 500 * 10 - 15000


def test_compute_roi_no_savings():
    result = compute_roi_payback(equipment_cost=15000, annual_savings=0)
    assert result["simple_payback_years"] == float("inf")


def test_compute_temperature_normalized_savings():
    before_params = {"k_heat": 2.25, "k_cool": 0.61, "baseload": 14.3}
    after_params = {"k_heat": 0.71, "k_cool": 0.61, "baseload": 14.3}

    # Simulate a year of temperatures (mostly cold for Portland)
    temps = pd.Series([45.0] * 180 + [75.0] * 185)  # 365 days

    result = compute_temperature_normalized_savings(
        before_params, after_params, temps, setpoint=60.0, rate_per_kwh=0.154
    )

    # k_heat dropped from 2.25 to 0.71, so savings should be significant on cold days
    assert result["annual_kwh_saved"] > 0
    assert result["annual_cost_saved"] > 0


def test_compute_cumulative_savings():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-11-05", "2024-11-06", "2024-11-07", "2024-11-08"]),
        "daily_cost": [6.0, 4.0, 3.5, 4.5],
    })
    result = compute_cumulative_savings(
        df, event_date=pd.Timestamp("2024-11-06"), baseline_daily_cost=6.0
    )

    assert len(result) == 3  # only after event date
    assert result["daily_saving"].iloc[0] == 2.0  # 6.0 - 4.0
    assert result["cumulative_saving"].iloc[2] == 2.0 + 2.5 + 1.5  # sum of savings
