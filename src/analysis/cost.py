"""Cost analysis, ROI, and payback calculations."""

import numpy as np
import pandas as pd


def compute_electricity_rate(interval_df: pd.DataFrame) -> float:
    """
    Derive blended $/kWh rate from PGN interval data.
    Returns weighted average: sum(cost) / sum(usage_kwh).
    """
    total_cost = interval_df["cost"].sum()
    total_kwh = interval_df["usage_kwh"].sum()
    if total_kwh == 0:
        return 0.0
    return total_cost / total_kwh


def estimate_annual_savings(
    before_stats: dict,
    after_stats: dict,
    rate_per_kwh: float,
) -> dict:
    """
    Project annual savings from simple mean comparison.

    Returns annual_kwh_saved, annual_cost_saved, monthly_cost_saved.
    """
    daily_kwh_saved = before_stats["mean_daily_kwh"] - after_stats["mean_daily_kwh"]
    annual_kwh = daily_kwh_saved * 365
    annual_cost = annual_kwh * rate_per_kwh
    return {
        "daily_kwh_saved": daily_kwh_saved,
        "annual_kwh_saved": annual_kwh,
        "annual_cost_saved": annual_cost,
        "monthly_cost_saved": annual_cost / 12,
    }


def compute_roi_payback(
    equipment_cost: float,
    annual_savings: float,
) -> dict:
    """
    Simple payback period and cumulative savings projections.
    """
    if annual_savings <= 0:
        return {
            "simple_payback_years": float("inf"),
            "monthly_savings": 0.0,
            "total_savings_5yr": 0.0,
            "total_savings_10yr": 0.0,
        }

    return {
        "simple_payback_years": equipment_cost / annual_savings,
        "monthly_savings": annual_savings / 12,
        "total_savings_5yr": annual_savings * 5 - equipment_cost,
        "total_savings_10yr": annual_savings * 10 - equipment_cost,
    }


def compute_temperature_normalized_savings(
    model_before_params: dict,
    model_after_params: dict,
    typical_year_temperatures: pd.Series,
    setpoint: float = 60.0,
    rate_per_kwh: float = 0.154,
) -> dict:
    """
    Using piecewise linear model parameters, compute what usage WOULD have been
    before vs after for a typical temperature year.

    Each params dict should have: k_heat, k_cool, baseload

    Returns annual_kwh_saved and annual_cost_saved.
    """

    def _piecewise_usage(temp_f: float, params: dict) -> float:
        if temp_f < setpoint:
            return params["k_heat"] * (setpoint - temp_f) + params["baseload"]
        else:
            return params["k_cool"] * (temp_f - setpoint) + params["baseload"]

    before_usage = typical_year_temperatures.apply(
        lambda t: _piecewise_usage(t, model_before_params)
    )
    after_usage = typical_year_temperatures.apply(
        lambda t: _piecewise_usage(t, model_after_params)
    )

    annual_kwh_saved = (before_usage - after_usage).sum()
    return {
        "annual_kwh_saved": annual_kwh_saved,
        "annual_cost_saved": annual_kwh_saved * rate_per_kwh,
    }


def compute_savings_from_full_model_result(
    model_result: dict,
    daily_df: pd.DataFrame,
    rate_per_kwh: float,
    setpoint: float = 60.0,
) -> dict | None:
    """
    Translate a Full Temperature model result into temperature-normalized savings.

    Returns None when the result isn't a full_temperature run, has fewer than two
    periods, or the dataframe lacks usable temp_f data. First period is treated as
    "before", last as "after".
    """
    if not model_result or model_result.get("type") != "full_temperature":
        return None

    params = model_result.get("params") or {}
    periods = params.get("periods") or {}
    if len(periods) < 2:
        return None

    if "temp_f" not in daily_df.columns:
        return None
    temps = daily_df["temp_f"].dropna()
    if temps.empty:
        return None

    period_names = list(periods.keys())
    before_name, after_name = period_names[0], period_names[-1]
    baseload_mean = float(params.get("baseload", {}).get("mean", 0.0))

    before_params = {
        "k_heat": float(periods[before_name]["k_heat_mean"]),
        "k_cool": float(periods[before_name]["k_cool_mean"]),
        "baseload": baseload_mean,
    }
    after_params = {
        "k_heat": float(periods[after_name]["k_heat_mean"]),
        "k_cool": float(periods[after_name]["k_cool_mean"]),
        "baseload": baseload_mean,
    }

    typical_year_days = 365
    if len(temps) >= typical_year_days:
        typical_temps = temps.iloc[-typical_year_days:].reset_index(drop=True)
    else:
        scale = typical_year_days / len(temps)
        typical_temps = temps.reset_index(drop=True)
        result = compute_temperature_normalized_savings(
            before_params,
            after_params,
            typical_temps,
            setpoint=setpoint,
            rate_per_kwh=rate_per_kwh,
        )
        return {
            "annual_kwh_saved": result["annual_kwh_saved"] * scale,
            "annual_cost_saved": result["annual_cost_saved"] * scale,
            "before_period": before_name,
            "after_period": after_name,
            "n_temp_days_used": len(temps),
        }

    result = compute_temperature_normalized_savings(
        before_params,
        after_params,
        typical_temps,
        setpoint=setpoint,
        rate_per_kwh=rate_per_kwh,
    )
    return {
        "annual_kwh_saved": result["annual_kwh_saved"],
        "annual_cost_saved": result["annual_cost_saved"],
        "before_period": before_name,
        "after_period": after_name,
        "n_temp_days_used": len(temps),
    }


def compute_cumulative_savings(
    df: pd.DataFrame,
    event_date: pd.Timestamp,
    baseline_daily_cost: float,
) -> pd.DataFrame:
    """
    Compute cumulative dollar savings since an event date.

    Returns DataFrame with: date, daily_saving, cumulative_saving
    """
    after = df[df["date"] >= event_date].copy()
    after["daily_saving"] = baseline_daily_cost - after["daily_cost"]
    after["cumulative_saving"] = after["daily_saving"].cumsum()
    return after[["date", "daily_saving", "cumulative_saving"]].reset_index(drop=True)
