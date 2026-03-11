"""Daily usage statistics and period comparisons."""

import numpy as np
import pandas as pd


def compute_period_statistics(
    df: pd.DataFrame,
    period_column: str = "period",
) -> dict[str, dict]:
    """
    Compute summary statistics for each period.

    Returns dict keyed by period label with:
    - mean_daily_kwh, median_daily_kwh, std_daily_kwh
    - mean_daily_cost, total_cost
    - count, date_start, date_end
    """
    stats = {}
    for period, group in df.groupby(period_column):
        stats[period] = {
            "mean_daily_kwh": group["usage_kwh"].mean(),
            "median_daily_kwh": group["usage_kwh"].median(),
            "std_daily_kwh": group["usage_kwh"].std(),
            "mean_daily_cost": group["daily_cost"].mean(),
            "total_cost": group["daily_cost"].sum(),
            "count": len(group),
            "date_start": group["date"].min(),
            "date_end": group["date"].max(),
        }
    return stats


def compute_usage_change(
    stats: dict[str, dict],
    before_period: str,
    after_period: str,
) -> dict:
    """
    Calculate change metrics between two periods.

    Returns dict with absolute/percent changes and projected annual savings.
    """
    before = stats[before_period]
    after = stats[after_period]

    kwh_change = after["mean_daily_kwh"] - before["mean_daily_kwh"]
    cost_change = after["mean_daily_cost"] - before["mean_daily_cost"]

    return {
        "absolute_change_kwh": kwh_change,
        "percent_change_kwh": (kwh_change / before["mean_daily_kwh"]) * 100,
        "absolute_change_cost": cost_change,
        "percent_change_cost": (cost_change / before["mean_daily_cost"]) * 100,
        "daily_savings_kwh": -kwh_change,
        "daily_savings_cost": -cost_change,
        "projected_annual_savings_kwh": -kwh_change * 365,
        "projected_annual_savings_cost": -cost_change * 365,
    }


def compute_heating_cooling_degree_days(
    df: pd.DataFrame,
    base_temp_f: float = 65.0,
) -> pd.DataFrame:
    """Add HDD and CDD columns to the DataFrame."""
    df = df.copy()
    df["hdd"] = np.maximum(0, base_temp_f - df["temp_f"])
    df["cdd"] = np.maximum(0, df["temp_f"] - base_temp_f)
    return df
