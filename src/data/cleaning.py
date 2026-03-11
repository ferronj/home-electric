"""Data aggregation, merging, and cleaning utilities."""

from datetime import date

import numpy as np
import pandas as pd

from src.data.schemas import EquipmentEvent


def aggregate_to_daily(interval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 15-minute interval data to daily totals.

    Returns DataFrame with columns: date, usage_kwh, daily_cost
    """
    daily = interval_df.groupby("date").agg(
        usage_kwh=("usage_kwh", "sum"),
        daily_cost=("cost", "sum"),
    ).reset_index()
    return daily.sort_values("date").reset_index(drop=True)


def merge_usage_temperature(
    daily_usage: pd.DataFrame,
    temperature: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join daily usage with temperature on date.
    Rows with missing temperature are kept (temp columns will be NaN).
    """
    merged = daily_usage.merge(temperature, on="date", how="left")
    return merged.sort_values("date").reset_index(drop=True)


def add_event_periods(
    df: pd.DataFrame,
    events: list[EquipmentEvent],
) -> pd.DataFrame:
    """
    Add period labels based on equipment event dates.

    With no events: all rows get period="Baseline".
    With events sorted chronologically, creates segments:
    - Before first event: "Baseline"
    - After each event until the next: "After {event.label}"
    """
    df = df.copy()

    if not events:
        df["period"] = "Baseline"
        return df

    sorted_events = sorted(events, key=lambda e: e.date)

    # Add boolean columns for each event
    for event in sorted_events:
        col_name = f"{event.name}_installed"
        df[col_name] = df["date"] >= pd.Timestamp(event.date)

    # Build period labels
    conditions = []
    labels = []

    # Before first event
    first_date = pd.Timestamp(sorted_events[0].date)
    conditions.append(df["date"] < first_date)
    labels.append("Baseline")

    # Between events and after last
    for i, event in enumerate(sorted_events):
        event_date = pd.Timestamp(event.date)
        if i < len(sorted_events) - 1:
            next_date = pd.Timestamp(sorted_events[i + 1].date)
            conditions.append((df["date"] >= event_date) & (df["date"] < next_date))
        else:
            conditions.append(df["date"] >= event_date)
        labels.append(f"After {event.label}")

    df["period"] = np.select(conditions, labels, default="Unknown")
    return df


def interpolate_temperature_gaps(
    df: pd.DataFrame, max_gap: int = 3
) -> pd.DataFrame:
    """
    Fill small temperature gaps (up to max_gap consecutive NaN days)
    via linear interpolation. Flags interpolated values.
    """
    df = df.copy()
    df["temp_interpolated"] = False

    if "temp_c" not in df.columns:
        return df

    # Identify NaN runs
    is_nan = df["temp_c"].isna()
    if not is_nan.any():
        return df

    # Group consecutive NaNs
    nan_groups = is_nan.ne(is_nan.shift()).cumsum()
    nan_run_lengths = is_nan.groupby(nan_groups).transform("sum")

    # Only interpolate gaps <= max_gap
    small_gaps = is_nan & (nan_run_lengths <= max_gap)
    df.loc[small_gaps, "temp_interpolated"] = True

    # Interpolate all, then restore large gaps
    temp_c_interp = df["temp_c"].interpolate(method="linear")
    temp_f_interp = df["temp_f"].interpolate(method="linear")

    large_gaps = is_nan & (nan_run_lengths > max_gap)
    temp_c_interp[large_gaps] = np.nan
    temp_f_interp[large_gaps] = np.nan
    df.loc[large_gaps, "temp_interpolated"] = False

    df["temp_c"] = temp_c_interp
    df["temp_f"] = temp_f_interp

    return df
