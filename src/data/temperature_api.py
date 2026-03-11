"""Open-Meteo API client for fetching daily temperature data."""

from datetime import date, timedelta

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_temperature_range(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timezone: str = "America/Los_Angeles",
) -> pd.DataFrame:
    """
    Fetch daily mean temperature from Open-Meteo.

    Uses Archive API for historical data (>7 days ago) and Forecast API
    for recent data (<7 days ago). Merges if range spans both.

    Returns DataFrame with columns: date, temp_c, temp_f
    """
    today = date.today()
    archive_cutoff = today - timedelta(days=7)

    frames = []

    # Historical portion
    if start_date < archive_cutoff:
        archive_end = min(end_date, archive_cutoff)
        frames.append(_fetch_archive(latitude, longitude, start_date, archive_end, timezone))

    # Recent portion
    if end_date >= archive_cutoff:
        recent_start = max(start_date, archive_cutoff)
        past_days = (today - recent_start).days + 1
        past_days = min(past_days, 92)  # API limit
        frames.append(_fetch_recent(latitude, longitude, past_days, timezone))

    if not frames:
        return pd.DataFrame(columns=["date", "temp_c", "temp_f"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined[
        (combined["date"] >= pd.Timestamp(start_date))
        & (combined["date"] <= pd.Timestamp(end_date))
    ]
    return combined.sort_values("date").reset_index(drop=True)


def _fetch_archive(
    lat: float, lon: float, start: date, end: date, tz: str
) -> pd.DataFrame:
    """Fetch from Open-Meteo Archive API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean",
        "timezone": tz,
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_response(resp.json())


def _fetch_recent(
    lat: float, lon: float, past_days: int, tz: str
) -> pd.DataFrame:
    """Fetch from Open-Meteo Forecast API with past_days."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": tz,
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_response(resp.json())


def _parse_response(data: dict) -> pd.DataFrame:
    """Parse Open-Meteo JSON response into a DataFrame."""
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])

    df = pd.DataFrame({"date": pd.to_datetime(dates), "temp_c": temps})
    df["temp_c"] = df["temp_c"].astype(float)
    df["temp_f"] = df["temp_c"] * 9 / 5 + 32
    return df


def get_missing_temperature_dates(
    existing_temp_df: pd.DataFrame,
    usage_df: pd.DataFrame,
) -> list[tuple[date, date]]:
    """
    Find contiguous date ranges in usage data that lack temperature data.
    Returns list of (start_date, end_date) tuples to fetch.
    """
    if usage_df.empty:
        return []

    usage_dates = set(usage_df["date"].dt.date)
    if existing_temp_df.empty:
        temp_dates = set()
    else:
        valid_temp = existing_temp_df.dropna(subset=["temp_c"])
        temp_dates = set(valid_temp["date"].dt.date)

    missing = sorted(usage_dates - temp_dates)
    if not missing:
        return []

    # Group into contiguous ranges
    ranges = []
    range_start = missing[0]
    prev = missing[0]
    for d in missing[1:]:
        if (d - prev).days > 1:
            ranges.append((range_start, prev))
            range_start = d
        prev = d
    ranges.append((range_start, prev))

    return ranges
