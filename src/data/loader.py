"""CSV parsers for PGN electricity usage and NASA POWER temperature data."""

from io import StringIO
from pathlib import Path
from typing import IO, Union

import pandas as pd


FileInput = Union[str, Path, IO[str], IO[bytes]]


def _read_csv(file_or_path: FileInput, **kwargs) -> pd.DataFrame:
    """Read CSV handling both file paths and file-like objects (e.g., st.file_uploader)."""
    if isinstance(file_or_path, (str, Path)):
        return pd.read_csv(file_or_path, encoding="utf-8-sig", **kwargs)
    # File-like object — try to read and decode if bytes
    content = file_or_path.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    return pd.read_csv(StringIO(content), **kwargs)


def load_pgn_interval_data(file_or_path: FileInput) -> pd.DataFrame:
    """
    Load PGN 15-minute interval electricity usage CSV.

    Returns DataFrame with columns: date, start_time, end_time, usage_kwh, cost
    """
    df = _read_csv(file_or_path)
    df = df.rename(columns=lambda c: c.strip())

    df["date"] = pd.to_datetime(df["DATE"])
    df["start_time"] = df["START TIME"].str.strip()
    df["end_time"] = df["END TIME"].str.strip()
    df["usage_kwh"] = df["USAGE (kWh)"].astype(float)
    df["cost"] = df["COST"].str.replace("$", "", regex=False).astype(float)

    return df[["date", "start_time", "end_time", "usage_kwh", "cost"]].copy()


def load_nasa_power_temperature(file_or_path: FileInput) -> pd.DataFrame:
    """
    Load NASA POWER daily temperature CSV.
    Replaces -999.0 sentinel with NaN.

    Returns DataFrame with columns: date, temp_c, temp_f
    """
    df = _read_csv(file_or_path)
    df = df.rename(columns=lambda c: c.strip())

    df["date"] = pd.to_datetime(
        df[["YEAR", "MO", "DY"]].rename(columns={"YEAR": "year", "MO": "month", "DY": "day"})
    )
    df["temp_c"] = df["T2M"].replace(-999.0, float("nan"))
    df["temp_f"] = df["temp_c"] * 9 / 5 + 32

    return df[["date", "temp_c", "temp_f"]].copy()


def load_multiple_pgn_files(files: list[FileInput]) -> pd.DataFrame:
    """Load and concatenate multiple PGN CSVs, deduplicating by (date, start_time)."""
    frames = [load_pgn_interval_data(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "start_time"], keep="last")
    return combined.sort_values(["date", "start_time"]).reset_index(drop=True)


def load_multiple_temperature_files(files: list[FileInput]) -> pd.DataFrame:
    """Load and concatenate multiple NASA POWER CSVs, preferring non-NaN values."""
    frames = [load_nasa_power_temperature(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date")
    # Prefer rows with actual temperature data over NaN
    combined = combined.dropna(subset=["temp_c"]).drop_duplicates(subset=["date"], keep="last")
    # Add back dates that only have NaN
    all_dates = pd.concat(frames, ignore_index=True)[["date"]].drop_duplicates()
    result = all_dates.merge(combined, on="date", how="left")
    return result.sort_values("date").reset_index(drop=True)
