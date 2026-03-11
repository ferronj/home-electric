"""Shared test fixtures."""

from datetime import date
from io import StringIO

import pandas as pd
import pytest

from src.data.schemas import EquipmentEvent


@pytest.fixture
def sample_pgn_csv():
    """Small PGN interval CSV string (2 days, 4 intervals each)."""
    return StringIO(
        "TYPE,DATE,START TIME,END TIME,USAGE (kWh),COST,NOTES\n"
        "Electric usage,2024-01-12,00:00,00:14,0.52,$0.08,\n"
        "Electric usage,2024-01-12,00:15,00:29,0.52,$0.08,\n"
        "Electric usage,2024-01-12,00:30,00:44,0.52,$0.08,\n"
        "Electric usage,2024-01-12,00:45,00:59,0.54,$0.08,\n"
        "Electric usage,2024-01-13,00:00,00:14,0.30,$0.05,\n"
        "Electric usage,2024-01-13,00:15,00:29,0.30,$0.05,\n"
        "Electric usage,2024-01-13,00:30,00:44,0.30,$0.05,\n"
        "Electric usage,2024-01-13,00:45,00:59,0.30,$0.05,\n"
    )


@pytest.fixture
def sample_nasa_csv():
    """Small NASA POWER CSV with a sentinel value."""
    return StringIO(
        "YEAR,MO,DY,T2M\n"
        "2024,1,12,-1.98\n"
        "2024,1,13,-11.64\n"
        "2024,1,14,-999.0\n"
        "2024,1,15,-8.70\n"
    )


@pytest.fixture
def sample_daily_usage():
    """Pre-aggregated daily usage DataFrame."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-12", "2024-01-13", "2024-01-14", "2024-01-15"]),
        "usage_kwh": [53.79, 34.44, 59.67, 130.14],
        "daily_cost": [8.28, 5.30, 9.19, 20.04],
    })


@pytest.fixture
def sample_temperature():
    """Temperature DataFrame with one NaN."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-12", "2024-01-13", "2024-01-14", "2024-01-15"]),
        "temp_c": [-1.98, -11.64, float("nan"), -8.70],
        "temp_f": [28.436, 11.048, float("nan"), 16.34],
    })


@pytest.fixture
def sample_events():
    """List of equipment events."""
    return [
        EquipmentEvent(
            name="heat_pump",
            label="Heat Pump",
            date=date(2024, 11, 6),
            equipment_cost=15000.0,
            rebates=2000.0,
        ),
    ]


@pytest.fixture
def two_events():
    """Two equipment events for multi-period testing."""
    return [
        EquipmentEvent(
            name="heat_pump",
            label="Heat Pump",
            date=date(2024, 11, 6),
        ),
        EquipmentEvent(
            name="hpwh",
            label="HP Water Heater",
            date=date(2025, 3, 15),
        ),
    ]
