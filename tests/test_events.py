"""Tests for events loading and money-value coercion."""

import json
from datetime import date

from src.analysis.events import _coerce_money, load_events


def test_coerce_money_numbers():
    assert _coerce_money(17000) == 17000.0
    assert _coerce_money(17000.5) == 17000.5
    assert isinstance(_coerce_money(17000), float)


def test_coerce_money_numeric_strings():
    assert _coerce_money("17000") == 17000.0
    assert _coerce_money(" 17000 ") == 17000.0
    assert _coerce_money("$17,000") == 17000.0
    assert _coerce_money("4200.50") == 4200.50


def test_coerce_money_none_and_blank():
    assert _coerce_money(None) is None
    assert _coerce_money("") is None
    assert _coerce_money("   ") is None


def test_coerce_money_garbage_returns_none():
    assert _coerce_money("free") is None
    assert _coerce_money("N/A") is None
    assert _coerce_money({}) is None


def test_load_events_coerces_string_equipment_cost(tmp_path):
    """The deployed events.json had string costs that crashed st.number_input."""
    events_file = tmp_path / "events.json"
    events_file.write_text(json.dumps({
        "events": [
            {
                "name": "heat_pump",
                "label": "Heat Pump",
                "date": "2024-11-06",
                "equipment_cost": "17000",
                "rebates": None,
            },
            {
                "name": "water_heater",
                "label": "Water Heater",
                "date": "2025-08-28",
                "equipment_cost": "4200",
                "rebates": "",
            },
        ]
    }))

    events = load_events(events_file)

    assert len(events) == 2
    assert events[0].equipment_cost == 17000.0
    assert isinstance(events[0].equipment_cost, float)
    assert events[0].rebates is None
    assert events[1].equipment_cost == 4200.0
    assert events[1].rebates is None
    assert events[1].date == date(2025, 8, 28)
    # net_cost still works
    assert events[0].net_cost == 17000.0


def test_load_events_missing_file_returns_empty(tmp_path):
    assert load_events(tmp_path / "does_not_exist.json") == []
