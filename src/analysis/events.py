"""Equipment event management — CRUD operations with JSON persistence."""

import json
from datetime import date
from pathlib import Path

from src.data.schemas import EquipmentEvent

DEFAULT_EVENTS_PATH = Path("data/events.json")


def _coerce_money(value) -> float | None:
    """Permissive parse for cost/rebate JSON values.

    Accepts numbers, numeric strings (with optional $ / commas / whitespace),
    None, and empty strings. Returns None when the value is missing or unparseable.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().lstrip("$").replace(",", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def load_events(path: Path = DEFAULT_EVENTS_PATH) -> list[EquipmentEvent]:
    """Load events from JSON file. Returns empty list if file doesn't exist."""
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    return [
        EquipmentEvent(
            name=e["name"],
            label=e["label"],
            date=date.fromisoformat(e["date"]),
            equipment_cost=_coerce_money(e.get("equipment_cost")),
            rebates=_coerce_money(e.get("rebates")),
            notes=e.get("notes"),
        )
        for e in data.get("events", [])
    ]


def save_events(events: list[EquipmentEvent], path: Path = DEFAULT_EVENTS_PATH) -> None:
    """Save events to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "events": [
            {
                "name": e.name,
                "label": e.label,
                "date": e.date.isoformat(),
                "equipment_cost": e.equipment_cost,
                "rebates": e.rebates,
                "notes": e.notes,
            }
            for e in sorted(events, key=lambda e: e.date)
        ]
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def validate_events(
    events: list[EquipmentEvent],
    data_start: date,
    data_end: date,
) -> list[str]:
    """
    Validate events against the data date range.
    Returns a list of warning messages (empty if all OK).
    """
    warnings = []

    for event in events:
        if event.date < data_start:
            warnings.append(
                f"'{event.label}' date ({event.date}) is before data starts ({data_start})"
            )
        if event.date > data_end:
            warnings.append(
                f"'{event.label}' date ({event.date}) is after data ends ({data_end})"
            )

        days_after = (data_end - event.date).days
        if 0 <= days_after < 14:
            warnings.append(
                f"'{event.label}' has only {days_after} days of post-install data"
            )

    # Check for duplicate dates
    dates = [e.date for e in events]
    if len(dates) != len(set(dates)):
        warnings.append("Multiple events share the same date")

    return warnings
