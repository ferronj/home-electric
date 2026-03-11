from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class EquipmentEvent:
    """A user-defined equipment install/change event."""

    name: str
    label: str
    date: date
    equipment_cost: Optional[float] = None
    rebates: Optional[float] = None
    notes: Optional[str] = None

    @property
    def net_cost(self) -> float:
        cost = self.equipment_cost or 0.0
        rebate = self.rebates or 0.0
        return cost - rebate


@dataclass
class AppConfig:
    """Runtime configuration for the app."""

    latitude: float = 45.47
    longitude: float = -122.72
    timezone: str = "America/Los_Angeles"
    electricity_rate: Optional[float] = None  # auto-derived if None
    events: list[EquipmentEvent] = field(default_factory=list)
