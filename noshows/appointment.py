"""Immutable, validated booking details for Maple Physio."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

import pandas as pd

from .pipeline import FEATURES


@dataclass(frozen=True)
class Appointment:
    """One appointment with booking-time inputs only."""

    lead_days: float
    age: float
    prior_no_shows: float
    hour: float
    weekday: str
    booking: str
    insurance: str
    treatment: str
    first_visit: int
    distance_km: float | None = None

    FIELDS: ClassVar[tuple[str, ...]] = tuple(FEATURES)
    NUMERIC_RULES: ClassVar[dict[str, tuple[float, float]]] = {
        "lead_days": (0, 120),
        "age": (12, 110),
        "prior_no_shows": (0, 30),
        "hour": (8, 17),
        "first_visit": (0, 1),
        "distance_km": (0.1, 150),
    }
    CATEGORIES: ClassVar[dict[str, tuple[str, ...]]] = {
        "weekday": ("mon", "tue", "wed", "thu", "fri"),
        "booking": ("online", "phone", "front_desk"),
        "insurance": ("private", "provincial", "self_pay"),
        "treatment": ("physio", "massage", "acupuncture"),
    }

    def __post_init__(self):
        for field, (low, high) in self.NUMERIC_RULES.items():
            value = getattr(self, field)
            if field == "distance_km" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not low <= value <= high
            ):
                raise ValueError(f"{field}: must be a number from {low:g} to {high:g}")

        if self.first_visit not in (0, 1):
            raise ValueError("first_visit: must be 0 or 1")

        for field, choices in self.CATEGORIES.items():
            value = getattr(self, field)
            if value not in choices:
                raise ValueError(f"{field}: choose one of {', '.join(choices)}")

    def to_frame(self) -> pd.DataFrame:
        """Return exactly the columns expected by the trained model."""
        values = {field: getattr(self, field) for field in self.FIELDS}
        if values["distance_km"] is None:
            values["distance_km"] = float("nan")
        return pd.DataFrame([values], columns=list(self.FIELDS))
