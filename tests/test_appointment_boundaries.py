
"""Test appointment input boundaries and unusual numeric values."""

import math

import pytest

from noshows.appointment import Appointment


@pytest.fixture
def valid_booking():
    return {
        "lead_days": 7,
        "age": 40,
        "prior_no_shows": 1,
        "hour": 10,
        "weekday": "tue",
        "booking": "phone",
        "insurance": "private",
        "treatment": "physio",
        "first_visit": 0,
        "distance_km": 8.0,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("lead_days", 0),
        ("lead_days", 120),
        ("age", 12),
        ("age", 110),
        ("prior_no_shows", 0),
        ("prior_no_shows", 30),
        ("hour", 8),
        ("hour", 17),
        ("distance_km", 0.1),
        ("distance_km", 150),
    ],
)
def test_allowed_boundary_values(valid_booking, field, value):
    booking = Appointment(**{**valid_booking, field: value})
    assert getattr(booking, field) == value


@pytest.mark.parametrize("value", [0.5, -0.5, 1.5])
def test_first_visit_rejects_fractional_values(valid_booking, value):
    with pytest.raises(ValueError, match="first_visit"):
        Appointment(**{**valid_booking, "first_visit": value})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_distance_rejected(valid_booking, value):
    with pytest.raises(ValueError, match="distance_km"):
        Appointment(**{**valid_booking, "distance_km": value})
