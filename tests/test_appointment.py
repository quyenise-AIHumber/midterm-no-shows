"""Validate immutable booking inputs and feature alignment."""

from dataclasses import FrozenInstanceError

import pandas as pd
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


def test_valid_appointment_has_correct_frame(valid_booking):
    booking = Appointment(**valid_booking)
    frame = booking.to_frame()
    assert frame.shape == (1, 10)
    assert list(frame.columns) == list(Appointment.FIELDS)


def test_appointment_cannot_change(valid_booking):
    booking = Appointment(**valid_booking)
    with pytest.raises(FrozenInstanceError):
        booking.age = 20


def test_missing_distance_allowed(valid_booking):
    booking = Appointment(**{**valid_booking, "distance_km": None})
    assert pd.isna(booking.to_frame().loc[0, "distance_km"])


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("lead_days", -1),
        ("age", 111),
        ("prior_no_shows", 31),
        ("hour", 18),
        ("first_visit", 2),
        ("distance_km", 151),
    ],
)
def test_invalid_numeric_field_names_error(valid_booking, field, bad_value):
    with pytest.raises(ValueError, match=field):
        Appointment(**{**valid_booking, field: bad_value})


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("weekday", "Mon"),
        ("booking", "email"),
        ("insurance", "other"),
        ("treatment", "dental"),
    ],
)
def test_unknown_category_is_rejected(valid_booking, field, bad_value):
    with pytest.raises(ValueError, match=field):
        Appointment(**{**valid_booking, field: bad_value})
