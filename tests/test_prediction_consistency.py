
"""Verify probability and action consistency across bookings."""

import pytest

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor


@pytest.mark.parametrize(
    "lead_days,prior_no_shows,booking_type",
    [
        (1, 0, "front_desk"),
        (15, 2, "phone"),
        (60, 5, "online"),
    ],
)
def test_prediction_consistency(
    lead_days, prior_no_shows, booking_type
):
    appointment = Appointment(
        lead_days=lead_days,
        age=40,
        prior_no_shows=prior_no_shows,
        hour=10,
        weekday="tue",
        booking=booking_type,
        insurance="private",
        treatment="physio",
        first_visit=0,
        distance_km=8.0,
    )

    predictor = NoShowPredictor()
    result = predictor.predict(appointment)

    probability = result["probability_no_show"]

    assert 0 <= probability <= 1
    assert result["probability_attend"] == pytest.approx(
        1 - probability
    )
    assert result["call_patient"] == (
        probability >= predictor.threshold
    )
    assert result["action"] == (
        "Call patient to confirm"
        if result["call_patient"]
        else "No call needed"
    )
