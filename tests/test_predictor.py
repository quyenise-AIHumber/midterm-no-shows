"""Check model loading, risk, decisions and missing distance warning."""

import joblib
import pytest

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor


@pytest.fixture
def booking_data():
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


def test_probability_and_action(booking_data):
    result = NoShowPredictor().predict(Appointment(**booking_data))
    assert 0 <= result["probability_no_show"] <= 1
    total = result["probability_no_show"] + result["probability_attend"]
    assert total == pytest.approx(1)
    expected_call = result["probability_no_show"] >= result["threshold"]
    assert result["call_patient"] == expected_call


def test_decision_at_threshold_boundaries():
    predictor = NoShowPredictor()
    cutoff = predictor.threshold
    assert predictor.should_call(cutoff)
    assert not predictor.should_call(cutoff - 0.000001)
    assert predictor.should_call(cutoff + 0.000001)


def test_missing_distance_gives_warning(booking_data):
    booking = Appointment(**{**booking_data, "distance_km": None})
    result = NoShowPredictor().predict(booking)
    assert 0 <= result["probability_no_show"] <= 1
    assert any("distance_km" in item for item in result["warnings"])


def test_model_loaded_once(monkeypatch, booking_data):
    real_load = joblib.load
    calls = []

    def tracked_load(path):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr("noshows.predictor.joblib.load", tracked_load)
    predictor = NoShowPredictor()
    booking = Appointment(**booking_data)
    predictor.predict(booking)
    predictor.predict(booking)
    assert len(calls) == 1


def test_riskier_booking_scores_higher():
    safe = Appointment(
        lead_days=1,
        age=60,
        prior_no_shows=0,
        hour=12,
        weekday="wed",
        booking="front_desk",
        insurance="provincial",
        treatment="acupuncture",
        first_visit=0,
        distance_km=1,
    )
    risky = Appointment(
        lead_days=55,
        age=19,
        prior_no_shows=5,
        hour=17,
        weekday="mon",
        booking="online",
        insurance="self_pay",
        treatment="physio",
        first_visit=1,
        distance_km=60,
    )
    predictor = NoShowPredictor()
    riskier_prob = predictor.predict(risky)["probability_no_show"]
    safer_prob = predictor.predict(safe)["probability_no_show"]
    assert riskier_prob > safer_prob


def test_readable_repr():
    assert "NoShowPredictor" in repr(NoShowPredictor())
