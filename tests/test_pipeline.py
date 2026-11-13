"""Tests for data preparation and the ML pipeline."""

import pytest
from sklearn.dummy import DummyClassifier

from noshows.data import make_appointments
from noshows.pipeline import (
    FEATURES,
    build_pipeline,
    split_appointments,
)


def test_dataset_is_reproducible():
    a = make_appointments()
    b = make_appointments()

    assert a.equals(b)


@pytest.mark.parametrize(
    "excluded",
    ["appointment_id", "fee_charged", "no_show"],
)
def test_excluded_columns(excluded):
    assert excluded not in FEATURES


def test_stratified_split():
    frame = make_appointments()

    X_train, X_test, y_train, y_test = split_appointments(frame)

    assert len(X_train) == 3000
    assert len(X_test) == 1000

    assert len(y_train) == 3000
    assert len(y_test) == 1000

    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_pipeline_handles_missing_distance():
    frame = make_appointments()

    X_train, _, y_train, _ = split_appointments(frame)

    model = build_pipeline(DummyClassifier(strategy="prior"))

    model.fit(X_train, y_train)

    sample = X_train.head(3).copy()
    sample.loc[sample.index[0], "distance_km"] = float("nan")

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (3, 2)
