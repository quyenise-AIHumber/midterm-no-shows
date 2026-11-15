"""Test threshold selection when costs are extreme."""

from noshows.evaluation import choose_threshold


def test_threshold_when_calls_are_expensive():
    labels = [1, 0, 1, 0]
    probabilities = [0.9, 0.8, 0.3, 0.2]

    best, results = choose_threshold(
        labels, probabilities, call_cost=100, missed_cost=1
    )

    # Calling nobody costs 2: two missed appointments at $1 each.
    assert best["threshold"] == 0.95
    assert best["cost"] == 2


def test_threshold_when_missed_visits_are_expensive():
    labels = [1, 1, 1, 1]
    probabilities = [0.9, 0.8, 0.3, 0.2]

    best, results = choose_threshold(
        labels, probabilities, call_cost=1, missed_cost=100
    )

    # Calling everyone costs 4: four calls at $1 each.
    assert best["threshold"] == 0.05
    assert best["cost"] == 4
