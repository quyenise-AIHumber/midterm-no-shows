"""Tests that threshold optimization reflects the business cost rule."""

import numpy as np
import pytest

from noshows.evaluation import choose_threshold, total_cost


def test_hand_worked_cost():
    assert total_cost([1, 0, 1, 0], [True, True, False, False]) == 51


@pytest.mark.parametrize("call_cost,missed_cost,expected", [(1000, 1, 0), (0, 1000, 3)])
def test_threshold_cost_extremes(call_cost, missed_cost, expected):
    labels = [1, 1, 1]
    probabilities = np.array([0.2, 0.5, 0.9])
    chosen, _ = choose_threshold(labels, probabilities, call_cost, missed_cost)
    predicted_calls = probabilities >= chosen["threshold"]
    assert predicted_calls.sum() == expected


def test_threshold_in_expected_range():
    result, rows = choose_threshold([1, 0, 1, 0], [0.8, 0.1, 0.6, 0.2])
    assert 0.05 <= result["threshold"] <= 0.95
    assert len(rows) == 19
