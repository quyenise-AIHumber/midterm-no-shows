"""Tests for the business cost rule."""

import pytest
from noshows.evaluation import total_cost


@pytest.mark.parametrize(
    "actual, called, expected",
    [
        ([1, 0, 1, 0], [True, True, False, False], 51),
        ([1, 0, 1, 0], [False, False, False, False], 70),
        ([1, 0, 1, 0], [True, True, True, True], 32),
    ],
)
def test_total_cost(actual, called, expected):
    assert total_cost(actual, called) == expected


def test_mismatched_input_lengths():
    with pytest.raises(ValueError):
        total_cost([1, 0, 1], [True, False])
