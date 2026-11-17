"""Tests for the business cost rule."""

from itertools import product

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


@pytest.mark.parametrize(
    "actual, expected_costs_by_call_count, cheapest_policy",
    [
        ([0, 0, 0], (0, 8, 16, 24), (False, False, False)),
        ([1, 1, 1], (105, 78, 51, 24), (True, True, True)),
    ],
    ids=["call-nobody-cheapest", "call-everybody-cheapest"],
)
def test_total_cost_identifies_cheapest_call_policy(
    actual, expected_costs_by_call_count, cheapest_policy
):
    """Compare hand-worked costs for all eight policies on three patients."""
    costs = {
        policy: total_cost(actual, policy)
        for policy in product((False, True), repeat=len(actual))
    }

    # With all attendees: 8 * calls. With all no-shows: 8 * calls + 35 * uncalled.
    for policy, cost in costs.items():
        assert cost == expected_costs_by_call_count[sum(policy)]

    assert min(costs, key=costs.get) == cheapest_policy
    assert all(
        costs[cheapest_policy] < cost
        for policy, cost in costs.items()
        if policy != cheapest_policy
    )


def test_mismatched_input_lengths():
    with pytest.raises(ValueError, match=r"^Actual labels and calls must align\.$"):
        total_cost([1, 0, 1], [True, False])


@pytest.mark.parametrize(
    "actual, called",
    [
        (1, True),
        ([[1, 0]], [[True, False]]),
        ([1, 0], [[True, False]]),
    ],
    ids=["scalar-inputs", "two-dimensional-inputs", "mismatched-dimensions"],
)
def test_total_cost_rejects_invalid_dimensions(actual, called):
    with pytest.raises(ValueError, match=r"^Actual labels and calls must align\.$"):
        total_cost(actual, called)


@pytest.mark.parametrize("call_cost, missed_cost", [(-1, 35), (8, -1)])
def test_total_cost_rejects_negative_costs(call_cost, missed_cost):
    with pytest.raises(ValueError, match=r"^Costs must not be negative\.$"):
        total_cost([1, 0], [True, False], call_cost=call_cost, missed_cost=missed_cost)
