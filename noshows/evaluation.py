"""Business cost and threshold selection for Maple Physio."""

import numpy as np

CALL_COST = 8
MISSED_NO_SHOW_COST = 35


def total_cost(
    y_true, call_flags, call_cost=CALL_COST, missed_cost=MISSED_NO_SHOW_COST
):
    """Cost = calls paid for + uncalled no-shows."""
    actual = np.asarray(y_true)
    called = np.asarray(call_flags)
    if actual.shape != called.shape or actual.ndim != 1:
        raise ValueError("Actual labels and calls must align.")
    if call_cost < 0 or missed_cost < 0:
        raise ValueError("Costs must not be negative.")
    called = called.astype(bool)
    return int(
        call_cost * called.sum() + missed_cost * ((actual == 1) & (~called)).sum()
    )


def choose_threshold(
    labels, probabilities, call_cost=CALL_COST, missed_cost=MISSED_NO_SHOW_COST
):
    """Find the lowest-cost threshold from 0.05 to 0.95."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if np.asarray(labels).shape != p.shape:
        raise ValueError("Labels and probabilities must align.")
    results = [
        {
            "threshold": t / 100,
            "cost": total_cost(labels, p >= t / 100, call_cost, missed_cost),
        }
        for t in range(5, 100, 5)
    ]
    return min(results, key=lambda result: result["cost"]), results
