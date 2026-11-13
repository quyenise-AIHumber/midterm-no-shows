
"""Business cost calculations for Maple Physio."""

import numpy as np

CALL_COST = 8
MISSED_NO_SHOW_COST = 35


def total_cost(y_true, call_flags):
    """Calculate the cost of a patient-calling policy."""

    actual = np.asarray(y_true)
    called = np.asarray(call_flags)

    if actual.shape != called.shape or actual.ndim != 1:
        raise ValueError("Actual labels and calls must align.")

    called = called.astype(bool)

    number_of_calls = called.sum()

    missed_no_shows = (
        (actual == 1) & (~called)
    ).sum()

    return int(
        CALL_COST * number_of_calls
        + MISSED_NO_SHOW_COST * missed_no_shows
    )
