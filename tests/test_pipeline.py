"""Tests for data preparation and the ML pipeline."""

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

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


@pytest.fixture(scope="module")
def fitted_pipeline():
    """Fit a test-only model using the training partition exclusively."""
    frame = make_appointments()
    X_train, _, y_train, _ = split_appointments(frame)
    model = build_pipeline(LogisticRegression(max_iter=1000, random_state=11))
    model.fit(X_train, y_train)
    return model, X_train


def assert_valid_probabilities(probabilities, n_rows):
    """Check binary probability shape, bounds, finiteness and normalization."""
    assert probabilities.shape == (n_rows, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_numeric_preprocessor_uses_median_imputer(fitted_pipeline):
    model, _ = fitted_pipeline
    numeric = model.named_steps["preprocessor"].named_transformers_["numeric"]
    imputer = numeric.named_steps["imputer"]

    assert isinstance(imputer, SimpleImputer)
    assert imputer.strategy == "median"


def test_imputer_learns_medians_from_training_data_only(fitted_pipeline):
    model, X_train = fitted_pipeline
    numeric = model.named_steps["preprocessor"].named_transformers_["numeric"]
    imputer = numeric.named_steps["imputer"]
    columns = list(imputer.feature_names_in_)
    distance_index = columns.index("distance_km")

    np.testing.assert_allclose(
        imputer.statistics_, X_train[columns].median().to_numpy()
    )
    assert imputer.statistics_[distance_index] == pytest.approx(
        X_train["distance_km"].median()
    )


@pytest.mark.parametrize(
    "distances",
    [
        [1, 2, 3, 4, 8, 30, np.nan, np.nan],
        [1, 2, 4, 8, 30, np.nan, np.nan, np.nan],
    ],
    ids=["even-observed-count", "odd-observed-count"],
)
def test_imputer_computes_synthetic_distance_median(distances):
    """Use skewed synthetic values to distinguish median from mean and mode."""
    X_train = make_appointments(n=len(distances))[FEATURES].copy()
    X_train["distance_km"] = distances
    y_train = np.array([0, 1] * (len(distances) // 2))
    model = build_pipeline(LogisticRegression(max_iter=1000, random_state=11))
    model.fit(X_train, y_train)
    numeric = model.named_steps["preprocessor"].named_transformers_["numeric"]
    imputer = numeric.named_steps["imputer"]
    columns = list(imputer.feature_names_in_)
    distance_index = columns.index("distance_km")
    expected_median = np.nanmedian(distances)

    assert imputer.statistics_[distance_index] == pytest.approx(expected_median)
    imputed = imputer.transform(X_train[columns])
    missing_rows = X_train["distance_km"].isna().to_numpy()
    np.testing.assert_allclose(imputed[missing_rows, distance_index], expected_median)


@pytest.mark.parametrize("missing_count", [1, 3], ids=["one-missing", "all-missing"])
def test_pipeline_handles_missing_distance(fitted_pipeline, missing_count):
    model, X_train = fitted_pipeline
    sample = X_train.head(3).copy()
    sample.loc[sample.index[:missing_count], "distance_km"] = np.nan
    numeric = model.named_steps["preprocessor"].named_transformers_["numeric"]
    imputer = numeric.named_steps["imputer"]
    columns = list(imputer.feature_names_in_)
    distance_index = columns.index("distance_km")
    expected_median = X_train["distance_km"].median()

    imputed = imputer.transform(sample[columns])
    missing_rows = sample["distance_km"].isna().to_numpy()
    np.testing.assert_allclose(imputed[missing_rows, distance_index], expected_median)

    filled_sample = sample.fillna({"distance_km": expected_median})
    np.testing.assert_allclose(
        numeric.transform(sample[columns]), numeric.transform(filled_sample[columns])
    )
    probabilities = model.predict_proba(sample)
    assert_valid_probabilities(probabilities, len(sample))
    np.testing.assert_allclose(probabilities, model.predict_proba(filled_sample))


def test_categorical_preprocessor_ignores_unknown_categories(fitted_pipeline):
    model, _ = fitted_pipeline
    encoder = model.named_steps["preprocessor"].named_transformers_["categorical"]

    assert isinstance(encoder, OneHotEncoder)
    assert encoder.handle_unknown == "ignore"


@pytest.mark.parametrize("field", ["weekday", "booking", "insurance", "treatment"])
def test_pipeline_predicts_with_unseen_category(fitted_pipeline, field):
    """Exercise the raw pipeline; Appointment separately rejects invalid choices."""
    model, X_train = fitted_pipeline
    sample = X_train.head(3).copy()
    unknown = "unseen_category"
    assert unknown not in set(X_train[field])
    sample[field] = unknown
    encoder = model.named_steps["preprocessor"].named_transformers_["categorical"]
    columns = list(encoder.feature_names_in_)
    field_index = columns.index(field)
    start = sum(len(categories) for categories in encoder.categories_[:field_index])
    stop = start + len(encoder.categories_[field_index])
    encoded = encoder.transform(sample[columns])

    # An unseen value contributes no active indicator in its feature's block.
    assert encoded[:, start:stop].sum() == 0
    assert_valid_probabilities(model.predict_proba(sample), len(sample))
