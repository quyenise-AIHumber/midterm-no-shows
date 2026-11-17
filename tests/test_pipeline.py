"""Tests for data preparation and the ML pipeline."""

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from noshows.data import make_appointments
from noshows.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    split_appointments,
)


@pytest.mark.parametrize("seed", [11, 29])
def test_dataset_is_reproducible(seed):
    a = make_appointments(seed=seed)
    b = make_appointments(seed=seed)

    assert a.equals(b)


def test_default_dataset_has_expected_schema_and_binary_target():
    frame = make_appointments()
    expected_columns = {
        "appointment_id",
        "lead_days",
        "age",
        "prior_no_shows",
        "hour",
        "weekday",
        "booking",
        "insurance",
        "treatment",
        "first_visit",
        "distance_km",
        "fee_charged",
        "no_show",
    }

    assert frame.shape == (4000, 13)
    assert frame.columns.is_unique
    assert set(frame.columns) == expected_columns
    assert frame["no_show"].isin([0, 1]).all()


def test_feature_lists_have_exact_expected_composition():
    expected_numeric = [
        "lead_days",
        "age",
        "prior_no_shows",
        "hour",
        "first_visit",
        "distance_km",
    ]
    expected_categorical = ["weekday", "booking", "insurance", "treatment"]

    assert NUMERIC_FEATURES == expected_numeric
    assert CATEGORICAL_FEATURES == expected_categorical
    assert FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES


@pytest.mark.parametrize(
    "excluded",
    ["appointment_id", "fee_charged", "no_show"],
)
def test_excluded_columns(excluded):
    for feature_list in (NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURES):
        assert excluded not in feature_list


@pytest.mark.parametrize("dataset_seed", [11, 29, 73])
def test_stratified_split(dataset_seed):
    """Verify split sizes, class allocation, row identity and label alignment."""
    frame = make_appointments(seed=dataset_seed)

    X_train, X_test, y_train, y_test = split_appointments(frame)

    assert len(X_train) == 3000
    assert len(X_test) == 1000

    assert len(y_train) == 3000
    assert len(y_test) == 1000

    assert len(X_train) + len(X_test) == len(frame) == 4000
    assert X_train.index.is_unique
    assert X_test.index.is_unique
    assert X_train.index.intersection(X_test.index).empty
    combined_indices = X_train.index.append(X_test.index)
    assert combined_indices.is_unique
    assert combined_indices.sort_values().equals(frame.index.sort_values())

    full_counts = frame["no_show"].value_counts().sort_index()
    full_no_show_rate = frame["no_show"].mean()
    train_no_show_rate = y_train.mean()
    test_no_show_rate = y_test.mean()

    # Integer allocation may round each expected class count up or down.
    for X_partition, y_partition, no_show_rate in (
        (X_train, y_train, train_no_show_rate),
        (X_test, y_test, test_no_show_rate),
    ):
        assert len(X_partition) == len(y_partition)
        assert X_partition.index.equals(y_partition.index)
        assert X_partition.equals(frame.loc[X_partition.index, FEATURES])
        assert y_partition.equals(frame.loc[y_partition.index, "no_show"])

        expected_counts = full_counts * (len(y_partition) / len(frame))
        actual_counts = y_partition.value_counts().reindex(
            full_counts.index, fill_value=0
        )
        assert (abs(actual_counts - expected_counts) < 1).all()
        assert no_show_rate == pytest.approx(
            full_no_show_rate, abs=1 / len(y_partition)
        )

    combined_counts = (
        y_train.value_counts().add(y_test.value_counts(), fill_value=0).sort_index()
    )
    np.testing.assert_array_equal(combined_counts.to_numpy(), full_counts.to_numpy())


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
