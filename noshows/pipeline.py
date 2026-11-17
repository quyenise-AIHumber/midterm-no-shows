"""Feature selection, split and preprocessing."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 11
TARGET = "no_show"
NUMERIC_FEATURES = [
    "lead_days",
    "age",
    "prior_no_shows",
    "hour",
    "first_visit",
    "distance_km",
]
CATEGORICAL_FEATURES = ["weekday", "booking", "insurance", "treatment"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_appointments(frame):
    return train_test_split(
        frame[FEATURES],
        frame[TARGET],
        test_size=0.25,
        random_state=SEED,
        stratify=frame[TARGET],
    )


def build_pipeline(estimator, numeric_features=None):
    nums = NUMERIC_FEATURES if numeric_features is None else numeric_features
    numeric_steps = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_steps, nums),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])
