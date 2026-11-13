"""Compare, tune, evaluate once, and save the no-show model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from sklearn.tree import DecisionTreeClassifier

from .data import make_appointments
from .evaluation import choose_threshold, total_cost
from .pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SEED,
    build_pipeline,
    split_appointments,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    dataset = make_appointments(seed=SEED)
    X_train, X_test, y_train, y_test = split_appointments(dataset)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    candidates = {
        "Dummy": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "DecisionTreeBalanced": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=80, min_samples_leaf=5, random_state=SEED
        ),
    }
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    comparison = []
    for name, candidate in candidates.items():
        scores = cross_validate(
            build_pipeline(clone(candidate)), X_train, y_train, cv=cv, scoring=scoring
        )
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(values.mean()), 4)
            row[f"{metric}_std"] = round(float(values.std()), 4)
        comparison.append(row)
    cv_report = pd.DataFrame(comparison)
    cv_report.to_csv(ROOT / "reports/cv_results.csv", index=False)
    print("\nFIVE-FOLD TRAINING CV\n", cv_report.to_string(index=False))

    cost_rows = []
    for name, candidate in candidates.items():
        if name == "Dummy":
            continue
        probabilities = cross_val_predict(
            build_pipeline(clone(candidate)),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        result, _ = choose_threshold(y_train, probabilities)
        cost_rows.append({"model": name, **result})
    cost_report = pd.DataFrame(cost_rows)
    cost_report.to_csv(ROOT / "reports/model_cost_comparison.csv", index=False)
    selected_name = min(cost_rows, key=lambda row: row["cost"])["model"]
    print("\nTRAINING OOF COSTS\n", cost_report.to_string(index=False))
    print("Selected model:", selected_name)

    grids = {
        "LogisticRegression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        "DecisionTreeBalanced": {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [1, 10],
        },
        "RandomForest": {
            "model__max_depth": [5, 10, None],
            "model__min_samples_leaf": [1, 5],
        },
    }
    search = GridSearchCV(
        build_pipeline(clone(candidates[selected_name])),
        param_grid=grids[selected_name],
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    untuned = cv_report.set_index("model").loc[selected_name]
    print("\nBest settings:", search.best_params_)
    print("Untuned ROC-AUC:", untuned["roc_auc_mean"])
    print("Tuned ROC-AUC:", round(search.best_score_, 4))
    print(
        "Tuning gain exceeds fold SD:",
        search.best_score_ - untuned["roc_auc_mean"] > untuned["roc_auc_std"],
    )

    probabilities = cross_val_predict(
        search.best_estimator_, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    best, threshold_rows = choose_threshold(y_train, probabilities)
    pd.DataFrame(threshold_rows).to_csv(
        ROOT / "reports/threshold_costs.csv", index=False
    )
    threshold = best["threshold"]
    print("\nBest training threshold:", threshold, "OOF cost:", best["cost"])

    # Diagnostic only: future-known fee_charged makes performance look better.
    tuned_estimator = clone(search.best_estimator_.named_steps["model"])
    honest_auc = cross_val_score(
        build_pipeline(clone(tuned_estimator)),
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
    ).mean()
    X_leaky = X_train.assign(fee_charged=dataset.loc[X_train.index, "fee_charged"])
    leaky_auc = cross_val_score(
        build_pipeline(
            clone(tuned_estimator),
            numeric_features=NUMERIC_FEATURES + ["fee_charged"],
        ),
        X_leaky,
        y_train,
        cv=cv,
        scoring="roc_auc",
    ).mean()
    print("\nHonest training CV ROC-AUC:", round(honest_auc, 4))
    print("Leaky training CV ROC-AUC:", round(leaky_auc, 4))

    # FINAL TEST: evaluate once, AFTER all model and threshold choices.
    model = search.best_estimator_
    test_proba = model.predict_proba(X_test)[:, 1]
    test_flags = test_proba >= threshold
    actual = np.asarray(y_test)
    metrics = {
        "test_samples": len(y_test),
        "threshold": threshold,
        "accuracy": float(accuracy_score(actual, test_flags)),
        "precision": float(precision_score(actual, test_flags, zero_division=0)),
        "recall": float(recall_score(actual, test_flags, zero_division=0)),
        "f1": float(f1_score(actual, test_flags, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, test_proba)),
        "confusion_matrix": confusion_matrix(actual, test_flags).tolist(),
        "costs_usd": {
            "call_nobody": total_cost(actual, np.zeros(len(actual), dtype=bool)),
            "call_everybody": total_cost(actual, np.ones(len(actual), dtype=bool)),
            "model": total_cost(actual, test_flags),
        },
        "training_leakage_diagnostic": {
            "honest_cv_roc_auc": float(honest_auc),
            "leaky_cv_roc_auc": float(leaky_auc),
        },
    }
    info = {
        "model_name": selected_name,
        "scikit_learn_version": sklearn.__version__,
        "seed": SEED,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_settings": search.best_params_,
        "test_results": metrics,
    }
    joblib.dump(model, ROOT / "models/no_show.joblib")
    (ROOT / "models/model_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    (ROOT / "reports/metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("\nFINAL HELD-OUT TEST\n", json.dumps(metrics, indent=2))
    print("\nSaved model, metadata and reports.")


if __name__ == "__main__":
    main()
