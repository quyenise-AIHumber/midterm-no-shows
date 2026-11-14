---
title: Maple Physio No-Show Check
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12
app_file: app.py
---

# Maple Physio no-show check

## What it does

This synthetic-data demonstration addresses missed appointments by predicting no-show and attendance probabilities and recommending confirmation calls. A call costs $8; an uncalled no-show costs $35.

Public Hugging Face Space: [Maple Physio No-Show Check](https://huggingface.co/spaces/quyenise/midterm-no-shows).

## How to run it locally

Create and activate the Conda environment:

```powershell
conda env create -f environment.yml
conda activate midterm-no-shows
```

Reproduce the training and evaluation results:

```powershell
python -m noshows.train
```

Launch the Gradio application:

```powershell
gradio app.py
```

## Team

Team members and individual responsibilities will be confirmed and updated before final submission.

## Decisions

The dataset contains 4,000 synthetic appointments, 13 columns and binary target `no_show` (1 = missed, 0 = attended). With `SEED = 11` defined in `noshows/pipeline.py`, a stratified 75/25 split produces 3,000 training and 1,000 test appointments. All ten predictors are available at booking time; accepted inputs and selection reasons follow.

| Feature | Type | Accepted input | Selection rationale |
|---|---|---|---|
| `lead_days` | Numeric | 0 to 120 | Advance booking may affect attendance. |
| `age` | Numeric | 12 to 110 | Captures possible age-related differences. |
| `prior_no_shows` | Numeric | 0 to 30 | Provides attendance history. |
| `hour` | Numeric | 8 to 17 | Represents scheduling differences within the day. |
| `first_visit` | Numeric | 0 or 1 | Distinguishes first visits from returning appointments. |
| `distance_km` | Numeric | 0.1 to 150, or empty | Represents potential travel burden. |
| `weekday` | Categorical | `mon`, `tue`, `wed`, `thu`, `fri` | Captures weekday differences. |
| `booking` | Categorical | `online`, `phone`, `front_desk` | Represents the booking channel. |
| `insurance` | Categorical | `private`, `provincial`, `self_pay` | Captures payment-plan differences in the synthetic generator. |
| `treatment` | Categorical | `physio`, `massage`, `acupuncture` | Allows patterns to vary by treatment type. |

Excluded columns are `appointment_id` (identifier), `fee_charged` (post-visit information causing leakage) and `no_show` (the prediction target).

Input limits prevent unrealistic values while allowing reasonable variation beyond the training data. Only `distance_km` may be missing because some distance values are not recorded in the dataset.

Numeric preprocessing uses `SimpleImputer(strategy="median")` and `StandardScaler`. Each CV fold learns preprocessing from its training portion; the final pipeline uses training data only. Empty `distance_km` is replaced with the learned median and triggers a warning. Other missing inputs, nonfinite numbers, booleans and out-of-range values are rejected. `first_visit` must be 0 or 1; other numeric inputs need not be integers.

`OneHotEncoder(handle_unknown="ignore")` tolerates unseen categories within the raw pipeline. Separately, `Appointment` rejects categories outside the exact, case-sensitive choices to catch input errors. The held-out test set is excluded from preprocessing fitting, model selection, tuning and threshold selection.

## Model comparison

Five-fold stratified cross-validation on training data produced these means +/- fold standard deviations (`reports/cv_results.csv`). Classification metrics use default class predictions, before calling-threshold optimization; precision, recall and F1 concern no-shows.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy (prior) | 0.7383 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.5000 +/- 0.0000 |
| Logistic Regression | 0.7580 +/- 0.0175 | 0.5918 +/- 0.0781 | 0.2293 +/- 0.0498 | 0.3299 +/- 0.0639 | 0.7273 +/- 0.0117 |
| Balanced Decision Tree | 0.6703 +/- 0.0198 | 0.4022 +/- 0.0227 | 0.5236 +/- 0.0333 | 0.4539 +/- 0.0169 | 0.6499 +/- 0.0117 |
| Random Forest | 0.7493 +/- 0.0129 | 0.5769 +/- 0.0926 | 0.1389 +/- 0.0421 | 0.2229 +/- 0.0618 | 0.7103 +/- 0.0101 |

Accuracy alone can be misleading with imbalanced classes: the Dummy model achieves 0.7383 accuracy while detecting no no-shows (recall 0). ROC-AUC and no-show precision, recall and F1 therefore complement accuracy, while business cost guides model selection.

Training out-of-fold (OOF) probabilities gave the following cost comparison (`reports/model_cost_comparison.csv`):

| Untuned candidate | Cost-selected threshold | OOF cost (3,000 appointments) |
|---|---:|---:|
| Logistic Regression | 0.25 | $18,829 |
| Balanced Decision Tree | 0.50 | $21,298 |
| Random Forest | 0.25 | $19,453 |

Logistic Regression was selected for the lowest OOF cost and highest mean ROC-AUC. **0.25 is its untuned comparison threshold.** `GridSearchCV(scoring="roc_auc")` searched `C = [0.1, 1.0, 10.0]` and `class_weight = [None, "balanced"]`; the best settings were **`C=0.1`, `class_weight="balanced"`**, with `max_iter=2000` retained.

| Tuning measure | Saved result |
|---|---:|
| Untuned mean CV ROC-AUC | 0.7273 |
| Tuned mean honest-feature CV ROC-AUC | 0.7274518 |
| Approximate improvement | 0.00015 |
| Untuned fold ROC-AUC standard deviation | 0.0117 |

The tuned mean comes from the honest-feature CV diagnostic using the selected estimator and the same folds. Because the untuned mean is rounded, the gain is approximate. It is much smaller than fold SD 0.0117, providing little evidence of meaningful improvement; this is not a significance test.

After tuning, training OOF probabilities were used to minimize cost across thresholds 0.05 to 0.95 in steps of 0.05. The minimum was **$18,921 at 0.45** (`reports/threshold_costs.csv`), $92 above the untuned cost. ROC-AUC tuning therefore did not improve OOF cost. **The final saved threshold is 0.45**, distinct from 0.25 above.

## Result

Final evaluation used 1,000 held-out synthetic appointments after model and threshold selection. `models/model_info.json` and `reports/metrics.json` agree: recommend a call when no-show probability is **at least 0.45**.

| Metric | Final test result |
|---|---:|
| Accuracy | 0.6290 |
| Precision (no-show) | 0.3876 |
| Recall (no-show) | 0.7176 |
| F1 (no-show) | 0.5033 |
| ROC-AUC | 0.7283 |

**Total Cost = $8 x Number of Calls + $35 x Uncalled No-shows.**

| Calling policy | Total cost |
|---|---:|
| Call nobody | $9,170 |
| Call everybody | $8,000 |
| Model recommendation | **$6,470** |

The model achieved a **$2,700 (29.4%) cost reduction** compared with calling nobody and a **$1,530 (19.1%) reduction** compared with calling everybody.

The model recommends 485 calls and leaves 74 no-shows uncalled: `485 x $8 + 74 x $35 = $6,470`. These savings depend on one sample's size and composition; different samples may change costs or policy rankings. No confidence interval was calculated. Fixed costs and perfect call effectiveness are assumed: called patients incur no no-show penalty, effectively assuming they attend. Actual calls may not reach or influence patients, so these are not guaranteed real-clinic savings.

### Data leakage assessment

A training-only CV diagnostic compared booking-time features with post-appointment `fee_charged` added:

| Feature set | Mean training CV ROC-AUC |
|---|---:|
| Honest features | 0.7274518 |
| Honest features plus `fee_charged` | 0.9231585 |

The inflated score uses future information unavailable at booking time. `fee_charged` remains excluded from the final model; these diagnostic scores are not held-out test results.

## Tests

**59 pytest cases passed**, counting parametrized cases rather than test functions. Coverage includes data reproducibility and feature lists; stratified splitting and row alignment; missing-value and categorical preprocessing; hand-worked costs and cheapest calling policies; threshold selection; `Appointment` input validation; model predictions; and saved artifacts and deployment configuration. Ruff formatting and lint checks also passed.

```powershell
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Limitations

Synthetic attendance patterns may differ from real patients, and predictive associations do not establish causes of no-shows. Call effectiveness and savings rely on the assumptions described above. Median replacement loses individual distance information, and valid inputs outside the generator's observed ranges may be unfamiliar to the model. Training CV/OOF estimates may be optimistic because selection reuses the same folds. Before deployment, real-data and call-effectiveness validation, fairness assessment (including age and insurance), privacy safeguards, calibration checks and ongoing monitoring are needed.
