---
title: Maple Physio No-Show Check
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12
app_file: app.py
---

# Maple Physio No-Show Check

## What it does

Predicts the probability of a patient missing an appointment
and recommends whether the clinic should call the patient.

Calling a patient costs $8.
An uncalled no-show costs $35.

## How to run it locally

1. Create the environment using environment.yml.
2. Activate midterm-no-shows.
3. Run python -m noshows.train.
4. Run python app.py.

## Team

Add each group member's name and responsibilities.

## Decisions

The model excludes appointment_id because it is an identifier.
It excludes fee_charged because it is recorded after the visit.
The target variable is no_show.

## Model comparison

Four models were evaluated using five-fold stratified cross-validation on the training data.

| Model | Mean accuracy | Mean ROC-AUC |
|---|---:|---:|
| Dummy | 0.7383 | 0.5000 |
| Logistic Regression | 0.7580 | 0.7273 |
| Balanced Decision Tree | 0.6703 | 0.6499 |
| Random Forest | 0.7493 | 0.7103 |

Logistic Regression achieved the highest mean ROC-AUC among the four evaluated models and the lowest reported training out-of-fold business cost of $18,829.

The selected calling threshold is **0.25**. This threshold was optimized using training-only predictions, keeping the final test set separate from model selection.

### Data leakage assessment

A diagnostic experiment compared the honest booking-time feature set against a version containing `fee_charged`, which is recorded after the appointment.

- Honest cross-validation ROC-AUC: **0.7275**
- Leaky cross-validation ROC-AUC: **0.9232**

Although the second model reports better discrimination, it relies on information unavailable at prediction time and is therefore unsuitable for deployment.

## Result

The final model policy was evaluated on a held-out test set of 1,000 simulated appointments.

| Calling policy | Total cost |
|---|---:|
| Call nobody | $9,170 |
| Call everybody | $8,000 |
| Model recommendation | **$6,470** |

The model achieved a **$2,700 (29.4%) cost reduction** compared with calling nobody and a **$1,530 (19.1%) reduction** compared with calling everybody.

The analysis uses a cost of $8 per confirmation call and $35 per uncalled no-show. The model applies its saved threshold of 0.25 to recommend which patients should receive a call.

These figures represent simulated financial results, not guaranteed real-world savings.

## Limitations

- The dataset is synthetically generated and may not represent actual patient attendance patterns.
- Model predictions are probabilistic estimates, not guarantees.
- Missing distance values are handled through preprocessing, with a warning in the application.
- Future information, including `fee_charged`, is deliberately excluded to prevent data leakage.
- The cost model assumes fixed calling and no-show costs and does not establish the actual effectiveness of confirmation calls.
- Real-world deployment would require operational validation, probability calibration, monitoring and consideration of patient privacy.

## Tests

The project currently has 24 identified test functions, five parametrization decorators and 37 passing pytest cases. Automated checks cover data preparation, model behaviour, business cost calculations, validation, saved artifacts and deployment configuration.

Run the following commands to verify the project:

```powershell
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
```

## Result

Add final test metrics, the three policy costs,
and the data-leakage comparison.

## Tests

Run:
- python -m ruff format --check .
- python -m ruff check .
- python -m pytest

## Limitations

The dataset is synthetic.
Predictions are estimates, not guarantees.
