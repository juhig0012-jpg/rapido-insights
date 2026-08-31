# Model Evaluation Report

## Ride Outcome Model

- Accuracy: **65.75%**
- AUC: **0.6004**
- Best params: `{'classifier__max_depth': None, 'classifier__n_estimators': 150}`
- Confusion matrix (Cancelled, Completed, Incomplete): `[[81, 204, 0], [122, 707, 2], [14, 69, 1]]`
- Benchmark (>= 85% accuracy): **BELOW**

## Fare Prediction Model

- MAE: **2.24**
- RMSE: **9.21** (5.8% of mean fare)
- R²: **0.9915**
- Best params: `{'regressor__learning_rate': 0.1, 'regressor__max_depth': 4, 'regressor__n_estimators': 400}`
- Benchmark (RMSE within 10% of mean fare): **MEETS**

## Customer Cancellation Risk Model

- Accuracy: **82.50%**
- AUC: **0.6100**
- Best params: `{'classifier__max_depth': None, 'classifier__n_estimators': 300}`
- Confusion matrix (No Cancel, Cancel): `[[980, 21], [189, 10]]`
- Benchmark (>= 85% accuracy): **BELOW**

## Driver Delay Prediction Model

- Accuracy: **92.58%**
- AUC: **0.8334**
- Best params: `{'classifier__max_depth': None, 'classifier__n_estimators': 300}`
- Confusion matrix (On Time, Delayed): `[[1095, 18], [71, 16]]`
- Benchmark (>= 85% accuracy): **MEETS**
