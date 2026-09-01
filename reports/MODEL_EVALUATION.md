# Model Evaluation Report

## Ride Outcome Model

- Accuracy: **69.33%**
- AUC: **0.5924**
- Best params: `{'classifier__max_depth': None, 'classifier__n_estimators': 300}`
- Confusion matrix (Cancelled, Completed, Incomplete): `[[3, 282, 0], [2, 829, 0], [0, 84, 0]]`
- Benchmark (>= 85% accuracy): **BELOW**

## Fare Prediction Model

- MAE: **2.24**
- RMSE: **9.21** (5.8% of mean fare)
- R²: **0.9915**
- Best params: `{'regressor__learning_rate': 0.1, 'regressor__max_depth': 4, 'regressor__n_estimators': 400}`
- Benchmark (RMSE within 10% of mean fare): **MEETS**

## Customer Cancellation Risk Model

- Accuracy: **83.42%**
- AUC: **0.6141**
- Best params: `{'classifier__max_depth': None, 'classifier__n_estimators': 150}`
- Confusion matrix (No Cancel, Cancel): `[[1001, 0], [199, 0]]`
- Benchmark (>= 85% accuracy): **BELOW**

## Driver Delay Prediction Model

- Accuracy: **92.75%**
- AUC: **0.8329**
- Best params: `{'classifier__max_depth': None, 'classifier__n_estimators': 300}`
- Confusion matrix (On Time, Delayed): `[[1111, 2], [85, 2]]`
- Benchmark (>= 85% accuracy): **MEETS**
