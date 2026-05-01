# Experiments Summary

## Scope

Baseline slice of stage 5: feature-set impact and sklearn model comparison.
All runs use time-based validation: train=2023, val=2024. No `gt_*` columns are used as features.

Reproduce:

```bash
make experiments-baseline
```

## Experiment 5.1 - Feature Sets

### Binary Delay Classifier

| feature_set | model | status | f1 | roc_auc | pr_auc | accuracy | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BASELINE | logreg | completed | 0.6018 | 0.8229 | 0.6995 | 0.7801 |  |
| EXTENDED | logreg | completed | 0.6022 | 0.8234 | 0.6999 | 0.7802 |  |
| WITH_NETWORK | logreg | completed | 0.6062 | 0.8246 | 0.7015 | 0.7871 |  |

### Cause Classifier

| feature_set | model | status | macro_f1 | weighted_f1 | accuracy | skip_reason |
| --- | --- | --- | --- | --- | --- | --- |
| BASELINE | logreg | completed | 0.6126 | 0.7400 | 0.7343 |  |
| EXTENDED | logreg | completed | 0.6110 | 0.7394 | 0.7333 |  |
| WITH_NETWORK | logreg | completed | 0.5987 | 0.7240 | 0.7113 |  |

## Experiment 5.2 - Model Comparison

Feature set fixed to `EXTENDED` for this baseline comparison.

### Binary Delay Classifier

| model | feature_set | status | f1 | roc_auc | pr_auc | accuracy | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logreg | EXTENDED | completed | 0.6022 | 0.8234 | 0.6999 | 0.7802 |  |
| random_forest | EXTENDED | completed | 0.6190 | 0.8295 | 0.6995 | 0.7811 |  |
| catboost | EXTENDED | completed | 0.6202 | 0.8375 | 0.7223 | 0.7787 |  |
| xgboost | EXTENDED | skipped |  |  |  |  | XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded. Likely causes: * OpenMP runtime is not installed - vcomp140.dll or libgomp-1.dll for Windows - libomp.dylib for Mac OSX - libgomp.so for Linux and ... |
| lightgbm | EXTENDED | skipped |  |  |  |  | OSError: dlopen(/Users/georgij/Projects/ВКР/flight-delay-mlops/.venv/lib/python3.13/site-packages/lightgbm/lib/lib_lightgbm.dylib, 0x0006): Library not loaded: @rpath/libomp.dylib Referenced from: <D44045CD-B874-3A27-... |

### Cause Classifier

| model | feature_set | status | macro_f1 | weighted_f1 | accuracy | skip_reason |
| --- | --- | --- | --- | --- | --- | --- |
| logreg | EXTENDED | completed | 0.6110 | 0.7394 | 0.7333 |  |
| random_forest | EXTENDED | completed | 0.6134 | 0.7545 | 0.7593 |  |
| catboost | EXTENDED | completed | 0.6075 | 0.7279 | 0.7235 |  |
| xgboost | EXTENDED | skipped |  |  |  | XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded. Likely causes: * OpenMP runtime is not installed - vcomp140.dll or libgomp-1.dll for Windows - libomp.dylib for Mac OSX - libgomp.so for Linux and ... |
| lightgbm | EXTENDED | skipped |  |  |  | OSError: dlopen(/Users/georgij/Projects/ВКР/flight-delay-mlops/.venv/lib/python3.13/site-packages/lightgbm/lib/lib_lightgbm.dylib, 0x0006): Library not loaded: @rpath/libomp.dylib Referenced from: <D44045CD-B874-3A27-... |

## Current Best

- Binary F1 by feature set: `logreg` + `WITH_NETWORK` = 0.6062.
- Cause macro-F1 by feature set: `logreg` + `BASELINE` = 0.6126.
- Binary F1 by model: `catboost` + `EXTENDED` = 0.6202.
- Cause macro-F1 by model: `random_forest` + `EXTENDED` = 0.6134.

## Next Experiments

- Install macOS `libomp` to unlock XGBoost and LightGBM runs in this environment.
- Run Optuna tuning for top models.
- Run imbalance and one-stage-vs-two-stage cause-classification experiments.
- Add SHAP and concept-drift reports.
