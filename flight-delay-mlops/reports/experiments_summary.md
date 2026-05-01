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

| feature_set | model | f1 | roc_auc | pr_auc | accuracy |
| --- | --- | --- | --- | --- | --- |
| BASELINE | logreg | 0.6018 | 0.8229 | 0.6995 | 0.7801 |
| EXTENDED | logreg | 0.6022 | 0.8234 | 0.6999 | 0.7802 |
| WITH_NETWORK | logreg | 0.6062 | 0.8246 | 0.7015 | 0.7871 |

### Cause Classifier

| feature_set | model | macro_f1 | weighted_f1 | accuracy |
| --- | --- | --- | --- | --- |
| BASELINE | logreg | 0.6126 | 0.7400 | 0.7343 |
| EXTENDED | logreg | 0.6110 | 0.7394 | 0.7333 |
| WITH_NETWORK | logreg | 0.5987 | 0.7240 | 0.7113 |

## Experiment 5.2 - Model Comparison

Feature set fixed to `EXTENDED` for this baseline comparison.

### Binary Delay Classifier

| model | feature_set | f1 | roc_auc | pr_auc | accuracy |
| --- | --- | --- | --- | --- | --- |
| logreg | EXTENDED | 0.6022 | 0.8234 | 0.6999 | 0.7802 |
| random_forest | EXTENDED | 0.6190 | 0.8295 | 0.6995 | 0.7811 |

### Cause Classifier

| model | feature_set | macro_f1 | weighted_f1 | accuracy |
| --- | --- | --- | --- | --- |
| logreg | EXTENDED | 0.6110 | 0.7394 | 0.7333 |
| random_forest | EXTENDED | 0.6134 | 0.7545 | 0.7593 |

## Current Best

- Binary F1 by feature set: `logreg` + `WITH_NETWORK` = 0.6062.
- Cause macro-F1 by feature set: `logreg` + `BASELINE` = 0.6126.
- Binary F1 by model: `random_forest` + `EXTENDED` = 0.6190.
- Cause macro-F1 by model: `random_forest` + `EXTENDED` = 0.6134.

## Next Experiments

- Add CatBoost, LightGBM, and XGBoost when the Python environment has compatible wheels.
- Run Optuna tuning for top models.
- Run imbalance and one-stage-vs-two-stage cause-classification experiments.
- Add SHAP and concept-drift reports.
