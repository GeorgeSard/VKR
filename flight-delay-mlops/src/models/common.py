"""Shared sklearn training/evaluation helpers for reproducible baseline runs."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_sets import assert_no_leakage, get_feature_columns


def load_processed_split(processed_dir: Path, split: str) -> pd.DataFrame:
    """Load a processed split produced by ``src.features.build``."""
    return pd.read_parquet(processed_dir / f"{split}.parquet")


def split_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return X/y after re-applying the central no-leakage feature selector."""
    feature_columns = get_feature_columns(df)
    assert_no_leakage(feature_columns)
    return df.loc[:, feature_columns], df[target]


def build_classifier(
    *,
    model_name: str,
    task: str,
    hyperparams: dict[str, Any],
    seed: int,
    class_weight: str | None = None,
) -> Pipeline:
    """Build a sklearn Pipeline with preprocessing and a supported classifier."""
    if model_name == "logreg":
        estimator: Any = LogisticRegression(
            C=float(hyperparams.get("C", 1.0)),
            max_iter=int(hyperparams.get("max_iter", 200)),
            solver=str(hyperparams.get("solver", "liblinear")),
            class_weight=class_weight,
            random_state=seed,
        )
        if task == "cause":
            estimator = OneVsRestClassifier(estimator)
        scale_numeric = True
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(hyperparams.get("n_estimators", 300)),
            max_depth=hyperparams.get("max_depth"),
            min_samples_leaf=int(hyperparams.get("min_samples_leaf", 2)),
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )
        scale_numeric = False
    elif model_name == "catboost":
        catboost = import_module("catboost")
        catboost_params: dict[str, Any] = {
            "iterations": int(hyperparams.get("iterations", 120)),
            "learning_rate": float(hyperparams.get("learning_rate", 0.08)),
            "depth": int(hyperparams.get("depth", 6)),
            "l2_leaf_reg": float(hyperparams.get("l2_leaf_reg", 3.0)),
            "random_seed": seed,
            "allow_writing_files": False,
            "verbose": False,
        }
        if class_weight == "balanced":
            catboost_params["auto_class_weights"] = "Balanced"
        if task == "cause":
            catboost_params["loss_function"] = "MultiClass"
        estimator = catboost.CatBoostClassifier(**catboost_params)
        scale_numeric = False
    elif model_name == "xgboost":
        xgboost = import_module("xgboost")
        xgb_params = {
            "n_estimators": int(hyperparams.get("n_estimators", 120)),
            "max_depth": int(hyperparams.get("max_depth", 6)),
            "learning_rate": float(hyperparams.get("learning_rate", 0.08)),
            "subsample": float(hyperparams.get("subsample", 0.9)),
            "colsample_bytree": float(hyperparams.get("colsample_bytree", 0.9)),
            "tree_method": str(hyperparams.get("tree_method", "hist")),
            "random_state": seed,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        estimator = xgboost.XGBClassifier(**xgb_params)
        if task == "cause":
            estimator = OneVsRestClassifier(estimator)
        scale_numeric = False
    elif model_name == "lightgbm":
        lightgbm = import_module("lightgbm")
        estimator = lightgbm.LGBMClassifier(
            n_estimators=int(hyperparams.get("n_estimators", 120)),
            max_depth=int(hyperparams.get("max_depth", -1)),
            learning_rate=float(hyperparams.get("learning_rate", 0.08)),
            num_leaves=int(hyperparams.get("num_leaves", 31)),
            subsample=float(hyperparams.get("subsample", 0.9)),
            colsample_bytree=float(hyperparams.get("colsample_bytree", 0.9)),
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        scale_numeric = False
    else:
        raise ValueError(
            f"Unsupported model '{model_name}' for {task}. "
            "Supported models: logreg | random_forest | catboost | xgboost | lightgbm."
        )

    return Pipeline(
        steps=[
            ("preprocess", _make_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def _make_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, _numeric_columns),
            ("cat", categorical_pipeline, _categorical_columns),
        ],
        remainder="drop",
    )


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["number", "bool"]).columns)


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(exclude=["number", "bool"]).columns)


def binary_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Compute task A metrics; F1 is the primary selection metric."""
    y_true = y.astype(int)
    y_pred = np.asarray(model.predict(X)).ravel().astype(int)
    probabilities = _positive_probabilities(model, X)

    return {
        "labels": [0, 1],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def multiclass_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Compute task B metrics; macro-F1 is the primary selection metric."""
    y_true = y.astype(str)
    y_pred = pd.Series(np.asarray(model.predict(X)).ravel(), index=y.index).astype(str)
    labels = sorted(set(y_true) | set(y_pred))

    return {
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def model_unavailability_reason(model_name: str) -> str | None:
    """Return a human-readable reason if an optional model backend is unavailable."""
    module_by_model = {
        "catboost": "catboost",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
    }
    module_name = module_by_model.get(model_name)
    if module_name is None:
        return None
    try:
        import_module(module_name)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _positive_probabilities(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    if not hasattr(model, "predict_proba"):
        return pd.Series(model.decision_function(X), index=X.index)

    probabilities = model.predict_proba(X)
    model_step = model.named_steps["model"]
    classes = list(model_step.classes_)
    positive_idx = classes.index(1) if 1 in classes else len(classes) - 1
    return pd.Series(probabilities[:, positive_idx], index=X.index)
