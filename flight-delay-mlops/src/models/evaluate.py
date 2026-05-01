"""Evaluate trained Stage 1 and Stage 2 models on the held-out 2025 test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config import load_params
from src.data.split import filter_delayed_only
from src.models.common import (
    binary_metrics,
    load_model,
    load_processed_split,
    multiclass_metrics,
    split_xy,
    write_json,
)


def evaluate_models(
    *,
    params: dict[str, Any],
    processed_dir: Path,
    binary_model_path: Path,
    cause_model_path: Path,
    metrics_path: Path,
) -> None:
    """Evaluate binary and delayed-only cause models on the test period."""
    test = load_processed_split(processed_dir, "test")
    binary_model = load_model(binary_model_path)
    cause_model = load_model(cause_model_path)

    X_test_binary, y_test_binary = split_xy(test, params["train_binary"]["target"])
    delayed_test = filter_delayed_only(test)
    X_test_cause, y_test_cause = split_xy(delayed_test, params["train_cause"]["target"])

    payload: dict[str, Any] = {
        "test_period": params["split"]["test_years"],
        "feature_set": params["features"]["feature_set"],
        "binary": {
            "model": params["train_binary"]["model"],
            "rows": len(test),
            "metrics": binary_metrics(binary_model, X_test_binary, y_test_binary),
        },
        "cause": {
            "model": params["train_cause"]["model"],
            "rows": len(delayed_test),
            "metrics": multiclass_metrics(cause_model, X_test_cause, y_test_cause),
        },
    }
    write_json(payload, metrics_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models on test split.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--binary-model", default="models/binary_model.joblib")
    parser.add_argument("--cause-model", default="models/cause_model.joblib")
    parser.add_argument("--metrics-out", default="reports/metrics/test_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = load_params(args.params)
    evaluate_models(
        params=params,
        processed_dir=Path(args.processed_dir or params["data"]["processed_dir"]),
        binary_model_path=Path(args.binary_model),
        cause_model_path=Path(args.cause_model),
        metrics_path=Path(args.metrics_out),
    )


if __name__ == "__main__":
    main()
