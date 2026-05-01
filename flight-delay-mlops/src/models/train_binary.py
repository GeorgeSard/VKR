"""Train Stage 1 binary delay classifier."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src.config import load_params
from src.models.common import (
    binary_metrics,
    build_classifier,
    load_processed_split,
    save_model,
    split_xy,
    write_json,
)
from src.models.tracking import log_training_run, mlflow_run


def train_binary(
    *,
    params: dict[str, Any],
    processed_dir: Path,
    model_path: Path,
    metrics_path: Path,
) -> None:
    """Fit and validate the binary delay classifier."""
    cfg = params["train_binary"]
    target = cfg["target"]

    train = load_processed_split(processed_dir, "train")
    val = load_processed_split(processed_dir, "val")
    X_train, y_train = split_xy(train, target)
    X_val, y_val = split_xy(val, target)

    model = build_classifier(
        model_name=cfg["model"],
        task="binary",
        hyperparams=cfg.get("hyperparams", {}),
        seed=int(params["seed"]),
        class_weight=cfg.get("class_weight"),
    )

    started = time.perf_counter()
    model.fit(X_train, y_train.astype(int))
    train_time = time.perf_counter() - started

    metrics = {
        "task": "binary_delay",
        "model": cfg["model"],
        "target": target,
        "feature_set": params["features"]["feature_set"],
        "train_rows": len(train),
        "val_rows": len(val),
        "train_time_seconds": float(train_time),
        "val": binary_metrics(model, X_val, y_val),
    }
    save_model(model, model_path)
    write_json(metrics, metrics_path)
    with mlflow_run(
        params=params,
        task_name="binary_delay",
        experiment_name=params["mlflow"]["experiment_binary"],
        run_name=f"binary_{cfg['model']}_{params['features']['feature_set']}",
    ):
        log_training_run(
            params=params,
            task_config=cfg,
            metrics=metrics,
            metrics_path=metrics_path,
            model=model,
            model_artifact_name="binary_model",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 binary delay classifier.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--model-out", default="models/binary_model.joblib")
    parser.add_argument("--metrics-out", default="reports/metrics/binary_val_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = load_params(args.params)
    train_binary(
        params=params,
        processed_dir=Path(args.processed_dir or params["data"]["processed_dir"]),
        model_path=Path(args.model_out),
        metrics_path=Path(args.metrics_out),
    )


if __name__ == "__main__":
    main()
