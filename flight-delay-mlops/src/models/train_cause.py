"""Train Stage 2 multiclass delay-cause classifier."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src.config import load_params
from src.data.split import filter_delayed_only
from src.models.common import (
    build_classifier,
    load_processed_split,
    multiclass_metrics,
    save_model,
    split_xy,
    write_json,
)


def train_cause(
    *,
    params: dict[str, Any],
    processed_dir: Path,
    model_path: Path,
    metrics_path: Path,
) -> None:
    """Fit and validate the delayed-only multiclass cause classifier."""
    cfg = params["train_cause"]
    target = cfg["target"]

    train = filter_delayed_only(load_processed_split(processed_dir, "train"))
    val = filter_delayed_only(load_processed_split(processed_dir, "val"))
    X_train, y_train = split_xy(train, target)
    X_val, y_val = split_xy(val, target)

    model = build_classifier(
        model_name=cfg["model"],
        task="cause",
        hyperparams=cfg.get("hyperparams", {}),
        seed=int(params["seed"]),
        class_weight=cfg.get("class_weight"),
    )

    started = time.perf_counter()
    model.fit(X_train, y_train.astype(str))
    train_time = time.perf_counter() - started

    metrics = {
        "task": "delay_cause",
        "model": cfg["model"],
        "target": target,
        "feature_set": params["features"]["feature_set"],
        "train_rows": len(train),
        "val_rows": len(val),
        "train_time_seconds": float(train_time),
        "val": multiclass_metrics(model, X_val, y_val),
    }
    save_model(model, model_path)
    write_json(metrics, metrics_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 delayed-only cause classifier.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--model-out", default="models/cause_model.joblib")
    parser.add_argument("--metrics-out", default="reports/metrics/cause_val_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = load_params(args.params)
    train_cause(
        params=params,
        processed_dir=Path(args.processed_dir or params["data"]["processed_dir"]),
        model_path=Path(args.model_out),
        metrics_path=Path(args.metrics_out),
    )


if __name__ == "__main__":
    main()
