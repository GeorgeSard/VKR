"""MLflow tracking helpers for training stages."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import yaml


@contextmanager
def mlflow_run(
    *,
    params: dict[str, Any],
    task_name: str,
    experiment_name: str,
    run_name: str,
) -> Iterator[Any | None]:
    """Start an MLflow run if tracking is enabled in params.yaml.

    ``MLFLOW_TRACKING_URI`` overrides ``params.yaml``. This lets the same DVC
    stage log to local ``mlruns/`` by default and to the Docker tracking server
    when the user runs it with ``MLFLOW_TRACKING_URI=http://localhost:5000``.
    """
    mlflow_cfg = params.get("mlflow", {})
    if not bool(mlflow_cfg.get("enabled", True)):
        with nullcontext(None) as run:
            yield run
        return

    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or str(
        mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(_reproducibility_tags(params=params, task_name=task_name))
        yield run


def log_training_run(
    *,
    params: dict[str, Any],
    task_config: dict[str, Any],
    metrics: dict[str, Any],
    metrics_path: Path,
    model: Any,
    model_artifact_name: str = "model",
) -> None:
    """Log params, scalar metrics, metric artifact, and sklearn model."""
    import mlflow
    import mlflow.sklearn

    mlflow.log_params(_flatten_params(task_config))
    mlflow.log_params(
        {
            "feature_set": params["features"]["feature_set"],
            "random_seed": params["seed"],
        }
    )
    mlflow.log_metrics(_flatten_metrics(metrics))
    mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
    mlflow.sklearn.log_model(model, name=model_artifact_name)


def _reproducibility_tags(*, params: dict[str, Any], task_name: str) -> dict[str, str]:
    return {
        "task": task_name,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain"))).lower(),
        "dvc_data_version": _dvc_raw_md5(),
        "feature_set_name": str(params["features"]["feature_set"]),
        "train_period": ",".join(str(y) for y in params["split"]["train_years"]),
        "val_period": ",".join(str(y) for y in params["split"]["val_years"]),
        "test_period": ",".join(str(y) for y in params["split"]["test_years"]),
        "random_seed": str(params["seed"]),
    }


def _flatten_params(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_params(value, name))
        elif isinstance(value, list | tuple):
            flat[name] = ",".join(str(item) for item in value)
        else:
            flat[name] = value
    return flat


def _flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, name))
        elif isinstance(value, bool | int | float):
            flat[name] = float(value)
    return flat


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _dvc_raw_md5(lock_path: Path | None = None) -> str:
    path = lock_path or Path(__file__).resolve().parents[2] / "dvc.lock"
    if not path.exists():
        return "unknown"

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for dep in data.get("stages", {}).get("load", {}).get("deps", []):
        if dep.get("path") == "data/raw/flight_delays_ru.parquet":
            return str(dep.get("md5", "unknown"))
    return "unknown"
