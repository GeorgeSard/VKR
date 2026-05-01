"""Run baseline stage-5 experiments and write a reproducible summary report."""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import load_params
from src.data.split import filter_delayed_only
from src.features.build import build_feature_splits
from src.features.feature_sets import FeatureSet
from src.models.common import (
    binary_metrics,
    build_classifier,
    load_processed_split,
    multiclass_metrics,
    split_xy,
    write_json,
)
from src.models.tracking import log_training_run, mlflow_run

DEFAULT_EXPERIMENT_MODELS: dict[str, dict[str, Any]] = {
    "logreg": {
        "class_weight": "balanced",
        "hyperparams": {
            "C": 1.0,
            "max_iter": 200,
            "solver": "liblinear",
        },
    },
    "random_forest": {
        "class_weight": "balanced",
        "hyperparams": {
            "n_estimators": 120,
            "max_depth": 12,
            "min_samples_leaf": 5,
        },
    },
}


@dataclass(frozen=True)
class ExperimentSpec:
    """A logical experiment row in the summary."""

    group: str
    feature_set: FeatureSet
    model: str
    task: str


def run_baseline_experiments(
    *,
    params: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    summary_path: Path,
    feature_sets: tuple[FeatureSet, ...] = (
        FeatureSet.BASELINE,
        FeatureSet.EXTENDED,
        FeatureSet.WITH_NETWORK,
    ),
    models: tuple[str, ...] = ("logreg", "random_forest"),
) -> list[dict[str, Any]]:
    """Run experiment 5.1 and a fast baseline slice of experiment 5.2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics_dir = output_dir / "runs"
    run_metrics_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_specs(feature_sets=feature_sets, models=models)
    needed_feature_sets = tuple(
        sorted({spec.feature_set for spec in specs}, key=lambda item: item.value)
    )
    records: list[dict[str, Any]] = []
    cache: dict[tuple[FeatureSet, str, str], dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="flight-delay-exp-") as tmp:
        tmp_root = Path(tmp)
        feature_dirs = _build_feature_dirs(
            params=params,
            input_dir=input_dir,
            tmp_root=tmp_root,
            feature_sets=needed_feature_sets,
        )

        for spec in specs:
            key = (spec.feature_set, spec.model, spec.task)
            if key not in cache:
                metrics_path = run_metrics_dir / (
                    f"{spec.task}_{spec.feature_set.value.lower()}_{spec.model}.json"
                )
                cache[key] = _run_one(
                    params=params,
                    processed_dir=feature_dirs[spec.feature_set],
                    spec=spec,
                    metrics_path=metrics_path,
                )
            records.append({"experiment_group": spec.group, **cache[key]})

    _write_outputs(records=records, output_dir=output_dir, summary_path=summary_path)
    return records


def _build_specs(
    *,
    feature_sets: tuple[FeatureSet, ...],
    models: tuple[str, ...],
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for feature_set in feature_sets:
        for task in ("binary", "cause"):
            specs.append(
                ExperimentSpec(
                    group="5.1_feature_sets",
                    feature_set=feature_set,
                    model="logreg",
                    task=task,
                )
            )

    for model in models:
        for task in ("binary", "cause"):
            specs.append(
                ExperimentSpec(
                    group="5.2_model_comparison",
                    feature_set=FeatureSet.EXTENDED,
                    model=model,
                    task=task,
                )
            )
    return specs


def _build_feature_dirs(
    *,
    params: dict[str, Any],
    input_dir: Path,
    tmp_root: Path,
    feature_sets: tuple[FeatureSet, ...],
) -> dict[FeatureSet, Path]:
    dirs: dict[FeatureSet, Path] = {}
    for feature_set in feature_sets:
        feature_dir = tmp_root / feature_set.value.lower()
        build_feature_splits(
            input_dir=input_dir,
            output_dir=feature_dir,
            feature_set_name=feature_set,
            metadata_path=feature_dir / "features.json",
        )
        dirs[feature_set] = feature_dir
    return dirs


def _run_one(
    *,
    params: dict[str, Any],
    processed_dir: Path,
    spec: ExperimentSpec,
    metrics_path: Path,
) -> dict[str, Any]:
    task_cfg = _task_config(params=params, spec=spec)
    train = load_processed_split(processed_dir, "train")
    val = load_processed_split(processed_dir, "val")
    if spec.task == "cause":
        train = filter_delayed_only(train)
        val = filter_delayed_only(val)

    X_train, y_train = split_xy(train, task_cfg["target"])
    X_val, y_val = split_xy(val, task_cfg["target"])

    model = build_classifier(
        model_name=spec.model,
        task=spec.task,
        hyperparams=task_cfg["hyperparams"],
        seed=int(params["seed"]),
        class_weight=task_cfg.get("class_weight"),
    )

    started = time.perf_counter()
    y_fit = y_train.astype(int) if spec.task == "binary" else y_train.astype(str)
    model.fit(X_train, y_fit)
    train_time = time.perf_counter() - started

    metric_values = (
        binary_metrics(model, X_val, y_val)
        if spec.task == "binary"
        else multiclass_metrics(model, X_val, y_val)
    )
    metrics = {
        "task": spec.task,
        "model": spec.model,
        "feature_set": spec.feature_set.value,
        "target": task_cfg["target"],
        "train_rows": len(train),
        "val_rows": len(val),
        "train_time_seconds": float(train_time),
        "val": metric_values,
    }
    write_json(metrics, metrics_path)

    run_params = deepcopy(params)
    run_params["features"]["feature_set"] = spec.feature_set.value
    with mlflow_run(
        params=run_params,
        task_name=f"experiment_{spec.task}",
        experiment_name=params["mlflow"].get("experiment_experiments", "flight-delay-experiments"),
        run_name=f"{spec.group}_{spec.task}_{spec.feature_set.value}_{spec.model}",
    ) as run:
        if run is not None:
            log_training_run(
                params=run_params,
                task_config=task_cfg,
                metrics=metrics,
                metrics_path=metrics_path,
                model=model,
                model_artifact_name=f"{spec.task}_{spec.feature_set.value.lower()}_{spec.model}",
            )

    return {
        "task": spec.task,
        "feature_set": spec.feature_set.value,
        "model": spec.model,
        "train_rows": len(train),
        "val_rows": len(val),
        "train_time_seconds": float(train_time),
        "metrics": metric_values,
    }


def _task_config(*, params: dict[str, Any], spec: ExperimentSpec) -> dict[str, Any]:
    base_cfg = params["train_binary"] if spec.task == "binary" else params["train_cause"]
    model_cfg = DEFAULT_EXPERIMENT_MODELS[spec.model]
    return {
        "model": spec.model,
        "target": base_cfg["target"],
        "class_weight": model_cfg["class_weight"],
        "hyperparams": model_cfg["hyperparams"],
    }


def _write_outputs(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    summary_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json({"runs": records}, output_dir / "baseline_runs.json")
    _write_csv(records, output_dir / "baseline_runs.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_render_summary(records), encoding="utf-8")


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    rows = [_flat_record(record) for record in records]
    fieldnames = [
        "experiment_group",
        "task",
        "feature_set",
        "model",
        "train_rows",
        "val_rows",
        "train_time_seconds",
        "accuracy",
        "f1",
        "roc_auc",
        "pr_auc",
        "macro_f1",
        "weighted_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _flat_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record["metrics"]
    return {
        "experiment_group": record["experiment_group"],
        "task": record["task"],
        "feature_set": record["feature_set"],
        "model": record["model"],
        "train_rows": record["train_rows"],
        "val_rows": record["val_rows"],
        "train_time_seconds": round(record["train_time_seconds"], 4),
        "accuracy": metrics.get("accuracy"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
    }


def _render_summary(records: list[dict[str, Any]]) -> str:
    feature_rows = [r for r in records if r["experiment_group"] == "5.1_feature_sets"]
    model_rows = [r for r in records if r["experiment_group"] == "5.2_model_comparison"]

    binary_feature_rows = [r for r in feature_rows if r["task"] == "binary"]
    cause_feature_rows = [r for r in feature_rows if r["task"] == "cause"]
    binary_model_rows = [r for r in model_rows if r["task"] == "binary"]
    cause_model_rows = [r for r in model_rows if r["task"] == "cause"]

    return "\n".join(
        [
            "# Experiments Summary",
            "",
            "## Scope",
            "",
            "Baseline slice of stage 5: feature-set impact and sklearn model comparison.",
            "All runs use time-based validation: train=2023, val=2024. No `gt_*` columns are used as features.",
            "",
            "Reproduce:",
            "",
            "```bash",
            "make experiments-baseline",
            "```",
            "",
            "## Experiment 5.1 - Feature Sets",
            "",
            "### Binary Delay Classifier",
            "",
            _markdown_table(
                binary_feature_rows,
                columns=["feature_set", "model", "f1", "roc_auc", "pr_auc", "accuracy"],
            ),
            "",
            "### Cause Classifier",
            "",
            _markdown_table(
                cause_feature_rows,
                columns=["feature_set", "model", "macro_f1", "weighted_f1", "accuracy"],
            ),
            "",
            "## Experiment 5.2 - Model Comparison",
            "",
            "Feature set fixed to `EXTENDED` for this baseline comparison.",
            "",
            "### Binary Delay Classifier",
            "",
            _markdown_table(
                binary_model_rows,
                columns=["model", "feature_set", "f1", "roc_auc", "pr_auc", "accuracy"],
            ),
            "",
            "### Cause Classifier",
            "",
            _markdown_table(
                cause_model_rows,
                columns=["model", "feature_set", "macro_f1", "weighted_f1", "accuracy"],
            ),
            "",
            "## Current Best",
            "",
            _best_line(binary_feature_rows, metric="f1", label="Binary F1 by feature set"),
            _best_line(cause_feature_rows, metric="macro_f1", label="Cause macro-F1 by feature set"),
            _best_line(binary_model_rows, metric="f1", label="Binary F1 by model"),
            _best_line(cause_model_rows, metric="macro_f1", label="Cause macro-F1 by model"),
            "",
            "## Next Experiments",
            "",
            "- Add CatBoost, LightGBM, and XGBoost when the Python environment has compatible wheels.",
            "- Run Optuna tuning for top models.",
            "- Run imbalance and one-stage-vs-two-stage cause-classification experiments.",
            "- Add SHAP and concept-drift reports.",
            "",
        ]
    )


def _markdown_table(records: list[dict[str, Any]], columns: list[str]) -> str:
    if not records:
        return "_No runs yet._"

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in records:
        flat = _flat_record(record)
        rows.append("| " + " | ".join(_format_value(flat.get(col)) for col in columns) + " |")
    return "\n".join([header, sep, *rows])


def _best_line(records: list[dict[str, Any]], *, metric: str, label: str) -> str:
    if not records:
        return f"- {label}: no runs."
    best = max(records, key=lambda r: float(r["metrics"].get(metric, float("-inf"))))
    return (
        f"- {label}: `{best['model']}` + `{best['feature_set']}` "
        f"= {_format_value(best['metrics'][metric])}."
    )


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline stage-5 experiments.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--input-dir", default=None, help="Directory with split parquet files")
    parser.add_argument("--output-dir", default="reports/experiments")
    parser.add_argument("--summary", default="reports/experiments_summary.md")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = load_params(args.params)
    run_baseline_experiments(
        params=params,
        input_dir=Path(args.input_dir or Path(params["data"]["interim_dir"]) / "splits"),
        output_dir=Path(args.output_dir),
        summary_path=Path(args.summary),
    )


if __name__ == "__main__":
    main()
