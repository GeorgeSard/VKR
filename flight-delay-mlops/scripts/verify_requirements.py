"""Project-level checks that map implementation artifacts to CLAUDE.md requirements."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    checks = [
        ("DVC pipeline has all required stages", check_dvc_stages),
        ("Raw data is tracked via DVC metadata", check_raw_data_dvc),
        ("Feature metadata contains no leakage columns", check_no_feature_leakage),
        ("Time-based split is 2023/2024/2025", check_time_split),
        ("Binary metrics include F1, ROC-AUC, PR-AUC", check_binary_metrics),
        ("Cause metrics include macro-F1", check_cause_metrics),
        ("Baseline experiment report exists", check_experiment_report),
        ("MLflow tracking produced local runs or server URI is configured", check_mlflow_tracking),
        ("No raw parquet/csv/model binaries are tracked by git", check_git_large_files),
    ]

    failed = 0
    for title, check in checks:
        ok, detail = check()
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] {title}: {detail}")
        failed += 0 if ok else 1

    return 1 if failed else 0


def check_dvc_stages() -> tuple[bool, str]:
    dvc = _read_yaml(ROOT / "dvc.yaml")
    expected = {"load", "split", "features", "train_binary", "train_cause", "evaluate"}
    actual = set(dvc.get("stages", {}))
    missing = expected - actual
    return not missing, f"stages={sorted(actual)}" if not missing else f"missing={sorted(missing)}"


def check_raw_data_dvc() -> tuple[bool, str]:
    paths = [
        ROOT / "data/raw/flight_delays_ru.parquet.dvc",
        ROOT / "data/raw/flight_delays_ru_sample.csv.dvc",
    ]
    ok = all(path.exists() for path in paths)
    return ok, ", ".join(path.name for path in paths)


def check_no_feature_leakage() -> tuple[bool, str]:
    metadata = _read_json(ROOT / "reports/metrics/features.json")
    features = metadata.get("feature_columns", [])
    forbidden = [c for c in features if c.startswith("gt_")]
    targets = {
        "dep_delay_minutes",
        "arr_delay_minutes",
        "is_departure_delayed_15m",
        "is_arrival_delayed_15m",
        "cancellation_flag",
        "cancellation_reason",
        "diversion_flag",
        "probable_delay_cause",
        "actual_departure_local",
        "actual_arrival_local",
    }
    leaked = forbidden + [c for c in features if c in targets]
    return not leaked, f"{len(features)} features" if not leaked else f"leaked={leaked}"


def check_time_split() -> tuple[bool, str]:
    report = _read_json(ROOT / "reports/metrics/split_report.json")
    ok = (
        report.get("train_years") == [2023]
        and report.get("val_years") == [2024]
        and report.get("test_years") == [2025]
    )
    return ok, json.dumps({k: report[k] for k in ("train_years", "val_years", "test_years")})


def check_binary_metrics() -> tuple[bool, str]:
    metrics = _read_json(ROOT / "reports/metrics/test_metrics.json")["binary"]["metrics"]
    required = {"f1", "roc_auc", "pr_auc"}
    missing = required - set(metrics)
    return not missing, _metric_summary(metrics, ["f1", "roc_auc", "pr_auc"])


def check_cause_metrics() -> tuple[bool, str]:
    metrics = _read_json(ROOT / "reports/metrics/test_metrics.json")["cause"]["metrics"]
    return "macro_f1" in metrics, _metric_summary(metrics, ["macro_f1", "weighted_f1"])


def check_experiment_report() -> tuple[bool, str]:
    summary = ROOT / "reports/experiments_summary.md"
    runs = ROOT / "reports/experiments/baseline_runs.json"
    if not summary.exists() or not runs.exists():
        return False, "run `make experiments-baseline`"
    payload = _read_json(runs)
    records = payload.get("runs", [])
    groups = {record.get("experiment_group") for record in records}
    required = {"5.1_feature_sets", "5.2_model_comparison"}
    missing = required - groups
    return not missing, f"runs={len(records)}" if not missing else f"missing={sorted(missing)}"


def check_mlflow_tracking() -> tuple[bool, str]:
    local_runs = ROOT / "mlruns"
    if local_runs.exists() and any(local_runs.glob("*/meta.yaml")):
        return True, "local mlruns exists"
    params = _read_yaml(ROOT / "params.yaml")
    uri = str(params.get("mlflow", {}).get("tracking_uri", ""))
    ok = uri.startswith(("http://", "https://", "file:", "sqlite:"))
    return ok, f"tracking_uri={uri}"


def check_git_large_files() -> tuple[bool, str]:
    tracked = _git("ls-files").splitlines()
    bad_suffixes = (".parquet", ".csv", ".joblib")
    allowed = {
        "flight-delay-mlops/data/raw/flight_delays_ru.parquet.dvc",
        "flight-delay-mlops/data/raw/flight_delays_ru_sample.csv.dvc",
        "reports/experiments/baseline_runs.csv",
    }
    bad = [
        p
        for p in tracked
        if p.endswith(bad_suffixes) and p not in allowed and not p.endswith(".dvc")
    ]
    return not bad, "ok" if not bad else f"tracked large artifacts={bad}"


def _metric_summary(metrics: dict[str, Any], keys: list[str]) -> str:
    return ", ".join(f"{key}={metrics.get(key):.4f}" for key in keys if key in metrics)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


if __name__ == "__main__":
    sys.exit(main())
