"""Tests for stage-5 baseline experiment runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.split import time_based_split
from src.experiments.run_baselines import run_baseline_experiments
from src.features.feature_sets import FeatureSet


def test_baseline_experiments_write_summary(
    tmp_path: Path,
    synthetic_frame: pd.DataFrame,
) -> None:
    input_dir = _write_split_inputs(tmp_path, synthetic_frame)
    output_dir = tmp_path / "experiments"
    summary_path = tmp_path / "experiments_summary.md"
    params = _params(mlflow_enabled=False)

    records = run_baseline_experiments(
        params=params,
        input_dir=input_dir,
        output_dir=output_dir,
        summary_path=summary_path,
        feature_sets=(FeatureSet.BASELINE,),
        models=("logreg",),
    )

    assert records
    assert (output_dir / "baseline_runs.json").exists()
    assert (output_dir / "baseline_runs.csv").exists()
    assert "Experiment 5.1" in summary_path.read_text(encoding="utf-8")


def _write_split_inputs(tmp_path: Path, frame: pd.DataFrame) -> Path:
    result = time_based_split(frame)
    input_dir = tmp_path / "splits"
    input_dir.mkdir()
    result.train.to_parquet(input_dir / "train.parquet", index=False)
    result.val.to_parquet(input_dir / "val.parquet", index=False)
    result.test.to_parquet(input_dir / "test.parquet", index=False)
    return input_dir


def _params(*, mlflow_enabled: bool) -> dict:
    return {
        "seed": 42,
        "split": {
            "train_years": [2023],
            "val_years": [2024],
            "test_years": [2025],
        },
        "features": {"feature_set": "BASELINE"},
        "train_binary": {"target": "is_departure_delayed_15m"},
        "train_cause": {"target": "probable_delay_cause"},
        "mlflow": {
            "enabled": mlflow_enabled,
            "tracking_uri": "sqlite:///mlflow.db",
            "experiment_experiments": "test-experiments",
        },
    }
