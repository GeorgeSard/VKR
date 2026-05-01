"""Tests for DVC feature-building stage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.split import time_based_split
from src.features.build import build_feature_splits
from src.features.feature_sets import FeatureSet, assert_no_leakage, get_feature_columns


def test_extended_feature_build_writes_no_leakage_outputs(
    tmp_path: Path,
    synthetic_frame: pd.DataFrame,
) -> None:
    input_dir = _write_split_inputs(tmp_path, synthetic_frame)
    output_dir = tmp_path / "processed"
    metadata_path = tmp_path / "features.json"

    build_feature_splits(
        input_dir=input_dir,
        output_dir=output_dir,
        feature_set_name=FeatureSet.EXTENDED,
        metadata_path=metadata_path,
    )

    train = pd.read_parquet(output_dir / "train.parquet")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = get_feature_columns(train)

    assert metadata["feature_set"] == "EXTENDED"
    assert "month_sin" in train.columns
    assert "temperature_diff_c" in train.columns
    assert "dep_delay_minutes" not in train.columns
    assert_no_leakage(feature_columns)


def test_network_feature_build_uses_train_only_encodings(
    tmp_path: Path,
    synthetic_frame: pd.DataFrame,
) -> None:
    input_dir = _write_split_inputs(tmp_path, synthetic_frame)
    output_dir = tmp_path / "processed"

    build_feature_splits(
        input_dir=input_dir,
        output_dir=output_dir,
        feature_set_name=FeatureSet.WITH_NETWORK,
        metadata_path=tmp_path / "features.json",
    )

    val = pd.read_parquet(output_dir / "val.parquet")
    encoding_cols = [c for c in val.columns if c.startswith("te_")]
    assert encoding_cols
    for column in encoding_cols:
        assert val[column].between(0, 1).all()


def _write_split_inputs(tmp_path: Path, frame: pd.DataFrame) -> Path:
    result = time_based_split(frame)
    input_dir = tmp_path / "splits"
    input_dir.mkdir()
    result.train.to_parquet(input_dir / "train.parquet", index=False)
    result.val.to_parquet(input_dir / "val.parquet", index=False)
    result.test.to_parquet(input_dir / "test.parquet", index=False)
    return input_dir
