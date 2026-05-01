"""Build reproducible feature datasets for DVC.

All base feature selection goes through ``get_feature_columns`` to preserve the
project's no-leakage rule from CLAUDE.md and DATA_DICTIONARY.md.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_params
from src.features.feature_sets import (
    FeatureSet,
    assert_no_leakage,
    get_feature_columns,
    get_feature_set,
)

TARGET_OUTPUT_COLUMNS: tuple[str, ...] = (
    "flight_id",
    "is_departure_delayed_15m",
    "probable_delay_cause",
)

CYCLIC_PERIODS: dict[str, int] = {
    "month": 12,
    "day_of_week": 7,
    "scheduled_dep_hour": 24,
}

NETWORK_GROUP_COLUMNS: tuple[str, ...] = (
    "airline_code",
    "origin_iata",
    "destination_iata",
    "route_pair",
)


def build_feature_splits(
    *,
    input_dir: Path,
    output_dir: Path,
    feature_set_name: str | FeatureSet,
    metadata_path: Path,
) -> None:
    """Read split parquet files and write feature-ready train/val/test files."""
    spec = get_feature_set(feature_set_name)
    train = pd.read_parquet(input_dir / "train.parquet")
    val = pd.read_parquet(input_dir / "val.parquet")
    test = pd.read_parquet(input_dir / "test.parquet")

    frames = {"train": train, "val": val, "test": test}
    if spec.add_cyclic_temporal:
        frames = {name: add_cyclic_temporal_features(frame) for name, frame in frames.items()}
    if spec.add_cross_route:
        frames = {name: add_cross_route_features(frame) for name, frame in frames.items()}
    if spec.add_network_encodings:
        frames = add_network_delay_rate_encodings(frames)

    feature_columns = get_feature_columns(frames["train"])
    assert_no_leakage(feature_columns)
    output_columns = _dedupe([c for c in TARGET_OUTPUT_COLUMNS if c in frames["train"].columns])
    output_columns = _dedupe(output_columns + feature_columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame in frames.items():
        frame.loc[:, output_columns].to_parquet(output_dir / f"{split_name}.parquet", index=False)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_set": spec.name.value,
        "feature_spec": {
            "name": spec.name.value,
            "add_cyclic_temporal": spec.add_cyclic_temporal,
            "add_cross_route": spec.add_cross_route,
            "add_network_encodings": spec.add_network_encodings,
            "extra_engineered": list(spec.extra_engineered),
        },
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "splits": {name: {"rows": len(frame)} for name, frame in frames.items()},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def add_cyclic_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings for month, day of week, and departure hour."""
    result = df.copy()
    for column, period in CYCLIC_PERIODS.items():
        if column not in result.columns:
            continue
        values = result[column].astype(float)
        angle = 2 * np.pi * values / period
        result[f"{column}_sin"] = np.sin(angle)
        result[f"{column}_cos"] = np.cos(angle)
    return result


def add_cross_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add route/weather deltas known before departure."""
    result = df.copy()
    if {"origin_utc_offset", "destination_utc_offset"}.issubset(result.columns):
        diff = result["origin_utc_offset"] - result["destination_utc_offset"]
        result["utc_offset_diff"] = diff
        result["abs_utc_offset_diff"] = diff.abs()
    if {"origin_temperature_c", "destination_temperature_c"}.issubset(result.columns):
        result["temperature_diff_c"] = (
            result["origin_temperature_c"] - result["destination_temperature_c"]
        )
    if {"origin_congestion_index", "destination_congestion_index"}.issubset(result.columns):
        result["congestion_diff"] = (
            result["origin_congestion_index"] - result["destination_congestion_index"]
        )
    if {"origin_hub_tier", "destination_hub_tier"}.issubset(result.columns):
        result["hub_tier_diff"] = result["origin_hub_tier"] - result["destination_hub_tier"]
    return result


def add_network_delay_rate_encodings(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add smoothed train-only delay-rate encodings without future leakage.

    Validation and test mappings are fitted on train only. Train rows use a
    leave-one-out variant, so a row's own target is not encoded back into itself.
    """
    target = "is_departure_delayed_15m"
    train = _with_route_pair(frames["train"])
    encoded = {"train": train.copy()}
    for split in ("val", "test"):
        encoded[split] = _with_route_pair(frames[split])

    prior = float(train[target].mean())
    smoothing = 20.0
    y = train[target].astype(float)

    for column in NETWORK_GROUP_COLUMNS:
        if column not in train.columns:
            continue
        train_keys = train[column].astype(str)
        grouped = (
            train.assign(_network_key=train_keys)
            .groupby("_network_key", observed=True)[target]
            .agg(["sum", "count"])
        )
        sums = train_keys.map(grouped["sum"]).astype(float)
        counts = train_keys.map(grouped["count"]).astype(float)

        train_values = (sums - y + prior * smoothing) / (counts - 1 + smoothing)
        encoded["train"][f"te_{column}_delay_rate"] = train_values.astype(float)

        mapping = ((grouped["sum"] + prior * smoothing) / (grouped["count"] + smoothing)).to_dict()
        for split in ("val", "test"):
            encoded[split][f"te_{column}_delay_rate"] = (
                encoded[split][column].astype(str).map(mapping).fillna(prior).astype(float)
            )

    return encoded


def _with_route_pair(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if {"origin_iata", "destination_iata"}.issubset(result.columns):
        result["route_pair"] = (
            result["origin_iata"].astype(str) + "_" + result["destination_iata"].astype(str)
        )
    return result


def _dedupe(columns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        result.append(column)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test feature parquet files.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--input-dir", default=None, help="Directory with split parquet files")
    parser.add_argument("--output-dir", default=None, help="Directory for processed features")
    parser.add_argument("--feature-set", default=None, choices=[item.value for item in FeatureSet])
    parser.add_argument("--metadata", default="reports/metrics/features.json")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint used by the DVC ``features`` stage."""
    args = _parse_args()
    params = load_params(args.params)
    build_feature_splits(
        input_dir=Path(args.input_dir or Path(params["data"]["interim_dir"]) / "splits"),
        output_dir=Path(args.output_dir or params["data"]["processed_dir"]),
        feature_set_name=args.feature_set or params["features"]["feature_set"],
        metadata_path=Path(args.metadata),
    )


if __name__ == "__main__":
    main()
