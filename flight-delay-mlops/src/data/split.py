"""Time-based train/val/test splitting.

CLAUDE.md rule 2: NEVER use ``train_test_split``. The dataset has built-in
concept drift between 2024 and 2025 — random splitting hides it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import load_params


@dataclass(frozen=True)
class SplitConfig:
    train_years: tuple[int, ...] = (2023,)
    val_years: tuple[int, ...] = (2024,)
    test_years: tuple[int, ...] = (2025,)
    drop_cancelled: bool = True

    def all_years(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.train_years + self.val_years + self.test_years)))


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    config: SplitConfig

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def time_based_split(
    df: pd.DataFrame,
    config: SplitConfig | None = None,
) -> SplitResult:
    """Split by ``year`` into disjoint train/val/test slices.

    The function asserts years do not overlap and that every requested year
    is present in the data.
    """
    cfg = config or SplitConfig()

    if "year" not in df.columns:
        raise ValueError("DataFrame must contain a 'year' column for time-based splitting")

    _assert_disjoint(cfg.train_years, cfg.val_years, cfg.test_years)

    available_years = set(df["year"].dropna().astype(int).unique())
    requested_years = set(cfg.all_years())
    missing = requested_years - available_years
    if missing:
        raise ValueError(
            f"Years requested for split are missing from the dataset: {sorted(missing)}"
        )

    work = df
    if cfg.drop_cancelled and "cancellation_flag" in work.columns:
        work = work[work["cancellation_flag"] == 0]

    train = work[work["year"].isin(cfg.train_years)].copy()
    val = work[work["year"].isin(cfg.val_years)].copy()
    test = work[work["year"].isin(cfg.test_years)].copy()

    return SplitResult(train=train, val=val, test=test, config=cfg)


def _assert_disjoint(*year_groups: Iterable[int]) -> None:
    seen: set[int] = set()
    for group in year_groups:
        s = set(group)
        overlap = seen & s
        if overlap:
            raise ValueError(f"Year split overlap detected: {sorted(overlap)}")
        seen |= s


def filter_delayed_only(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 subset: only flights with ``is_departure_delayed_15m == 1``.

    Used to train the multiclass cause classifier.
    """
    if "is_departure_delayed_15m" not in df.columns:
        raise ValueError("Column 'is_departure_delayed_15m' is required")
    return df[df["is_departure_delayed_15m"] == 1].copy()


def write_splits(
    *,
    input_path: Path,
    output_dir: Path,
    report_path: Path,
    config: SplitConfig,
) -> None:
    """Persist train/val/test parquet files and a compact split report."""
    df = pd.read_parquet(input_path)
    result = time_based_split(df, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    result.train.to_parquet(output_dir / "train.parquet", index=False)
    result.val.to_parquet(output_dir / "val.parquet", index=False)
    result.test.to_parquet(output_dir / "test.parquet", index=False)

    payload = {
        "train_years": list(config.train_years),
        "val_years": list(config.val_years),
        "test_years": list(config.test_years),
        "drop_cancelled": config.drop_cancelled,
        "sizes": result.sizes(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create mandatory time-based data splits.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--input", default=None, help="Validated parquet input path")
    parser.add_argument("--output-dir", default=None, help="Directory for split parquet files")
    parser.add_argument("--report", default="reports/metrics/split_report.json")
    parser.add_argument("--keep-cancelled", action="store_true")
    return parser.parse_args()


def _years(values: list[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def main() -> None:
    """CLI entrypoint used by the DVC ``split`` stage."""
    args = _parse_args()
    params = load_params(args.params)
    interim_dir = Path(params["data"]["interim_dir"])

    cfg = SplitConfig(
        train_years=_years(params["split"]["train_years"]),
        val_years=_years(params["split"]["val_years"]),
        test_years=_years(params["split"]["test_years"]),
        drop_cancelled=not args.keep_cancelled,
    )
    write_splits(
        input_path=Path(args.input or interim_dir / "validated.parquet"),
        output_dir=Path(args.output_dir or interim_dir / "splits"),
        report_path=Path(args.report),
        config=cfg,
    )


if __name__ == "__main__":
    main()
