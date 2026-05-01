"""Loading and validating the raw flight delay dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_params, project_path
from src.data.validate import validate

DEFAULT_RAW_PATH = project_path("data", "raw", "flight_delays_ru.parquet")

# Categorical columns that should be loaded as ``category`` dtype (faster + less memory).
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "airline_code",
    "aircraft_family",
    "origin_iata",
    "origin_region",
    "destination_iata",
    "destination_region",
    "route_group",
    "origin_weather_severity",
    "destination_weather_severity",
    "cancellation_reason",
    "probable_delay_cause",
)


def load_raw(path: Path | str | None = None, *, parse_dates: bool = True) -> pd.DataFrame:
    """Read the raw parquet (or CSV) file into a DataFrame.

    - Casts known categorical columns to ``category`` dtype.
    - Optionally parses ``flight_date`` to datetime64.
    """
    p = Path(path) if path else DEFAULT_RAW_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at {p}. Run `dvc pull` or "
            f"`python -m src.data.generate_dataset --out {p} --rows 220000`."
        )

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if parse_dates and "flight_date" in df.columns:
        df["flight_date"] = pd.to_datetime(df["flight_date"])

    return df


def write_validated_dataset(
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    strict: bool = True,
) -> None:
    """Load raw data, run quality checks, and persist the validated parquet."""
    df = load_raw(input_path)
    report = validate(df, strict=strict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    payload: dict[str, Any] = {
        "ok": report.ok,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "error_count": len(report.errors),
        "warning_count": len(report.warnings),
        "errors": report.errors,
        "warnings": report.warnings,
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw flight delay data.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--input", default=None, help="Raw parquet/csv path")
    parser.add_argument("--output", default=None, help="Validated parquet output path")
    parser.add_argument("--report", default="reports/metrics/data_validation.json")
    parser.add_argument("--non-strict", action="store_true", help="Keep report on validation errors")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint used by the DVC ``load`` stage."""
    args = _parse_args()
    params = load_params(args.params)
    input_path = Path(args.input or params["data"]["raw_path"])
    output_path = Path(args.output or Path(params["data"]["interim_dir"]) / "validated.parquet")
    report_path = Path(args.report)

    write_validated_dataset(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        strict=not args.non_strict,
    )


if __name__ == "__main__":
    main()
