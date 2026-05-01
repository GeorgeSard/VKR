"""Programmatic data-quality checks (formal rules from DATA_QUALITY_REPORT.md).

The validator is intentionally strict: any unexpected schema change should fail loudly,
because feature engineering and split logic depend on these invariants.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

EXPECTED_COLUMN_COUNT = 66
EXPECTED_AIRPORTS = 22
EXPECTED_AIRLINES = 11
EXPECTED_YEARS: frozenset[int] = frozenset({2023, 2024, 2025})
GT_PREFIX = "gt_"

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "flight_id",
        "schedule_id",
        "flight_date",
        "year",
        "month",
        "day",
        "day_of_week",
        "scheduled_dep_hour",
        "airline_code",
        "origin_iata",
        "destination_iata",
        "distance_km",
        "is_departure_delayed_15m",
        "dep_delay_minutes",
        "probable_delay_cause",
        "cancellation_flag",
    }
)


@dataclass
class ValidationReport:
    """Result of running validation. ``ok`` is True iff there are no errors."""

    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise ValueError(f"Dataset validation failed:\n  - {joined}")


def validate(df: pd.DataFrame, *, strict: bool = True) -> ValidationReport:
    """Run all quality checks. Use ``strict=True`` in CI / training pipelines."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Required columns are present.
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    # 2. Column count sanity.
    if len(df.columns) != EXPECTED_COLUMN_COUNT:
        warnings.append(
            f"Expected {EXPECTED_COLUMN_COUNT} columns, found {len(df.columns)}. "
            "Schema drift — review feature_sets.py."
        )

    # 3. ``flight_id`` must be unique.
    if "flight_id" in df.columns and df["flight_id"].duplicated().any():
        errors.append("flight_id contains duplicates")

    # 4. Years inside expected range.
    if "year" in df.columns:
        years = set(df["year"].dropna().unique().astype(int))
        unexpected_years = years - EXPECTED_YEARS
        if unexpected_years:
            errors.append(f"Unexpected years in data: {sorted(unexpected_years)}")

    # 5. Airports / airlines counts.
    if "origin_iata" in df.columns and "destination_iata" in df.columns:
        all_airports = set(df["origin_iata"].astype(str)).union(
            df["destination_iata"].astype(str)
        )
        if len(all_airports) > EXPECTED_AIRPORTS:
            warnings.append(
                f"Expected {EXPECTED_AIRPORTS} airports, found {len(all_airports)}"
            )
    if "airline_code" in df.columns:
        n_airlines = df["airline_code"].nunique()
        if n_airlines != EXPECTED_AIRLINES:
            warnings.append(f"Expected {EXPECTED_AIRLINES} airlines, found {n_airlines}")

    # 6. Cancellation invariant: cancelled flights have NaN dep_delay_minutes.
    if {"cancellation_flag", "dep_delay_minutes"}.issubset(df.columns):
        cancelled_with_delay = df[
            (df["cancellation_flag"] == 1) & df["dep_delay_minutes"].notna()
        ]
        if len(cancelled_with_delay) > 0:
            errors.append(
                f"{len(cancelled_with_delay)} cancelled flights have non-null dep_delay_minutes"
            )

    # 7. Binary target invariant: is_departure_delayed_15m matches dep_delay_minutes >= 15.
    if {"is_departure_delayed_15m", "dep_delay_minutes"}.issubset(df.columns):
        non_cancelled = df[df.get("cancellation_flag", 0) == 0].copy()
        if len(non_cancelled):
            expected = (non_cancelled["dep_delay_minutes"] >= 15).astype(int)
            actual = non_cancelled["is_departure_delayed_15m"].astype(int)
            mismatches = int((expected != actual).sum())
            if mismatches:
                errors.append(
                    f"{mismatches} flights where is_departure_delayed_15m disagrees "
                    f"with dep_delay_minutes >= 15"
                )

    # 8. Decomposition: sum of gt_* approximately equals dep_delay_minutes (within buffer).
    gt_cols = [c for c in df.columns if c.startswith(GT_PREFIX)]
    if gt_cols and "dep_delay_minutes" in df.columns:
        non_cancelled = df[df.get("cancellation_flag", 0) == 0].copy()
        gt_sum = non_cancelled[gt_cols].sum(axis=1)
        # Buffer absorbs up to ~8 min, so dep_delay should be in [gt_sum - 8, gt_sum].
        diff = gt_sum - non_cancelled["dep_delay_minutes"].fillna(0)
        bad = ((diff < -1) | (diff > 12)).sum()
        if bad > 0:
            warnings.append(
                f"{int(bad)} rows where sum(gt_*) − dep_delay_minutes is outside [−1, 12]"
            )

    # 9. ``probable_delay_cause`` is "none" only when not delayed.
    if {"probable_delay_cause", "is_departure_delayed_15m"}.issubset(df.columns):
        delayed_with_none = df[
            (df["is_departure_delayed_15m"] == 1)
            & (df["probable_delay_cause"] == "none")
        ]
        if len(delayed_with_none) > 0:
            errors.append(
                f"{len(delayed_with_none)} delayed flights have probable_delay_cause='none'"
            )

    report = ValidationReport(errors=errors, warnings=warnings)
    if strict:
        report.raise_if_errors()
    return report
