"""Single source of truth for feature column selection.

CLAUDE.md rule 1: ``gt_*`` columns are NEVER features (target leakage).
Any code that builds a feature matrix MUST go through :func:`get_feature_columns`.
Never write ``df.drop([...])`` ad-hoc to bypass this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

# --- Forbidden / reserved columns ---------------------------------------------------------------

FORBIDDEN_FEATURE_PREFIXES: tuple[str, ...] = ("gt_",)
"""Prefixes of columns that decompose the target — using them as features = target leakage."""

TARGET_COLUMNS: frozenset[str] = frozenset(
    {
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
)
"""Targets and post-flight outcomes — never features."""

ID_COLUMNS: frozenset[str] = frozenset(
    {
        "flight_id",
        "schedule_id",
        "flight_date",
    }
)
"""Identifiers — used for splitting/logging, not as features."""

# Human-readable string columns whose information is already captured by their codes.
# Keeping them as features would just add noise / cardinality.
REDUNDANT_HUMAN_LABELS: frozenset[str] = frozenset(
    {
        "airline_name",
        "origin_city",
        "destination_city",
        "flight_number",
        "scheduled_departure_local",
        "scheduled_arrival_local",
    }
)
"""Readable/high-cardinality duplicates already represented by structured columns."""


# --- Feature set catalogue ----------------------------------------------------------------------


class FeatureSet(StrEnum):
    """Three feature sets compared in experiment 5.1 (CLAUDE.md §7 Этап 2).

    BASELINE       — base columns only, no engineered derivatives.
    EXTENDED       — BASELINE + cyclic encodings (sin/cos) + cross-route engineered features.
    WITH_NETWORK   — EXTENDED + airport/airline target encodings (computed without leakage).
    """

    BASELINE = "BASELINE"
    EXTENDED = "EXTENDED"
    WITH_NETWORK = "WITH_NETWORK"


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative description of a feature set."""

    name: FeatureSet
    add_cyclic_temporal: bool = False
    add_cross_route: bool = False
    add_network_encodings: bool = False
    extra_engineered: tuple[str, ...] = field(default_factory=tuple)


FEATURE_SPECS: dict[FeatureSet, FeatureSpec] = {
    FeatureSet.BASELINE: FeatureSpec(name=FeatureSet.BASELINE),
    FeatureSet.EXTENDED: FeatureSpec(
        name=FeatureSet.EXTENDED,
        add_cyclic_temporal=True,
        add_cross_route=True,
    ),
    FeatureSet.WITH_NETWORK: FeatureSpec(
        name=FeatureSet.WITH_NETWORK,
        add_cyclic_temporal=True,
        add_cross_route=True,
        add_network_encodings=True,
    ),
}


def get_feature_set(name: str | FeatureSet) -> FeatureSpec:
    """Resolve a feature set by name (case-insensitive)."""
    key = FeatureSet(name.upper()) if isinstance(name, str) else name
    return FEATURE_SPECS[key]


# --- Public API ---------------------------------------------------------------------------------


def get_feature_columns(
    df: pd.DataFrame,
    *,
    drop_redundant_labels: bool = True,
    extra_excluded: frozenset[str] | None = None,
) -> list[str]:
    """Return the list of columns from ``df`` that are valid model features.

    Filters out (in order):
      1. Any column starting with a forbidden prefix (``gt_*``).
      2. Any target / post-flight column.
      3. Any identifier column.
      4. Optionally, redundant human-readable labels (default True).
      5. Any caller-specified extras.

    The function never raises on missing columns — it inspects the DataFrame as-is.
    Order of returned columns matches DataFrame column order (deterministic).
    """
    excluded: set[str] = set(TARGET_COLUMNS) | set(ID_COLUMNS)
    if drop_redundant_labels:
        excluded |= set(REDUNDANT_HUMAN_LABELS)
    if extra_excluded:
        excluded |= set(extra_excluded)

    return [
        col
        for col in df.columns
        if not col.startswith(FORBIDDEN_FEATURE_PREFIXES) and col not in excluded
    ]


def assert_no_leakage(feature_columns: list[str]) -> None:
    """Hard guard: raise if any forbidden / target / id column slipped into features."""
    forbidden = [c for c in feature_columns if c.startswith(FORBIDDEN_FEATURE_PREFIXES)]
    targets = [c for c in feature_columns if c in TARGET_COLUMNS]
    ids = [c for c in feature_columns if c in ID_COLUMNS]
    problems = forbidden + targets + ids
    if problems:
        raise ValueError(
            f"Target leakage detected — these columns are not allowed as features: {problems}"
        )
