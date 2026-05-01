"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import project_path

RAW_PATH = project_path("data", "raw", "flight_delays_ru.parquet")


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    """A tiny in-memory frame matching the real schema (subset of columns).

    Used to keep unit tests fast and independent of the 12 MB parquet.
    Three years, balanced positives/negatives, all targets present.
    """
    rng = np.random.default_rng(42)
    n_per_year = 200
    rows = []
    for year in (2023, 2024, 2025):
        for i in range(n_per_year):
            month = rng.integers(1, 13)
            dep_delay = int(rng.gamma(2.0, 8.0)) if rng.random() < 0.3 else 0
            cancelled = 1 if rng.random() < 0.02 else 0
            cause = (
                "none"
                if dep_delay < 15 or cancelled
                else rng.choice(
                    [
                        "weather",
                        "airport_congestion",
                        "reactionary",
                        "carrier_operational",
                        "security",
                    ]
                )
            )
            rows.append(
                {
                    "flight_id": f"RU{year}{i:07d}",
                    "schedule_id": f"SCH{i % 50:04d}",
                    "flight_date": pd.Timestamp(f"{year}-{int(month):02d}-15"),
                    "year": year,
                    "month": int(month),
                    "day": 15,
                    "day_of_week": int(rng.integers(0, 7)),
                    "scheduled_dep_hour": int(rng.integers(5, 24)),
                    "airline_code": rng.choice(["SU", "S7", "DP", "U6"]),
                    "aircraft_family": rng.choice(["A320", "B737", "SSJ100"]),
                    "origin_iata": rng.choice(["SVO", "DME", "LED", "OVB"]),
                    "destination_iata": rng.choice(["AER", "KZN", "SVX"]),
                    "distance_km": int(rng.integers(300, 5000)),
                    "origin_temperature_c": float(rng.normal(10, 15)),
                    "destination_temperature_c": float(rng.normal(10, 15)),
                    "origin_congestion_index": float(rng.uniform(0.1, 0.9)),
                    "destination_congestion_index": float(rng.uniform(0.1, 0.9)),
                    "inbound_delay_minutes": int(max(0, rng.normal(5, 10))),
                    # gt_* — must NEVER appear in feature lists
                    "gt_carrier_delay_minutes": int(rng.gamma(1.0, 5.0)),
                    "gt_weather_delay_minutes": int(rng.gamma(1.0, 5.0)),
                    "gt_airport_congestion_delay_minutes": int(rng.gamma(1.0, 5.0)),
                    "gt_reactionary_delay_minutes": int(rng.gamma(1.0, 5.0)),
                    "gt_security_delay_minutes": int(rng.gamma(0.5, 2.0)),
                    # targets
                    "dep_delay_minutes": None if cancelled else dep_delay,
                    "is_departure_delayed_15m": None if cancelled else int(dep_delay >= 15),
                    "cancellation_flag": cancelled,
                    "cancellation_reason": "weather" if cancelled else None,
                    "diversion_flag": 0,
                    "probable_delay_cause": "cancelled" if cancelled else cause,
                    "actual_departure_local": None,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def real_frame_path() -> Path:
    """Path to the real raw parquet (used by ``@pytest.mark.integration`` tests)."""
    return RAW_PATH


@pytest.fixture(scope="session")
def real_frame(real_frame_path: Path) -> pd.DataFrame:
    """Load the real dataset, skipping the test if it's not available locally."""
    if not real_frame_path.exists():
        pytest.skip(f"Real dataset not found at {real_frame_path}")
    return pd.read_parquet(real_frame_path)
