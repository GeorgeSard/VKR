"""CLAUDE.md rule 1: gt_* columns and targets must NEVER appear among features."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.feature_sets import (
    FORBIDDEN_FEATURE_PREFIXES,
    ID_COLUMNS,
    TARGET_COLUMNS,
    FeatureSet,
    assert_no_leakage,
    get_feature_columns,
    get_feature_set,
)


class TestGetFeatureColumns:
    def test_no_gt_prefix(self, synthetic_frame: pd.DataFrame) -> None:
        cols = get_feature_columns(synthetic_frame)
        for c in cols:
            for prefix in FORBIDDEN_FEATURE_PREFIXES:
                assert not c.startswith(prefix), f"feature '{c}' has forbidden prefix '{prefix}'"

    def test_no_targets(self, synthetic_frame: pd.DataFrame) -> None:
        cols = set(get_feature_columns(synthetic_frame))
        leaked = cols & TARGET_COLUMNS
        assert not leaked, f"target columns leaked into features: {leaked}"

    def test_no_ids(self, synthetic_frame: pd.DataFrame) -> None:
        cols = set(get_feature_columns(synthetic_frame))
        leaked = cols & ID_COLUMNS
        assert not leaked, f"id columns leaked into features: {leaked}"

    def test_drops_redundant_labels_by_default(self, synthetic_frame: pd.DataFrame) -> None:
        df = synthetic_frame.assign(airline_name="X", origin_city="Y")
        cols = get_feature_columns(df)
        assert "airline_name" not in cols
        assert "origin_city" not in cols

    def test_extra_excluded_respected(self, synthetic_frame: pd.DataFrame) -> None:
        cols = get_feature_columns(synthetic_frame, extra_excluded=frozenset({"distance_km"}))
        assert "distance_km" not in cols

    def test_deterministic_order(self, synthetic_frame: pd.DataFrame) -> None:
        a = get_feature_columns(synthetic_frame)
        b = get_feature_columns(synthetic_frame)
        assert a == b

    def test_returns_only_existing_columns(self, synthetic_frame: pd.DataFrame) -> None:
        cols = get_feature_columns(synthetic_frame)
        assert set(cols).issubset(set(synthetic_frame.columns))

    def test_includes_useful_features(self, synthetic_frame: pd.DataFrame) -> None:
        """Sanity: real predictors must be selected."""
        cols = set(get_feature_columns(synthetic_frame))
        for must_have in (
            "airline_code",
            "origin_iata",
            "destination_iata",
            "distance_km",
            "scheduled_dep_hour",
            "origin_congestion_index",
            "inbound_delay_minutes",
        ):
            assert must_have in cols, f"expected feature '{must_have}' is missing"


class TestAssertNoLeakage:
    def test_passes_on_clean(self, synthetic_frame: pd.DataFrame) -> None:
        assert_no_leakage(get_feature_columns(synthetic_frame))

    def test_fails_on_gt(self) -> None:
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage(["airline_code", "gt_weather_delay_minutes"])

    def test_fails_on_target(self) -> None:
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage(["airline_code", "dep_delay_minutes"])

    def test_fails_on_id(self) -> None:
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage(["airline_code", "flight_id"])


class TestFeatureSetCatalogue:
    @pytest.mark.parametrize(
        "name", [FeatureSet.BASELINE, FeatureSet.EXTENDED, FeatureSet.WITH_NETWORK]
    )
    def test_all_feature_sets_resolvable(self, name: FeatureSet) -> None:
        spec = get_feature_set(name)
        assert spec.name == name

    def test_resolve_by_lowercase_string(self) -> None:
        spec = get_feature_set("baseline")
        assert spec.name == FeatureSet.BASELINE

    def test_extended_adds_engineered_flags(self) -> None:
        spec = get_feature_set(FeatureSet.EXTENDED)
        assert spec.add_cyclic_temporal
        assert spec.add_cross_route
        assert not spec.add_network_encodings

    def test_with_network_adds_encodings(self) -> None:
        spec = get_feature_set(FeatureSet.WITH_NETWORK)
        assert spec.add_network_encodings


@pytest.mark.integration
class TestRealDatasetNoLeakage:
    """Run the same guarantees against the real 220 000-row parquet."""

    def test_no_leakage_on_real_data(self, real_frame: pd.DataFrame) -> None:
        cols = get_feature_columns(real_frame)
        assert_no_leakage(cols)
        assert len(cols) > 20, f"only {len(cols)} features selected — schema likely broken"
