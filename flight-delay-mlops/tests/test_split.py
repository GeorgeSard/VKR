"""CLAUDE.md rule 2: only time-based split, never random."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import SplitConfig, filter_delayed_only, time_based_split


class TestTimeBasedSplit:
    def test_default_split_uses_2023_2024_2025(self, synthetic_frame: pd.DataFrame) -> None:
        result = time_based_split(synthetic_frame)
        assert set(result.train["year"].unique()) == {2023}
        assert set(result.val["year"].unique()) == {2024}
        assert set(result.test["year"].unique()) == {2025}

    def test_no_overlap_between_splits(self, synthetic_frame: pd.DataFrame) -> None:
        result = time_based_split(synthetic_frame)
        train_ids = set(result.train["flight_id"])
        val_ids = set(result.val["flight_id"])
        test_ids = set(result.test["flight_id"])
        assert not (train_ids & val_ids), "train and val overlap"
        assert not (train_ids & test_ids), "train and test overlap"
        assert not (val_ids & test_ids), "val and test overlap"

    def test_drops_cancelled_by_default(self, synthetic_frame: pd.DataFrame) -> None:
        result = time_based_split(synthetic_frame)
        for split in (result.train, result.val, result.test):
            assert (split["cancellation_flag"] == 0).all()

    def test_keeps_cancelled_when_requested(self, synthetic_frame: pd.DataFrame) -> None:
        cfg = SplitConfig(drop_cancelled=False)
        result = time_based_split(synthetic_frame, cfg)
        total_cancelled = (synthetic_frame["cancellation_flag"] == 1).sum()
        kept_cancelled = sum(
            (split["cancellation_flag"] == 1).sum()
            for split in (result.train, result.val, result.test)
        )
        assert kept_cancelled == total_cancelled

    def test_overlapping_years_raise(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            SplitConfig(train_years=(2023, 2024), val_years=(2024,))
            # constructing config does not raise; the validator runs in time_based_split
            df = pd.DataFrame({"year": [2023, 2024], "cancellation_flag": [0, 0]})
            time_based_split(df, SplitConfig(train_years=(2023, 2024), val_years=(2024,)))

    def test_missing_year_in_data_raises(self) -> None:
        df = pd.DataFrame({"year": [2023, 2024], "cancellation_flag": [0, 0]})
        with pytest.raises(ValueError, match="missing"):
            time_based_split(df)

    def test_missing_year_column_raises(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="year"):
            time_based_split(df)

    def test_split_sizes_reported(self, synthetic_frame: pd.DataFrame) -> None:
        result = time_based_split(synthetic_frame)
        sizes = result.sizes()
        assert set(sizes.keys()) == {"train", "val", "test"}
        total = sum(sizes.values())
        # All three years should yield roughly equal sizes (after dropping cancellations).
        assert total > 0
        assert all(v > 0 for v in sizes.values())


class TestFilterDelayedOnly:
    def test_keeps_only_delayed(self, synthetic_frame: pd.DataFrame) -> None:
        non_cancelled = synthetic_frame[synthetic_frame["cancellation_flag"] == 0]
        delayed = filter_delayed_only(non_cancelled)
        assert (delayed["is_departure_delayed_15m"] == 1).all()

    def test_excludes_undelayed(self, synthetic_frame: pd.DataFrame) -> None:
        non_cancelled = synthetic_frame[synthetic_frame["cancellation_flag"] == 0]
        delayed = filter_delayed_only(non_cancelled)
        assert (delayed["is_departure_delayed_15m"] != 0).all()

    def test_raises_without_target(self) -> None:
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="is_departure_delayed_15m"):
            filter_delayed_only(df)


@pytest.mark.integration
class TestRealDatasetSplit:
    """Run split on the real 220 000-row dataset."""

    def test_real_split_sizes_reasonable(self, real_frame: pd.DataFrame) -> None:
        result = time_based_split(real_frame)
        sizes = result.sizes()
        # Each year should contribute >10 % of the (non-cancelled) total.
        total = sum(sizes.values())
        assert total > 100_000
        for key, n in sizes.items():
            assert n > total * 0.1, f"split '{key}' is suspiciously small: {n}/{total}"

    def test_drift_detectable_in_security_share(self, real_frame: pd.DataFrame) -> None:
        """Sanity: 2025 must show higher security share than 2023 (concept drift signal)."""
        moscow = {"SVO", "DME", "VKO"}
        delayed = real_frame[
            (real_frame["is_departure_delayed_15m"] == 1)
            & (real_frame["origin_iata"].astype(str).isin(moscow))
        ]
        share_2023 = (delayed[delayed["year"] == 2023]["probable_delay_cause"] == "security").mean()
        share_2025 = (delayed[delayed["year"] == 2025]["probable_delay_cause"] == "security").mean()
        assert share_2025 > share_2023, (
            f"expected drift: security share 2023={share_2023:.4f} < 2025={share_2025:.4f}"
        )
