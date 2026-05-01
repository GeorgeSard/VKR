"""Tests for the dataset validator."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validate import validate


class TestValidate:
    def test_clean_synthetic_passes(self, synthetic_frame: pd.DataFrame) -> None:
        report = validate(synthetic_frame, strict=False)
        # Synthetic frame is intentionally minimal (column count warning is OK).
        assert report.ok, f"unexpected errors: {report.errors}"

    def test_duplicate_flight_id_fails(self, synthetic_frame: pd.DataFrame) -> None:
        df = pd.concat([synthetic_frame, synthetic_frame.iloc[:5]], ignore_index=True)
        report = validate(df, strict=False)
        assert any("duplicate" in e.lower() for e in report.errors)

    def test_unexpected_year_fails(self, synthetic_frame: pd.DataFrame) -> None:
        df = synthetic_frame.copy()
        df.loc[0, "year"] = 2030
        report = validate(df, strict=False)
        assert any("year" in e.lower() for e in report.errors)

    def test_strict_mode_raises(self, synthetic_frame: pd.DataFrame) -> None:
        df = pd.concat([synthetic_frame, synthetic_frame.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="validation failed"):
            validate(df, strict=True)


@pytest.mark.integration
class TestValidateRealDataset:
    def test_real_dataset_passes(self, real_frame: pd.DataFrame) -> None:
        report = validate(real_frame, strict=False)
        # We tolerate warnings, not errors.
        assert report.ok, "validation errors on real data:\n  - " + "\n  - ".join(report.errors)
