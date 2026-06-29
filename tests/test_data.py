"""Tests for the data-access layer."""

from __future__ import annotations

import pandas as pd
import pytest

from power_forecaster.data import REQUIRED_COLUMNS, SyntheticDataSource, get_data_source
from power_forecaster.data.base import DataSource


def test_synthetic_source_schema_and_determinism():
    a = SyntheticDataSource(seed=7).load(30)
    b = SyntheticDataSource(seed=7).load(30)
    assert list(a.columns) == REQUIRED_COLUMNS
    assert len(a) == 30 * 24
    # Same seed -> identical data (reproducibility is a production guarantee).
    pd.testing.assert_frame_equal(a, b)


def test_timestamps_are_sorted_and_unique():
    df = SyntheticDataSource(seed=1).load(10)
    assert df["timestamp"].is_monotonic_increasing
    assert not df["timestamp"].duplicated().any()


def test_factory_returns_requested_source():
    assert isinstance(get_data_source("synthetic"), SyntheticDataSource)
    with pytest.raises(ValueError):
        get_data_source("does-not-exist")


def test_load_validates_missing_columns():
    class BadSource(DataSource):
        def _fetch(self, history_days: int) -> pd.DataFrame:
            return pd.DataFrame({"timestamp": pd.date_range("2024", periods=3, freq="h")})

    with pytest.raises(ValueError, match="missing columns"):
        BadSource().load(1)
