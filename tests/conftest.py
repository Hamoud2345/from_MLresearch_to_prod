"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from power_forecaster.data import SyntheticDataSource
from power_forecaster.features import build_default_pipeline


@pytest.fixture(scope="session")
def raw_data() -> pd.DataFrame:
    return SyntheticDataSource(seed=0).load(history_days=120)


@pytest.fixture(scope="session")
def features(raw_data: pd.DataFrame) -> pd.DataFrame:
    return build_default_pipeline().transform(raw_data).dropna().reset_index(drop=True)
