"""Tests for the feature-engineering layer."""

from __future__ import annotations

import numpy as np

from power_forecaster.features import build_default_pipeline
from power_forecaster.features.transformers import LagFeatures


def test_pipeline_adds_expected_features(raw_data):
    pipe = build_default_pipeline()
    out = pipe.transform(raw_data)
    for col in ["hour_sin", "dow_cos", "renewables", "price_lag_24", "price_rollmean_24"]:
        assert col in out.columns


def test_lag_feature_has_no_lookahead(raw_data):
    out = LagFeatures(lags=(24,)).transform(raw_data)
    # The lag column at row i must equal the price 24 rows earlier.
    np.testing.assert_allclose(
        out["price_lag_24"].iloc[24:].to_numpy(),
        raw_data["price"].iloc[:-24].to_numpy(),
    )


def test_feature_names_exclude_target_and_timestamp(raw_data):
    names = build_default_pipeline().feature_names(raw_data)
    assert "price" not in names
    assert "timestamp" not in names
