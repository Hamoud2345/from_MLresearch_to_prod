"""Tests for the model layer."""

from __future__ import annotations

import numpy as np
import pytest

from power_forecaster.features.base import TARGET
from power_forecaster.models import LightGBMForecaster, available_models, create_model
from power_forecaster.models.base import Forecaster


def _xy(features):
    cols = [c for c in features.columns if c not in {"timestamp", TARGET}]
    return features[cols], features[TARGET]


def test_registry_lists_models():
    assert {"lightgbm", "seasonal_naive"} <= set(available_models())


def test_create_unknown_model_raises():
    with pytest.raises(ValueError):
        create_model("nope")


@pytest.mark.parametrize("name", ["seasonal_naive", "lightgbm"])
def test_forecaster_fit_predict_interval_ordering(features, name):
    X, y = _xy(features)
    model = create_model(name)
    model.fit(X, y)
    pred = model.predict(X)
    assert len(pred.median) == len(X)
    # Prediction interval must be ordered: lower <= median <= upper.
    assert np.all(pred.lower <= pred.median + 1e-6)
    assert np.all(pred.median <= pred.upper + 1e-6)


def test_lightgbm_beats_naive_on_train(features):
    X, y = _xy(features)
    gbm = LightGBMForecaster(n_estimators=100).fit(X, y)
    naive = create_model("seasonal_naive").fit(X, y)
    rmse_gbm = np.sqrt(np.mean((y.to_numpy() - gbm.predict(X).median) ** 2))
    rmse_naive = np.sqrt(np.mean((y.to_numpy() - naive.predict(X).median) ** 2))
    assert rmse_gbm < rmse_naive


def test_model_save_load_roundtrip(features, tmp_path):
    X, y = _xy(features)
    model = create_model("lightgbm", n_estimators=50).fit(X, y)
    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = Forecaster.load(path)
    np.testing.assert_allclose(model.predict(X).median, loaded.predict(X).median)
