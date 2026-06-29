"""Tests du backtest : moteur, strategies et metriques."""

from __future__ import annotations

import numpy as np

from power_forecaster.backtest import DirectionalStrategy, WalkForwardBacktester, metrics
from power_forecaster.features import build_default_pipeline
from power_forecaster.models import create_model
from power_forecaster.models.base import Prediction


def test_metrics_basic():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.rmse(y, y) == 0.0
    assert metrics.mae(y, y) == 0.0
    assert metrics.sharpe_ratio(np.zeros(10)) == 0.0


def test_directional_strategy_signs():
    forecast = Prediction(
        median=np.array([10.0, 5.0, 7.0]),
        lower=np.array([9.0, 4.0, 6.0]),
        upper=np.array([11.0, 6.0, 8.0]),
    )
    reference = np.array([8.0, 8.0, 7.0])
    pos = DirectionalStrategy(threshold=1.0).positions(forecast, reference)
    assert pos.tolist() == [1.0, -1.0, 0.0]  # long, short, flat (dans la dead-band)


def test_walkforward_runs_and_reports(raw_data):
    bt = WalkForwardBacktester(
        pipeline=build_default_pipeline(),
        train_window_days=30,
        test_window_days=15,
    )
    report = bt.run(raw_data, lambda: create_model("seasonal_naive"))
    agg = report.aggregate()
    assert agg["n_folds"] >= 1
    assert "rmse" in agg and "total_pnl" in agg and "sharpe" in agg
    assert len(report.cum_pnl) > 0
