"""Forecast-accuracy and trading-performance metrics."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent — robust to prices near zero."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-9
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Quantile (pinball) loss — scores the calibration of a prediction band."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def sharpe_ratio(pnl: np.ndarray, periods_per_year: int = 24 * 365) -> float:
    """Annualised Sharpe ratio of an hourly PnL stream."""
    if pnl.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * pnl.mean() / pnl.std())


def max_drawdown(cum_pnl: np.ndarray) -> float:
    """Largest peak-to-trough drop of the cumulative PnL curve."""
    running_max = np.maximum.accumulate(cum_pnl)
    return float(np.min(cum_pnl - running_max))
