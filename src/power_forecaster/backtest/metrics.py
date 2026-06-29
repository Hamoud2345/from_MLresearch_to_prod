"""Metriques d'accuracy de prevision et de performance de trading."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE symetrique en %, robuste aux prix proches de zero."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-9
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball loss (quantile), mesure la calibration d'une bande de prediction."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def sharpe_ratio(pnl: np.ndarray, periods_per_year: int = 24 * 365) -> float:
    """Sharpe annualise sur un flux de PnL horaire."""
    if pnl.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * pnl.mean() / pnl.std())


def max_drawdown(cum_pnl: np.ndarray) -> float:
    """Plus grosse baisse pic-creux de la courbe de PnL cumule."""
    running_max = np.maximum.accumulate(cum_pnl)
    return float(np.min(cum_pnl - running_max))
