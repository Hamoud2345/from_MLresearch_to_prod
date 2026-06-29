"""Couche de backtest : evaluation walk-forward et strategies de trading."""

from __future__ import annotations

from .engine import BacktestReport, FoldResult, WalkForwardBacktester
from .strategy import ConfidenceGatedStrategy, DirectionalStrategy, TradingStrategy

__all__ = [
    "WalkForwardBacktester",
    "BacktestReport",
    "FoldResult",
    "TradingStrategy",
    "DirectionalStrategy",
    "ConfidenceGatedStrategy",
]
