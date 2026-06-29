"""Trading strategies that turn a price forecast into a position.

Design note (interview-ready):
A :class:`TradingStrategy` is, again, the Strategy pattern. The backtester does
not know *how* a position is decided — it only asks the strategy for one. We can
plug a naive directional strategy, a confidence-gated one, or anything else,
without changing the engine.

Toy market model: each hour we may go long (+1), short (-1) or flat (0) one unit
of power versus a reference price (here the 24h-ago price, a proxy for the price
already known when the auction closes). PnL = position * (realised - reference).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models.base import Prediction


class TradingStrategy(ABC):
    """Map a forecast and a reference price to a position in {-1, 0, +1}."""

    name: str = "base"

    @abstractmethod
    def positions(self, forecast: Prediction, reference_price: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DirectionalStrategy(TradingStrategy):
    """Go long if the forecast is above the reference price, short otherwise.

    ``threshold`` is a dead-band (in €/MWh) avoiding trades on tiny edges that
    transaction costs would eat.
    """

    name = "directional"

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def positions(self, forecast: Prediction, reference_price: np.ndarray) -> np.ndarray:
        edge = forecast.median - reference_price
        pos = np.where(edge > self.threshold, 1.0, np.where(edge < -self.threshold, -1.0, 0.0))
        return pos


class ConfidenceGatedStrategy(TradingStrategy):
    """Trade only when the prediction interval is tight relative to the edge.

    A wide band means the model is unsure, so we stand aside. This expresses how
    a real desk uses uncertainty to size/skip trades.
    """

    name = "confidence_gated"

    def __init__(self, threshold: float = 1.0, max_relative_band: float = 1.5) -> None:
        self.threshold = threshold
        self.max_relative_band = max_relative_band

    def positions(self, forecast: Prediction, reference_price: np.ndarray) -> np.ndarray:
        edge = forecast.median - reference_price
        band = np.abs(forecast.upper - forecast.lower) + 1e-9
        confident = np.abs(edge) / band > (1.0 / self.max_relative_band)
        raw = np.where(edge > self.threshold, 1.0, np.where(edge < -self.threshold, -1.0, 0.0))
        return np.where(confident, raw, 0.0)
