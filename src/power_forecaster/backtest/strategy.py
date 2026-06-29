"""Strategies de trading : transforment une prevision de prix en position.

Le backtester ne sait pas comment la position est decidee, il demande juste une
position a la strategie.

Marche jouet : chaque heure on peut etre long (+1), short (-1) ou flat (0) sur
une unite de power face a un prix de reference (ici le prix d'il y a 24h).
PnL = position * (realise - reference).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models.base import Prediction


class TradingStrategy(ABC):
    """Renvoie une position dans {-1, 0, +1} a partir d'une prevision et d'un prix de reference."""

    name: str = "base"

    @abstractmethod
    def positions(self, forecast: Prediction, reference_price: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DirectionalStrategy(TradingStrategy):
    """Long si la prevision est au-dessus du prix de reference, short sinon.

    ``threshold`` est une dead-band (en €/MWh) pour eviter de trader sur des
    micro-edges que les frais mangeraient.
    """

    name = "directional"

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def positions(self, forecast: Prediction, reference_price: np.ndarray) -> np.ndarray:
        edge = forecast.median - reference_price
        pos = np.where(edge > self.threshold, 1.0, np.where(edge < -self.threshold, -1.0, 0.0))
        return pos


class ConfidenceGatedStrategy(TradingStrategy):
    """Ne trade que si l'intervalle de prediction est etroit par rapport a l'edge.

    Une bande large = modele pas sur, donc on reste flat.
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
