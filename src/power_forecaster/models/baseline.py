"""Baseline saisonnier naïf.

L'idée : le prix de demain à l'heure h = le prix d'hier à l'heure h. Ça sert de
point de comparaison, si le vrai modèle ne bat pas ça il ne sert à rien. Comme
il respecte la même interface, la comparaison dans le backtest est directe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster, Prediction


class SeasonalNaiveForecaster(Forecaster):
    """Prédit le prix avec la valeur d'il y a 24h (``price_lag_24``)."""

    name = "seasonal_naive"

    def __init__(self, lag_column: str = "price_lag_24") -> None:
        self.lag_column = lag_column
        self._residual_std: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaiveForecaster:
        # ici le "fit" calibre juste la largeur de la bande à partir des erreurs in-sample
        preds = X[self.lag_column].to_numpy()
        self._residual_std = float(np.nanstd(y.to_numpy() - preds)) or 1.0
        return self

    def predict(self, X: pd.DataFrame) -> Prediction:
        median = X[self.lag_column].to_numpy(dtype=float)
        band = 1.2816 * self._residual_std  # quantiles ~10/90 d'une loi normale
        return Prediction(median=median, lower=median - band, upper=median + band)
