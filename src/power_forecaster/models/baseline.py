"""Seasonal-naive baseline.

A credible ML project always ships a baseline: "the price tomorrow at hour h
equals the price yesterday at hour h". If the fancy model cannot beat this, it
is not worth deploying. Having the baseline behind the same interface makes the
comparison a one-liner in the backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster, Prediction


class SeasonalNaiveForecaster(Forecaster):
    """Predict price using the value 24h earlier (``price_lag_24``)."""

    name = "seasonal_naive"

    def __init__(self, lag_column: str = "price_lag_24") -> None:
        self.lag_column = lag_column
        self._residual_std: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaiveForecaster:
        # "Training" only calibrates the uncertainty band from in-sample errors.
        preds = X[self.lag_column].to_numpy()
        self._residual_std = float(np.nanstd(y.to_numpy() - preds)) or 1.0
        return self

    def predict(self, X: pd.DataFrame) -> Prediction:
        median = X[self.lag_column].to_numpy(dtype=float)
        band = 1.2816 * self._residual_std  # ~10/90 percentile of a normal.
        return Prediction(median=median, lower=median - band, upper=median + band)
