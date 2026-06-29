"""Concrete feature transformers for day-ahead price forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import TARGET, FeatureTransformer


class CalendarFeatures(FeatureTransformer):
    """Cyclical calendar encodings — capture daily/weekly/yearly seasonality.

    Hour and day-of-week are encoded as sine/cosine pairs so the model sees that
    hour 23 is adjacent to hour 0 (a plain integer would not express that).
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ts = out["timestamp"].dt
        out["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
        out["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
        out["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)
        out["is_weekend"] = (ts.dayofweek >= 5).astype(int)
        out["month"] = ts.month
        return out


class LagFeatures(FeatureTransformer):
    """Past price values. The day-ahead price is strongly autocorrelated.

    We use lags of 24h, 48h and 168h (one week) — the prices a trader would
    naturally look back to. Only *past* information is used, so there is no
    look-ahead leakage.
    """

    def __init__(self, lags: tuple[int, ...] = (24, 48, 168)) -> None:
        self.lags = lags

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for lag in self.lags:
            out[f"price_lag_{lag}"] = out[TARGET].shift(lag)
        return out


class RollingFeatures(FeatureTransformer):
    """Rolling statistics of past prices (trend & local volatility)."""

    def __init__(self, windows: tuple[int, ...] = (24, 168)) -> None:
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # shift(1) guarantees the window contains only strictly-past observations.
        past = out[TARGET].shift(1)
        for w in self.windows:
            out[f"price_rollmean_{w}"] = past.rolling(w).mean()
            out[f"price_rollstd_{w}"] = past.rolling(w).std()
        return out


class FundamentalFeatures(FeatureTransformer):
    """Market fundamentals: load and renewable generation drive the merit order.

    In real desks the day-ahead forecast uses *forecast* load/wind/solar (known
    before the auction). Here we use the realised values as a stand-in, which is
    a documented simplification.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["renewables"] = out["wind"] + out["solar"]
        out["residual_load"] = out["load"] - 400 * out["renewables"]
        return out


def default_transformers() -> list[FeatureTransformer]:
    """The standard feature set used in training and serving."""
    return [
        CalendarFeatures(),
        FundamentalFeatures(),
        LagFeatures(),
        RollingFeatures(),
    ]
