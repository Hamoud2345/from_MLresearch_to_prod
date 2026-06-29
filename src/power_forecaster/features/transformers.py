"""Transformers concrets pour la prévision des prix day-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import TARGET, FeatureTransformer


class CalendarFeatures(FeatureTransformer):
    """Encodages calendaires cycliques pour la saisonnalité jour/semaine/année.

    L'heure et le jour de la semaine sont encodés en sin/cos pour que le modèle
    voie que 23h est juste à côté de 0h (un entier brut ne le dirait pas).
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
    """Prix passés. Le prix day-ahead est fortement autocorrélé.

    Lags de 24h, 48h et 168h (une semaine). On n'utilise que du passé, donc pas
    de fuite d'info future.
    """

    def __init__(self, lags: tuple[int, ...] = (24, 48, 168)) -> None:
        self.lags = lags

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for lag in self.lags:
            out[f"price_lag_{lag}"] = out[TARGET].shift(lag)
        return out


class RollingFeatures(FeatureTransformer):
    """Statistiques rolling des prix passés (tendance et volatilité locale)."""

    def __init__(self, windows: tuple[int, ...] = (24, 168)) -> None:
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # shift(1) pour que la fenêtre ne contienne que du strictement passé
        past = out[TARGET].shift(1)
        for w in self.windows:
            out[f"price_rollmean_{w}"] = past.rolling(w).mean()
            out[f"price_rollstd_{w}"] = past.rolling(w).std()
        return out


class FundamentalFeatures(FeatureTransformer):
    """Fondamentaux marché : conso et prod renouvelable pilotent le merit order.

    En vrai on utiliserait les prévisions de load/wind/solar (connues avant
    l'enchère). Ici on prend les valeurs réalisées à la place, c'est une
    simplification assumée.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["renewables"] = out["wind"] + out["solar"]
        out["residual_load"] = out["load"] - 400 * out["renewables"]
        return out


def default_transformers() -> list[FeatureTransformer]:
    """Jeu de features standard utilisé en training et en serving."""
    return [
        CalendarFeatures(),
        FundamentalFeatures(),
        LagFeatures(),
        RollingFeatures(),
    ]
