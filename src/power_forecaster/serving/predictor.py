"""Model holder used by the API: loads the model once and builds serving features.

At serving time we receive point-in-time market context (load, renewables and a
few price lags) rather than a full history, so rolling-window statistics are
approximated from the supplied lags. The builder always emits exactly the
columns the model was trained on (``feature_names_``), so the contract between
training and serving cannot silently drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings
from ..features.transformers import CalendarFeatures, FundamentalFeatures
from ..models.base import Forecaster, Prediction


class Predictor:
    """Lazy-loaded singleton-style wrapper around the production model."""

    def __init__(self) -> None:
        self._model: Forecaster | None = None
        self._reference: np.ndarray | None = None

    @property
    def model(self) -> Forecaster:
        if self._model is None:
            self._model = Forecaster.load(settings.model_path())
        return self._model

    @property
    def model_name(self) -> str:
        return getattr(self.model, "name", "unknown")

    @property
    def reference_prices(self) -> np.ndarray | None:
        if self._reference is None:
            ref_path = settings.artifacts_dir / "reference.json"
            if ref_path.exists():
                self._reference = np.array(json.loads(ref_path.read_text())["price"])
        return self._reference

    def is_ready(self) -> bool:
        return settings.model_path().exists()

    def build_features(self, observations: list[dict]) -> pd.DataFrame:
        """Reconstruct the model's feature columns from API observations."""
        df = pd.DataFrame(observations)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df = CalendarFeatures().transform(df)
        df = FundamentalFeatures().transform(df)

        # Approximate rolling stats from the available lags (documented proxy).
        lags = df[["price_lag_24", "price_lag_48", "price_lag_168"]]
        df["price_rollmean_24"] = lags.mean(axis=1)
        df["price_rollmean_168"] = lags.mean(axis=1)
        df["price_rollstd_24"] = lags.std(axis=1).fillna(0.0)
        df["price_rollstd_168"] = lags.std(axis=1).fillna(0.0)

        expected = getattr(self.model, "feature_names_", None)
        if expected:
            for col in expected:
                if col not in df.columns:
                    df[col] = 0.0
            return df[expected]
        return df

    def predict(self, observations: list[dict]) -> Prediction:
        features = self.build_features(observations)
        return self.model.predict(features)


def get_predictor(path: Path | None = None) -> Predictor:  # pragma: no cover - DI hook
    return Predictor()
