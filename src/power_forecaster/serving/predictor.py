"""Conteneur du modele cote API : charge le modele une fois et construit les features.

Au serving on recoit un contexte marche ponctuel (consommation, renouvelables et
quelques lags de prix) et pas tout l'historique, donc les stats rolling sont
approximees a partir des lags fournis. Le builder renvoie toujours exactement les
colonnes vues a l'entrainement (``feature_names_``), comme ca le contrat
entrainement/serving ne peut pas deriver en silence.
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
    """Wrapper autour du modele de prod, charge en lazy facon singleton."""

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
        """Reconstruit les colonnes de features du modele a partir des observations de l'API."""
        df = pd.DataFrame(observations)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df = CalendarFeatures().transform(df)
        df = FundamentalFeatures().transform(df)

        # stats rolling approximees depuis les lags dispo (proxy assume)
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
