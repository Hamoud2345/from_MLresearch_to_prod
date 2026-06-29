"""Interface commune des modèles de prévision.

Tous les modèles (du baseline tout bête au gradient boosting) exposent la même
interface Forecaster : fit / predict / save / load. Du coup le backtest, le
script d'entraînement et l'API ne dépendent que de ça, on peut comparer ou
remplacer un modèle sans rien casser ailleurs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass
class Prediction:
    """Prévision probabiliste : estimation centrale + borne basse/haute.

    L'intervalle sert pour le trading, on dimensionne les positions selon la
    confiance du modèle et pas juste selon la valeur centrale.
    """

    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    quantiles: tuple[float, float, float] = field(default=(0.1, 0.5, 0.9))


class Forecaster(ABC):
    """Interface commune à tous les modèles de prévision de prix."""

    #: nom lisible, défini par les sous-classes
    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Prediction:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Sauvegarde le modèle entraîné (joblib)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> Forecaster:
        return joblib.load(path)
