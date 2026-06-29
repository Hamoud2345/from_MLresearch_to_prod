"""Abstract forecaster contract.

Design note (interview-ready):
Every model — a trivial baseline or a tuned gradient-boosting machine — hides
behind the same :class:`Forecaster` interface (``fit`` / ``predict`` /
``save`` / ``load``). The backtester, the training script and the API all speak
to this interface only. That is the Strategy pattern: algorithms are
interchangeable behind a common contract, so we can compare or swap models
without touching the surrounding system.
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
    """A probabilistic forecast: central estimate plus a lower/upper band.

    The interval matters for trading: a desk sizes positions by how *confident*
    the model is, not only by the point forecast.
    """

    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    quantiles: tuple[float, float, float] = field(default=(0.1, 0.5, 0.9))


class Forecaster(ABC):
    """Common interface for all price-forecasting models."""

    #: Human-readable identifier, set by subclasses.
    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Prediction:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Persist the fitted model (joblib by default)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> Forecaster:
        return joblib.load(path)
