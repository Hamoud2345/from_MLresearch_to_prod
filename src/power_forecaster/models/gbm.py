"""LightGBM gradient-boosting forecaster with quantile prediction bands.

Gradient boosting is the workhorse for tabular price forecasting: it captures
non-linear interactions (e.g. price explodes only when residual load is high AND
wind is low) without manual feature crossing. We fit three models — one per
quantile — so the output is a full prediction interval rather than a single
number, which is what a risk-aware trading desk needs.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import Forecaster, Prediction


class LightGBMForecaster(Forecaster):
    """Quantile gradient boosting (lower / median / upper)."""

    name = "lightgbm"

    def __init__(
        self,
        quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.quantiles = quantiles
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "random_state": random_state,
            "objective": "quantile",
            "verbose": -1,
        }
        self._models: dict[float, lgb.LGBMRegressor] = {}
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMForecaster:
        self.feature_names_ = list(X.columns)
        for q in self.quantiles:
            model = lgb.LGBMRegressor(alpha=q, **self.params)
            model.fit(X, y)
            self._models[q] = model
        return self

    def predict(self, X: pd.DataFrame) -> Prediction:
        if not self._models:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = X[self.feature_names_]
        lo, med, hi = self.quantiles
        # Sort columns to guarantee lower <= median <= upper (quantile crossing).
        preds = np.vstack([self._models[q].predict(X) for q in self.quantiles])
        preds = np.sort(preds, axis=0)
        return Prediction(
            median=preds[1],
            lower=preds[0],
            upper=preds[2],
            quantiles=(lo, med, hi),
        )

    def feature_importance(self) -> dict[str, float]:
        """Average gain importance across the quantile models (for monitoring)."""
        if not self._models:
            return {}
        importances = np.mean(
            [m.booster_.feature_importance(importance_type="gain") for m in self._models.values()],
            axis=0,
        )
        pairs = zip(self.feature_names_, importances, strict=True)
        return dict(sorted(pairs, key=lambda x: -x[1]))
