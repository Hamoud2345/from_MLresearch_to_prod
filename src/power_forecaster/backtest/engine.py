"""Moteur de backtest walk-forward.

On entraine sur le passe et on teste sur le futur, puis on fait glisser la
fenetre. Un split aleatoire ferait fuiter des infos du futur dans le train.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.base import TARGET, FeaturePipeline
from ..models.base import Forecaster
from . import metrics
from .strategy import DirectionalStrategy, TradingStrategy


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    rmse: float
    mae: float
    smape: float
    pnl: float
    sharpe: float


@dataclass
class BacktestReport:
    folds: list[FoldResult]
    cum_pnl: np.ndarray

    def aggregate(self) -> dict[str, float]:
        # moyenne des metriques d'accuracy, somme/annualisation pour le trading
        if not self.folds:
            return {}
        return {
            "rmse": float(np.mean([f.rmse for f in self.folds])),
            "mae": float(np.mean([f.mae for f in self.folds])),
            "smape": float(np.mean([f.smape for f in self.folds])),
            "total_pnl": float(np.sum([f.pnl for f in self.folds])),
            "sharpe": float(np.mean([f.sharpe for f in self.folds])),
            "max_drawdown": metrics.max_drawdown(self.cum_pnl),
            "n_folds": len(self.folds),
        }


class WalkForwardBacktester:
    """Fait glisser une fenetre train/test sur l'historique et score chaque fold.

    On passe une factory de modele (chaque fold reentraine un modele neuf) et une
    strategie, ce qui permet de tester n'importe quel modele avec n'importe quelle
    strategie sans toucher au moteur.
    """

    def __init__(
        self,
        pipeline: FeaturePipeline,
        train_window_days: int = 365,
        test_window_days: int = 30,
        strategy: TradingStrategy | None = None,
        reference_column: str = "price_lag_24",
    ) -> None:
        self.pipeline = pipeline
        self.train_window = train_window_days * 24
        self.test_window = test_window_days * 24
        self.strategy = strategy or DirectionalStrategy()
        self.reference_column = reference_column

    def run(self, raw: pd.DataFrame, model_factory) -> BacktestReport:
        data = self.pipeline.transform(raw).dropna().reset_index(drop=True)
        feature_cols = [c for c in data.columns if c not in {"timestamp", TARGET}]

        folds: list[FoldResult] = []
        pnl_stream: list[np.ndarray] = []

        start = 0
        fold_id = 0
        while start + self.train_window + self.test_window <= len(data):
            train = data.iloc[start : start + self.train_window]
            test_end = start + self.train_window + self.test_window
            test = data.iloc[start + self.train_window : test_end]

            model: Forecaster = model_factory()
            model.fit(train[feature_cols], train[TARGET])
            forecast = model.predict(test[feature_cols])

            y_true = test[TARGET].to_numpy()
            reference = test[self.reference_column].to_numpy()
            positions = self.strategy.positions(forecast, reference)
            pnl = positions * (y_true - reference)

            folds.append(
                FoldResult(
                    fold=fold_id,
                    train_start=train["timestamp"].iloc[0],
                    test_start=test["timestamp"].iloc[0],
                    test_end=test["timestamp"].iloc[-1],
                    rmse=metrics.rmse(y_true, forecast.median),
                    mae=metrics.mae(y_true, forecast.median),
                    smape=metrics.smape(y_true, forecast.median),
                    pnl=float(pnl.sum()),
                    sharpe=metrics.sharpe_ratio(pnl),
                )
            )
            pnl_stream.append(pnl)
            start += self.test_window
            fold_id += 1

        cum = np.cumsum(np.concatenate(pnl_stream)) if pnl_stream else np.array([])
        return BacktestReport(folds=folds, cum_pnl=cum)
