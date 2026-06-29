"""Training pipeline with MLflow experiment tracking and model registry.

This is the heart of the MLOps story:
  1. Load data through the abstract DataSource.
  2. Build features through the composable pipeline.
  3. Backtest (walk-forward) to get honest, leakage-free metrics.
  4. Fit the final model on the full history.
  5. Log params, metrics, feature importances and the serialized model + a
     reference feature sample (for drift monitoring) to MLflow, and register the
     model under a named version so serving can always pull "the latest".
"""

from __future__ import annotations

import json
from dataclasses import asdict

import mlflow
import pandas as pd

from ..backtest.engine import WalkForwardBacktester
from ..backtest.strategy import DirectionalStrategy
from ..config import settings
from ..data import get_data_source
from ..features import build_default_pipeline
from ..features.base import TARGET
from ..models import create_model
from ..models.base import Forecaster


class _ForecasterPyfunc(mlflow.pyfunc.PythonModel):
    """Wrap our custom Forecaster so MLflow can log, version and serve it.

    Logging through ``mlflow.pyfunc`` is the idiomatic way to register a model:
    it captures the environment, gives the model a versioned entry in the
    registry, and exposes a standard ``predict`` interface usable by MLflow's
    own serving tools — not just our FastAPI app.
    """

    def __init__(self, model: Forecaster) -> None:
        self._model = model

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG002
        pred = self._model.predict(model_input)
        return pd.DataFrame({"median": pred.median, "lower": pred.lower, "upper": pred.upper})


def train(model_name: str | None = None) -> dict:
    """Run the full training + evaluation + logging pipeline.

    Returns the aggregate backtest metrics so the CLI / CI can print or assert
    on them.
    """
    model_name = model_name or settings.model_name
    pipeline = build_default_pipeline()
    source = get_data_source(settings.data_source, seed=settings.random_seed)
    raw = source.load(settings.history_days)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(
            {
                "model_name": model_name,
                "data_source": settings.data_source,
                "history_days": settings.history_days,
                "train_window_days": settings.train_window_days,
                "test_window_days": settings.test_window_days,
            }
        )

        # --- Honest evaluation via walk-forward backtest --------------------
        backtester = WalkForwardBacktester(
            pipeline=pipeline,
            train_window_days=settings.train_window_days,
            test_window_days=settings.test_window_days,
            strategy=DirectionalStrategy(),
        )
        report = backtester.run(raw, lambda: create_model(model_name, **_model_kwargs(model_name)))
        agg = report.aggregate()
        mlflow.log_metrics(agg)

        # --- Fit the production model on the full available history ---------
        features = pipeline.transform(raw).dropna().reset_index(drop=True)
        feature_cols = [c for c in features.columns if c not in {"timestamp", TARGET}]
        model = create_model(model_name, **_model_kwargs(model_name))
        model.fit(features[feature_cols], features[TARGET])

        # Persist locally for the API, and to MLflow for lineage/rollback.
        settings.model_path().parent.mkdir(parents=True, exist_ok=True)
        model.save(settings.model_path())

        # Reference sample of the target — used by the API to detect drift.
        reference_path = settings.artifacts_dir / "reference.json"
        reference_path.write_text(
            json.dumps({"price": features[TARGET].tail(24 * 30).round(3).tolist()})
        )

        mlflow.log_artifact(str(reference_path))
        if hasattr(model, "feature_importance"):
            mlflow.log_dict(model.feature_importance(), "feature_importance.json")

        mlflow.log_dict({"folds": [asdict(f) | _stringify_ts(f) for f in report.folds]},
                        "backtest_folds.json")

        # Log + register the model as an MLflow pyfunc (versioned in the registry).
        signature = mlflow.models.infer_signature(
            features[feature_cols].head(),
            _ForecasterPyfunc(model).predict(None, features[feature_cols].head()),
        )
        _register(model, signature, features[feature_cols].head())
        return agg


def _model_kwargs(model_name: str) -> dict:
    if model_name == "lightgbm":
        return {"quantiles": settings.quantiles, "random_state": settings.random_seed}
    return {}


def _stringify_ts(fold) -> dict:
    return {
        "train_start": str(fold.train_start),
        "test_start": str(fold.test_start),
        "test_end": str(fold.test_end),
    }


def _register(model: Forecaster, signature, input_example) -> None:
    """Log the model as a pyfunc and register a new version in the registry."""
    try:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=_ForecasterPyfunc(model),
            signature=signature,
            input_example=input_example,
            registered_model_name=settings.registered_model_name,
        )
    except Exception as exc:  # registry is optional in some backends
        print(f"[mlflow] model registry skipped: {exc}")


if __name__ == "__main__":  # pragma: no cover
    metrics = train()
    print(json.dumps(metrics, indent=2))
