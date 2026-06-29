"""Pipeline d'entrainement avec suivi MLflow et enregistrement du modele.

Les etapes : on charge les donnees, on construit les features, on fait un
backtest walk-forward pour avoir des metriques propres (sans fuite), puis on
entraine le modele final sur tout l'historique. On logue les params, metriques,
importances et le modele dans MLflow, plus un echantillon de reference pour le
suivi de drift. Le modele est enregistre dans le registry avec une version pour
que le serving recupere toujours la derniere.
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
    """Emballe notre Forecaster pour que MLflow puisse le loguer et le servir.

    On passe par pyfunc pour avoir une version dans le registry et une interface
    predict standard, utilisable aussi par les outils de serving de MLflow.
    """

    def __init__(self, model: Forecaster) -> None:
        self._model = model

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG002
        pred = self._model.predict(model_input)
        return pd.DataFrame({"median": pred.median, "lower": pred.lower, "upper": pred.upper})


def train(model_name: str | None = None) -> dict:
    """Lance tout le pipeline : entrainement, evaluation et logging.

    Renvoie les metriques agregees du backtest pour que la CLI ou la CI puissent
    les afficher ou tester dessus.
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

        # evaluation via backtest walk-forward
        backtester = WalkForwardBacktester(
            pipeline=pipeline,
            train_window_days=settings.train_window_days,
            test_window_days=settings.test_window_days,
            strategy=DirectionalStrategy(),
        )
        report = backtester.run(raw, lambda: create_model(model_name, **_model_kwargs(model_name)))
        agg = report.aggregate()
        mlflow.log_metrics(agg)

        # modele de prod entraine sur tout l'historique dispo
        features = pipeline.transform(raw).dropna().reset_index(drop=True)
        feature_cols = [c for c in features.columns if c not in {"timestamp", TARGET}]
        model = create_model(model_name, **_model_kwargs(model_name))
        model.fit(features[feature_cols], features[TARGET])

        # sauvegarde locale pour l'API, et dans MLflow pour le suivi / rollback
        settings.model_path().parent.mkdir(parents=True, exist_ok=True)
        model.save(settings.model_path())

        # echantillon de reference de la cible, sert a l'API pour detecter le drift
        reference_path = settings.artifacts_dir / "reference.json"
        reference_path.write_text(
            json.dumps({"price": features[TARGET].tail(24 * 30).round(3).tolist()})
        )

        mlflow.log_artifact(str(reference_path))
        if hasattr(model, "feature_importance"):
            mlflow.log_dict(model.feature_importance(), "feature_importance.json")

        mlflow.log_dict({"folds": [asdict(f) | _stringify_ts(f) for f in report.folds]},
                        "backtest_folds.json")

        # log + enregistrement du modele en pyfunc (versionne dans le registry)
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
    """Logue le modele en pyfunc et enregistre une nouvelle version dans le registry."""
    try:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=_ForecasterPyfunc(model),
            signature=signature,
            input_example=input_example,
            registered_model_name=settings.registered_model_name,
        )
    except Exception as exc:  # le registry n'est pas dispo sur tous les backends
        print(f"[mlflow] model registry skipped: {exc}")


if __name__ == "__main__":  # pragma: no cover
    metrics = train()
    print(json.dumps(metrics, indent=2))
