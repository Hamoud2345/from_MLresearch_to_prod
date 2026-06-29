"""Interface en ligne de commande, point d'entree unique du pipeline.

    ppf train                 # train + backtest + log MLflow
    ppf backtest --model ...  # compare un modele a la baseline
    ppf serve                 # lance l'API de prediction
    ppf sample-request        # affiche un payload /predict pret a l'emploi
"""

from __future__ import annotations

import json

import typer

from .backtest.engine import WalkForwardBacktester
from .backtest.strategy import DirectionalStrategy
from .config import settings
from .data import get_data_source
from .features import build_default_pipeline
from .models import available_models, create_model

app = typer.Typer(add_completion=False, help="Prevision des prix day-ahead (demo MLOps).")


@app.command()
def train(model: str = typer.Option(settings.model_name, help="Modele a entrainer.")) -> None:
    """Entraine le modele, lance un backtest walk-forward et log le tout dans MLflow."""
    from .training import train as run_training

    metrics = run_training(model)
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def backtest(
    model: str = typer.Option(settings.model_name, help="Modele a evaluer."),
    compare_baseline: bool = typer.Option(True, help="Backtest aussi la baseline seasonal-naive."),
) -> None:
    """Backtest un ou plusieurs modeles et affiche accuracy + metriques de trading."""
    pipeline = build_default_pipeline()
    source = get_data_source(settings.data_source, seed=settings.random_seed)
    raw = source.load(settings.history_days)
    bt = WalkForwardBacktester(
        pipeline=pipeline,
        train_window_days=settings.train_window_days,
        test_window_days=settings.test_window_days,
        strategy=DirectionalStrategy(),
    )

    names = [model] + (["seasonal_naive"] if compare_baseline and model != "seasonal_naive" else [])
    results = {}
    for name in names:
        report = bt.run(raw, lambda n=name: create_model(n))
        results[name] = report.aggregate()
    typer.echo(json.dumps(results, indent=2))


@app.command()
def models() -> None:
    """Liste les modeles de prevision enregistres."""
    typer.echo("\n".join(available_models()))


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Lance le service de prediction FastAPI avec uvicorn."""
    import uvicorn

    uvicorn.run("power_forecaster.serving.api:app", host=host, port=port, reload=reload)


@app.command("sample-request")
def sample_request(hours: int = 24) -> None:
    """Affiche un payload /predict valide, construit sur des donnees synthetiques."""
    raw = get_data_source("synthetic", seed=settings.random_seed).load(40)
    pipeline = build_default_pipeline()
    feats = pipeline.transform(raw).dropna().tail(hours)
    obs = [
        {
            "timestamp": ts.isoformat(),
            "load": float(row.load),
            "wind": float(row.wind),
            "solar": float(row.solar),
            "price_lag_24": float(row.price_lag_24),
            "price_lag_48": float(row.price_lag_48),
            "price_lag_168": float(row.price_lag_168),
        }
        for ts, row in zip(feats["timestamp"], feats.itertuples(), strict=True)
    ]
    typer.echo(json.dumps({"observations": obs}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
