"""Command-line interface — the single entry point for the whole pipeline.

    ppf train                 # train + backtest + log to MLflow
    ppf backtest --model ...  # compare a model against the baseline
    ppf serve                 # launch the prediction API
    ppf sample-request        # print a ready-to-use /predict payload
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

app = typer.Typer(add_completion=False, help="Day-ahead power price forecaster (MLOps demo).")


@app.command()
def train(model: str = typer.Option(settings.model_name, help="Model name to train.")) -> None:
    """Train the model, run a walk-forward backtest and log everything to MLflow."""
    from .training import train as run_training

    metrics = run_training(model)
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def backtest(
    model: str = typer.Option(settings.model_name, help="Model to evaluate."),
    compare_baseline: bool = typer.Option(True, help="Also backtest the seasonal-naive baseline."),
) -> None:
    """Backtest one or more models and print accuracy + trading metrics."""
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
    """List the registered forecasting models."""
    typer.echo("\n".join(available_models()))


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Launch the FastAPI prediction service with uvicorn."""
    import uvicorn

    uvicorn.run("power_forecaster.serving.api:app", host=host, port=port, reload=reload)


@app.command("sample-request")
def sample_request(hours: int = 24) -> None:
    """Print a valid /predict payload built from synthetic data (for demos/tests)."""
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
