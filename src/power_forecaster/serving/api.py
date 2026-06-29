"""FastAPI application serving day-ahead price forecasts.

Production features baked in:
  - ``/health`` and ``/ready`` probes for orchestrators (Docker/K8s).
  - ``/predict`` with validated Pydantic I/O and prediction intervals.
  - ``/metrics`` exposing Prometheus counters/histograms for monitoring.
  - Inline data-drift check (PSI) on every request, surfaced in the response
    and as a metric, so a shifting market regime is observable, not silent.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from ..config import settings
from ..monitoring import population_stability_index
from .predictor import Predictor
from .schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PricePrediction,
)

app = FastAPI(
    title="Day-Ahead Power Price Forecaster",
    version="0.1.0",
    description="Forecast hourly day-ahead electricity prices with uncertainty bands.",
)

# --- Prometheus metrics ----------------------------------------------------
PREDICTION_COUNT = Counter("ppf_predictions_total", "Number of forecasted hours.")
REQUEST_LATENCY = Histogram("ppf_request_latency_seconds", "Prediction request latency.")
DRIFT_GAUGE = Histogram("ppf_drift_psi", "Population Stability Index per request.")

app.mount("/metrics", make_asgi_app())  # exposes Prometheus scrape endpoint.

predictor = Predictor()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe: the process is up."""
    return HealthResponse(status="ok", model_loaded=predictor.is_ready())


@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    """Readiness probe: the model artifact is present and loadable."""
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run training first.")
    return HealthResponse(status="ready", model_loaded=True, model_name=predictor.model_name)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Forecast day-ahead prices for the supplied hours."""
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run training first.")

    start = time.perf_counter()
    observations = [obs.model_dump() for obs in request.observations]
    forecast = predictor.predict(observations)

    predictions = [
        PricePrediction(
            timestamp=obs["timestamp"],
            median=round(float(m), 2),
            lower=round(float(lo), 2),
            upper=round(float(hi), 2),
        )
        for obs, m, lo, hi in zip(
            observations, forecast.median, forecast.lower, forecast.upper, strict=True
        )
    ]

    # --- Drift check: compare request prices to the training reference ------
    psi = 0.0
    reference = predictor.reference_prices
    if reference is not None and len(observations) >= 5:
        request_prices = [o["price_lag_24"] for o in observations]
        psi = population_stability_index(reference, request_prices)
    DRIFT_GAUGE.observe(psi)

    PREDICTION_COUNT.inc(len(predictions))
    REQUEST_LATENCY.observe(time.perf_counter() - start)

    return PredictResponse(
        model_name=predictor.model_name,
        predictions=predictions,
        drift_psi=round(psi, 4),
        drift_alert=psi > settings.drift_threshold,
    )
