"""Application FastAPI qui sert les previsions de prix day-ahead.

Ce qu'on expose : les probes /health et /ready pour l'orchestrateur, /predict
avec entrees/sorties validees par Pydantic et intervalles de prediction,
/metrics pour Prometheus, et un check de drift (PSI) a chaque requete, renvoye
dans la reponse et en metrique pour reperer un changement de regime du marche.
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
    description="Prevision horaire des prix day-ahead avec intervalles d'incertitude.",
)

# metriques Prometheus
PREDICTION_COUNT = Counter("ppf_predictions_total", "Nombre d'heures prévues.")
REQUEST_LATENCY = Histogram("ppf_request_latency_seconds", "Latence des requêtes de prédiction.")
DRIFT_GAUGE = Histogram("ppf_drift_psi", "PSI (Population Stability Index) par requête.")

app.mount("/metrics", make_asgi_app())  # endpoint de scrape Prometheus

predictor = Predictor()
_NOT_TRAINED = "Modele pas encore entraine. Lance d'abord l'entrainement."


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe : le process tourne."""
    return HealthResponse(status="ok", model_loaded=predictor.is_ready())


@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    """Readiness probe : le modele est present et chargeable."""
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail=_NOT_TRAINED)
    return HealthResponse(status="ready", model_loaded=True, model_name=predictor.model_name)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Prevoit les prix day-ahead pour les heures fournies."""
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail=_NOT_TRAINED)

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

    # drift : on compare les prix de la requete a la reference d'entrainement
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
