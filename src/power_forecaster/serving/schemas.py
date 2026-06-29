"""Pydantic request/response models for the prediction API.

Pydantic gives us validation at the edge: a malformed request is rejected with a
clear 422 before it ever reaches the model. The schema also documents the API
automatically (OpenAPI / Swagger UI), which is exactly what you want when other
teams consume your service.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MarketObservation(BaseModel):
    """One hour of market context needed to forecast the next day-ahead price."""

    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of the hour.")
    load: float = Field(..., description="System load (MW).")
    wind: float = Field(..., ge=0, description="Wind generation (GW-equivalent index).")
    solar: float = Field(..., ge=0, description="Solar generation (GW-equivalent index).")
    price_lag_24: float = Field(..., description="Price 24h earlier (€/MWh).")
    price_lag_48: float = Field(..., description="Price 48h earlier (€/MWh).")
    price_lag_168: float = Field(..., description="Price 168h earlier (€/MWh).")


class PredictRequest(BaseModel):
    observations: list[MarketObservation] = Field(..., min_length=1, max_length=720)


class PricePrediction(BaseModel):
    timestamp: str
    median: float = Field(..., description="Central price forecast (€/MWh).")
    lower: float = Field(..., description="Lower bound of the prediction interval.")
    upper: float = Field(..., description="Upper bound of the prediction interval.")


class PredictResponse(BaseModel):
    model_name: str
    predictions: list[PricePrediction]
    drift_psi: float = Field(..., description="PSI of request prices vs training reference.")
    drift_alert: bool = Field(..., description="True if PSI exceeds the configured threshold.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
