"""Modeles Pydantic pour les requetes/reponses de l'API de prediction.

Pydantic valide en entree : une requete mal formee est rejetee avec un 422 avant
d'atteindre le modele. Le schema documente aussi l'API tout seul (OpenAPI /
Swagger UI).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MarketObservation(BaseModel):
    """Une heure de contexte marche, necessaire pour prevoir le prix day-ahead suivant."""

    timestamp: str = Field(..., description="Horodatage UTC de l'heure (ISO-8601).")
    load: float = Field(..., description="Consommation du systeme (MW).")
    wind: float = Field(..., ge=0, description="Production eolienne (indice equivalent GW).")
    solar: float = Field(..., ge=0, description="Production solaire (indice equivalent GW).")
    price_lag_24: float = Field(..., description="Prix il y a 24h (€/MWh).")
    price_lag_48: float = Field(..., description="Prix il y a 48h (€/MWh).")
    price_lag_168: float = Field(..., description="Prix il y a 168h (€/MWh).")


class PredictRequest(BaseModel):
    observations: list[MarketObservation] = Field(..., min_length=1, max_length=720)


class PricePrediction(BaseModel):
    timestamp: str
    median: float = Field(..., description="Prevision centrale du prix (€/MWh).")
    lower: float = Field(..., description="Borne basse de l'intervalle de prediction.")
    upper: float = Field(..., description="Borne haute de l'intervalle de prediction.")


class PredictResponse(BaseModel):
    model_name: str
    predictions: list[PricePrediction]
    drift_psi: float = Field(..., description="PSI des prix de la requete vs reference.")
    drift_alert: bool = Field(..., description="Vrai si le PSI depasse le seuil configure.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
