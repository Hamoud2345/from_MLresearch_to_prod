"""Couche modèles : des forecasters derrière une interface commune, créés via un registry."""

from __future__ import annotations

from collections.abc import Callable

from ..config import settings
from .base import Forecaster, Prediction
from .baseline import SeasonalNaiveForecaster
from .gbm import LightGBMForecaster

#: nom -> constructeur. un nouveau modèle s'ajoute ici sans toucher au reste
_REGISTRY: dict[str, Callable[..., Forecaster]] = {
    SeasonalNaiveForecaster.name: SeasonalNaiveForecaster,
    LightGBMForecaster.name: LightGBMForecaster,
}


def register_model(name: str, factory: Callable[..., Forecaster]) -> None:
    """Ajoute un forecaster au registry."""
    _REGISTRY[name] = factory


def create_model(name: str | None = None, **kwargs) -> Forecaster:
    """Construit un forecaster à partir de son nom enregistré.

    Le code appelant demande ``"lightgbm"`` et récupère un objet prêt à l'emploi,
    sans connaître la classe concrète ni son constructeur.
    """
    name = name or settings.model_name
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_models() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "Forecaster",
    "Prediction",
    "SeasonalNaiveForecaster",
    "LightGBMForecaster",
    "create_model",
    "register_model",
    "available_models",
]
