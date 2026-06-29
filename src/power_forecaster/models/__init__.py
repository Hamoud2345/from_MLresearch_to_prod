"""Model layer: forecasters behind a common interface, built via a registry."""

from __future__ import annotations

from collections.abc import Callable

from ..config import settings
from .base import Forecaster, Prediction
from .baseline import SeasonalNaiveForecaster
from .gbm import LightGBMForecaster

#: Name -> constructor. New models register here without touching call sites.
_REGISTRY: dict[str, Callable[..., Forecaster]] = {
    SeasonalNaiveForecaster.name: SeasonalNaiveForecaster,
    LightGBMForecaster.name: LightGBMForecaster,
}


def register_model(name: str, factory: Callable[..., Forecaster]) -> None:
    """Add a new forecaster to the registry (extension point)."""
    _REGISTRY[name] = factory


def create_model(name: str | None = None, **kwargs) -> Forecaster:
    """Factory: build a forecaster by registered name.

    This is the Factory pattern — callers ask for ``"lightgbm"`` and get a ready
    object, decoupled from the concrete class and its constructor signature.
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
