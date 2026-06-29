"""Feature-engineering layer."""

from __future__ import annotations

from .base import TARGET, FeaturePipeline, FeatureTransformer
from .transformers import (
    CalendarFeatures,
    FundamentalFeatures,
    LagFeatures,
    RollingFeatures,
    default_transformers,
)


def build_default_pipeline() -> FeaturePipeline:
    """Construct the standard feature pipeline used across the project."""
    return FeaturePipeline(default_transformers())


__all__ = [
    "TARGET",
    "FeaturePipeline",
    "FeatureTransformer",
    "CalendarFeatures",
    "FundamentalFeatures",
    "LagFeatures",
    "RollingFeatures",
    "default_transformers",
    "build_default_pipeline",
]
