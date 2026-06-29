"""Data-drift monitoring via the Population Stability Index (PSI).

In production a model silently rots when the live data distribution drifts away
from the training distribution (a new market regime, a cold winter, a price
cap...). PSI is a standard, cheap drift signal: we bucket a reference sample and
a live sample and measure how much the population shifted. A PSI above ~0.2
conventionally flags meaningful drift and is a trigger to retrain.
"""

from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Compute PSI between a reference and a current 1-D sample."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    # Floor to avoid division-by-zero / log(0) in empty buckets.
    ref_pct = np.clip(ref_pct, 1e-4, None)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
