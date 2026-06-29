"""Synthetic but *realistic* day-ahead market generator.

Why synthetic data?
  - It runs fully offline and deterministically (seeded), so unit tests and CI
    never depend on a flaky external API or secret credentials.
  - It reproduces the structural effects a power trader actually cares about:
    daily/weekly seasonality, a merit-order effect (more wind & solar push the
    price down), load-driven price spikes, and heavy-tailed noise.

The generator is intentionally a drop-in replacement for the real ENTSO-E feed:
both expose the exact same ``DataSource`` interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DataSource


class SyntheticDataSource(DataSource):
    """Procedurally generate hourly electricity-market data."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _fetch(self, history_days: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = history_days * 24
        # Anchor the series to a fixed end date for full reproducibility.
        index = pd.date_range(end="2025-01-01", periods=n, freq="h", tz="UTC")

        hour = index.hour.to_numpy()
        dow = index.dayofweek.to_numpy()
        doy = index.dayofyear.to_numpy()

        # Renewable generation: solar follows the sun, wind is autocorrelated noise.
        solar = np.clip(np.sin((hour - 6) / 24 * 2 * np.pi), 0, None) * rng.uniform(20, 60, n)
        wind = np.abs(np.convolve(rng.normal(0, 1, n), np.ones(12) / 12, mode="same")) * 30

        # Demand (load): higher on weekday daytime, seasonal winter/summer peaks.
        daily = 1.0 + 0.35 * np.sin((hour - 8) / 24 * 2 * np.pi)
        weekly = np.where(dow < 5, 1.0, 0.85)
        seasonal = 1.0 + 0.2 * np.cos((doy - 15) / 365 * 2 * np.pi)
        load = 50_000 * daily * weekly * seasonal + rng.normal(0, 1_500, n)

        # Price: residual demand (load minus renewables) drives the merit order.
        residual = load - 400 * (wind + solar)
        price = 20 + 0.0011 * residual - 0.15 * (wind + solar)
        # Heavy-tailed spikes during high residual demand (scarcity pricing).
        spike = rng.gamma(1.2, 1.0, n) * (residual > residual.mean() + residual.std())
        price = price + 8 * spike + rng.normal(0, 3, n)
        price = np.maximum(price, -50)  # power prices can go negative.

        return pd.DataFrame(
            {
                "timestamp": index,
                "price": price.round(2),
                "load": load.round(0),
                "wind": wind.round(1),
                "solar": solar.round(1),
            }
        )
