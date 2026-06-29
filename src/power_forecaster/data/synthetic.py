"""Générateur de données day-ahead synthétiques mais réalistes.

Tout tourne hors ligne et de façon déterministe (seed fixé), pratique pour les
tests et la CI. On reproduit les effets qui comptent vraiment : saisonnalité
jour/semaine, merit order (plus d'éolien/solaire fait baisser le prix), pics de
prix liés à la demande et bruit à queue lourde.

Même interface ``DataSource`` que la vraie source ENTSO-E, donc interchangeable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DataSource


class SyntheticDataSource(DataSource):
    """Génère des données horaires de marché de l'électricité."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _fetch(self, history_days: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = history_days * 24
        # date de fin fixe pour rester reproductible
        index = pd.date_range(end="2025-01-01", periods=n, freq="h", tz="UTC")

        hour = index.hour.to_numpy()
        dow = index.dayofweek.to_numpy()
        doy = index.dayofyear.to_numpy()

        # renouvelables : le solaire suit le soleil, l'éolien est un bruit autocorrélé
        solar = np.clip(np.sin((hour - 6) / 24 * 2 * np.pi), 0, None) * rng.uniform(20, 60, n)
        wind = np.abs(np.convolve(rng.normal(0, 1, n), np.ones(12) / 12, mode="same")) * 30

        # demande : plus forte en journée la semaine, avec un effet saisonnier hiver/été
        daily = 1.0 + 0.35 * np.sin((hour - 8) / 24 * 2 * np.pi)
        weekly = np.where(dow < 5, 1.0, 0.85)
        seasonal = 1.0 + 0.2 * np.cos((doy - 15) / 365 * 2 * np.pi)
        load = 50_000 * daily * weekly * seasonal + rng.normal(0, 1_500, n)

        # le prix est piloté par la demande résiduelle (load moins renouvelables)
        residual = load - 400 * (wind + solar)
        price = 20 + 0.0011 * residual - 0.15 * (wind + solar)
        # pics à queue lourde quand la demande résiduelle est forte (prix de rareté)
        spike = rng.gamma(1.2, 1.0, n) * (residual > residual.mean() + residual.std())
        price = price + 8 * spike + rng.normal(0, 3, n)
        price = np.maximum(price, -50)  # les prix peuvent être négatifs

        return pd.DataFrame(
            {
                "timestamp": index,
                "price": price.round(2),
                "load": load.round(0),
                "wind": wind.round(1),
                "solar": solar.round(1),
            }
        )
