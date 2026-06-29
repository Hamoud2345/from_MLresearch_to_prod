"""Interface commune des sources de données.

Le reste du code dépend de l'abstraction ``DataSource`` et pas d'un fournisseur
précis, ce qui permet de basculer du générateur synthétique à l'API ENTSO-E sans
toucher au pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

# colonnes attendues en sortie de toute source
REQUIRED_COLUMNS = ["timestamp", "price", "load", "wind", "solar"]


class DataSource(ABC):
    """Source de données horaires du marché day-ahead.

    Les implémentations n'ont qu'à remplir :meth:`_fetch`. C'est ``load`` qui
    valide le schéma, comme ça le reste du code peut faire confiance aux données.
    """

    @abstractmethod
    def _fetch(self, history_days: int) -> pd.DataFrame:
        """Renvoie les données horaires brutes, avec au moins REQUIRED_COLUMNS."""
        raise NotImplementedError

    def load(self, history_days: int) -> pd.DataFrame:
        """Récupère, valide et normalise le DataFrame."""
        df = self._fetch(history_days)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{type(self).__name__} is missing columns: {sorted(missing)}")
        df = df.loc[:, REQUIRED_COLUMNS].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        if df["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps detected in data source.")
        return df

