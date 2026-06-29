"""Transformers de feature engineering, composables.

Chaque transformer fait une seule chose et partage la même interface
(``transform``), et ``FeaturePipeline`` les enchaîne dans l'ordre. Ajouter une
feature revient à écrire une classe, sans toucher au reste.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

TARGET = "price"


class FeatureTransformer(ABC):
    """Transformation sans état : prend un DataFrame, le renvoie enrichi."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renvoie ``df`` avec les nouvelles colonnes de features."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


class FeaturePipeline:
    """Applique une liste ordonnée de FeatureTransformer."""

    def __init__(self, transformers: list[FeatureTransformer]) -> None:
        self.transformers = transformers

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for t in self.transformers:
            out = t.transform(out)
        return out

    def feature_names(self, df: pd.DataFrame) -> list[str]:
        """Colonnes produites par le pipeline, hors colonnes brutes et target."""
        transformed = self.transform(df)
        excluded = {"timestamp", TARGET}
        return [c for c in transformed.columns if c not in excluded]
