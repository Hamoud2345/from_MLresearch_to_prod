"""Couche d'accès aux données : plusieurs sources derrière une même interface."""

from __future__ import annotations

from .base import REQUIRED_COLUMNS, DataSource
from .synthetic import SyntheticDataSource


def get_data_source(name: str, seed: int = 42) -> DataSource:
    """Renvoie la bonne implémentation de DataSource selon son nom."""
    name = name.lower()
    if name == "synthetic":
        return SyntheticDataSource(seed=seed)
    if name == "entsoe":
        from .entsoe import EntsoeDataSource  # paresseux : dépendance optionnelle

        return EntsoeDataSource()
    raise ValueError(f"Unknown data source: {name!r}. Use 'synthetic' or 'entsoe'.")


__all__ = ["DataSource", "SyntheticDataSource", "REQUIRED_COLUMNS", "get_data_source"]
