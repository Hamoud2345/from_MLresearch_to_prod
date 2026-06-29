"""Data-access layer: pluggable market-data sources behind one interface."""

from __future__ import annotations

from .base import REQUIRED_COLUMNS, DataSource
from .synthetic import SyntheticDataSource


def get_data_source(name: str, seed: int = 42) -> DataSource:
    """Factory selecting a :class:`DataSource` implementation by name.

    Keeping construction in one factory means callers never hard-code a concrete
    class — they ask for ``"synthetic"`` or ``"entsoe"`` and get the right object.
    """
    name = name.lower()
    if name == "synthetic":
        return SyntheticDataSource(seed=seed)
    if name == "entsoe":
        from .entsoe import EntsoeDataSource  # lazy: optional dependency

        return EntsoeDataSource()
    raise ValueError(f"Unknown data source: {name!r}. Use 'synthetic' or 'entsoe'.")


__all__ = ["DataSource", "SyntheticDataSource", "REQUIRED_COLUMNS", "get_data_source"]
