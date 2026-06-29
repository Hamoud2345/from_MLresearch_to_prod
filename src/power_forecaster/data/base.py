"""Abstract data-source contract.


We depend on an *abstraction* (``DataSource``) rather than on a concrete
provider. This is the Dependency Inversion Principle: the training and serving
code asks for "some source of price data" and does not care whether that data
comes from a synthetic generator (offline, deterministic, great for CI) or from
the real ENTSO-E API. Swapping one for the other is a one-line change and needs
zero modification to the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

#: Canonical schema every DataSource must return.
REQUIRED_COLUMNS = ["timestamp", "price", "load", "wind", "solar"]


class DataSource(ABC):
    """A provider of hourly day-ahead market data.

    Concrete implementations only need to fill :meth:`_fetch`. The public
    :meth:`load` method validates the schema so the rest of the system can trust
    its input — a small but important production guarantee.
    """

    @abstractmethod
    def _fetch(self, history_days: int) -> pd.DataFrame:
        """Return raw hourly data with at least :data:`REQUIRED_COLUMNS`."""
        raise NotImplementedError

    def load(self, history_days: int) -> pd.DataFrame:
        """Fetch, validate and normalise the data frame."""
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
