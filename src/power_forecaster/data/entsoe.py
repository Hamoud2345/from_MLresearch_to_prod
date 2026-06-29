"""Real-data adapter for the ENTSO-E Transparency Platform.

This implementation is deliberately optional: it is only importable when the
``entsoe`` extra is installed and an API token is configured. Its mere presence
demonstrates the value of the ``DataSource`` abstraction — production code can
switch from synthetic to live market data by changing a single config value
(``PPF_DATA_SOURCE=entsoe``), with no change anywhere downstream.

The ENTSO-E platform (https://transparency.entsoe.eu) is the actual public
source many European energy-trading desks rely on for day-ahead prices, load
and generation.
"""

from __future__ import annotations

import os

import pandas as pd

from .base import DataSource


class EntsoeDataSource(DataSource):
    """Fetch hourly day-ahead price, load and renewable generation from ENTSO-E."""

    def __init__(self, country_code: str = "FR", api_token: str | None = None) -> None:
        self.country_code = country_code
        self.api_token = api_token or os.environ.get("ENTSOE_API_TOKEN")
        if not self.api_token:
            raise RuntimeError(
                "ENTSOE_API_TOKEN is required for EntsoeDataSource. "
                "Use the synthetic source for offline/CI runs."
            )

    def _fetch(self, history_days: int) -> pd.DataFrame:  # pragma: no cover - needs network
        from entsoe import EntsoePandasClient  # imported lazily: optional dependency

        client = EntsoePandasClient(api_key=self.api_token)
        end = pd.Timestamp.utcnow().floor("h")
        start = end - pd.Timedelta(days=history_days)

        price = client.query_day_ahead_prices(self.country_code, start=start, end=end)
        load = client.query_load(self.country_code, start=start, end=end)
        gen = client.query_generation(self.country_code, start=start, end=end, psr_type=None)

        df = pd.DataFrame({"price": price})
        df["load"] = load.reindex(df.index).iloc[:, 0]
        df["wind"] = gen.filter(like="Wind").sum(axis=1).reindex(df.index)
        df["solar"] = gen.filter(like="Solar").sum(axis=1).reindex(df.index)
        df = df.reset_index(names="timestamp").ffill().dropna()
        return df
