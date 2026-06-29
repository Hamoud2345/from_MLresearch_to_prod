"""Adaptateur vers la plateforme ENTSO-E pour des données réelles.

Implémentation optionnelle : utilisable seulement si le paquet ``entsoe`` est
installé et qu'un token API est configuré. On passe du synthétique à la vraie
source en changeant juste ``PPF_DATA_SOURCE=entsoe``.

ENTSO-E (https://transparency.entsoe.eu) est la source publique utilisée par
pas mal de desks énergie en Europe pour les prix day-ahead, la conso et la prod.
"""

from __future__ import annotations

import os

import pandas as pd

from .base import DataSource


class EntsoeDataSource(DataSource):
    """Récupère prix day-ahead, conso et prod renouvelable horaires depuis ENTSO-E."""

    def __init__(self, country_code: str = "FR", api_token: str | None = None) -> None:
        self.country_code = country_code
        self.api_token = api_token or os.environ.get("ENTSOE_API_TOKEN")
        if not self.api_token:
            raise RuntimeError(
                "ENTSOE_API_TOKEN is required for EntsoeDataSource. "
                "Use the synthetic source for offline/CI runs."
            )

    def _fetch(self, history_days: int) -> pd.DataFrame:  # pragma: no cover - needs network
        from entsoe import EntsoePandasClient  # import paresseux : dépendance optionnelle

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
