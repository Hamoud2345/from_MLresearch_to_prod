"""Configuration de l'appli, typée.

On regroupe toute la config dans un seul objet typé au lieu de balancer des
``os.environ[...]`` partout. Comme ça les paramètres sont explicites, validés au
démarrage, et faciles à surcharger selon l'environnement (local / CI / Docker)
via les variables ``PPF_*``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Paramètres globaux, surchargeables via les variables d'env préfixées ``PPF_``."""

    model_config = SettingsConfigDict(env_prefix="PPF_", env_file=".env", extra="ignore")

    # --- Données ------------------------------------------------------------
    data_source: str = Field(default="synthetic", description="synthetic | entsoe")
    history_days: int = Field(default=730, ge=30, description="Jours d'historique à charger.")
    random_seed: int = 42

    # --- Modèle -------------------------------------------------------------
    model_name: str = Field(default="lightgbm", description="Nom du modèle enregistré.")
    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)

    # --- Backtest -----------------------------------------------------------
    train_window_days: int = 365
    test_window_days: int = 30

    # --- MLOps --------------------------------------------------------------
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db")
    mlflow_experiment: str = "day-ahead-price-forecast"
    registered_model_name: str = "day_ahead_price_forecaster"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    # --- Monitoring ---------------------------------------------------------
    drift_threshold: float = Field(default=0.2, description="Seuil d'alerte de drift (PSI).")

    def model_path(self) -> Path:
        """Chemin du modèle de prod sérialisé."""
        return self.artifacts_dir / "model.joblib"


settings = Settings()
