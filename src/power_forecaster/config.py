"""Typed application configuration.

Centralising configuration in a single typed object (instead of scattering
``os.environ[...]`` calls across the code) is a production habit: it makes the
app's knobs explicit, validated at start-up, and easy to override per
environment (local / CI / Docker) through ``PPF_*`` environment variables.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Global settings, overridable via environment variables prefixed ``PPF_``."""

    model_config = SettingsConfigDict(env_prefix="PPF_", env_file=".env", extra="ignore")

    # --- Data ---------------------------------------------------------------
    data_source: str = Field(default="synthetic", description="synthetic | entsoe")
    history_days: int = Field(default=730, ge=30, description="Days of history to load.")
    random_seed: int = 42

    # --- Model --------------------------------------------------------------
    model_name: str = Field(default="lightgbm", description="Registered forecaster name.")
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
    drift_threshold: float = Field(default=0.2, description="PSI threshold flagging drift.")

    def model_path(self) -> Path:
        """Filesystem location of the serialized production model."""
        return self.artifacts_dir / "model.joblib"


settings = Settings()
