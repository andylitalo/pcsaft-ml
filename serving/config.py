"""Configuration for the PC-SAFT prediction API via pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API settings, overridable via PCSAFT_* environment variables."""

    model_type: str = "rf"
    submissions_path: str = "serving/submissions.csv"
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "PCSAFT_"}


settings = Settings()
