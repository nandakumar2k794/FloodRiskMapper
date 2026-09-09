"""
Application settings loaded from environment variables / .env file.
Uses Pydantic BaseSettings for validation and type coercion.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import List, Any


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    postgres_host: str = Field("localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    postgres_db: str = Field("floodmap", alias="POSTGRES_DB")
    postgres_user: str = Field("floodmap", alias="POSTGRES_USER")
    postgres_password: str = Field("floodmap123", alias="POSTGRES_PASSWORD")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_dsn_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── MongoDB ─────────────────────────────────────────────────────────────
    mongo_uri: str = Field("mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db: str = Field("drainwatch", alias="MONGO_DB")

    # ── External APIs ───────────────────────────────────────────────────────
    openweather_api_key: str = Field("", alias="OPENWEATHER_API_KEY")

    # ── Application ─────────────────────────────────────────────────────────
    app_host: str = Field("0.0.0.0", alias="APP_HOST")
    app_port: int = Field(8000, alias="APP_PORT")
    app_debug: bool = Field(True, alias="APP_DEBUG")
    cors_origins: Any = Field(
        default=["http://localhost:5173", "http://localhost:3000", "https://nandakumar2k794.github.io"],
        alias="CORS_ORIGINS",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.startswith("[") and v_stripped.endswith("]"):
                import json
                try:
                    return json.loads(v_stripped)
                except Exception:
                    pass
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    # ── Model Paths ─────────────────────────────────────────────────────────
    risk_model_path: str = Field("models/risk_scorer.pkl", alias="RISK_MODEL_PATH")
    yolo_model_path: str = Field("models/yolov8n_drainwatch.pt", alias="YOLO_MODEL_PATH")


# Singleton
settings = Settings()
