import logging
from pathlib import Path
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class Settings(BaseSettings):

    PROJECT_NAME: str = "EcoShift Forecaster Service"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    API_KEY: str = None

    MODEL_PATH: Path = Field(default=Path("artifacts/forecaster_model.joblib"), description="Artifact mode path of the forecaster")
    MINIMUM_HISTORY_POINTS: int = Field(default=336, description="Minimum number of hours to do a prediction")

    ALLOWED_HOSTS: List[str] = Field(default=["*"], description="Origin list authorized for HTTP requests")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive="True", extra="ignore")

    @field_validator("MODEL_PATH")
    @classmethod
    def validate_model_path_exists(cls, v: Path) -> Path:
        absolute_path = v.resolve()

        if not absolute_path.exists():
            raise FileNotFoundError(f"ML model artifact not found at : {absolute_path}")

        if not absolute_path.is_file():
            raise ValueError(f"The given model path is not a file path : {absolute_path}")
        
        return v


settings = Settings()