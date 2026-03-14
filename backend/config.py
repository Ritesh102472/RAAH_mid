from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://raah_user:raah_pass@localhost:5432/raah_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    STORAGE_BACKEND: str = "local"
    UPLOAD_DIR: str = "./uploads"
    MAPILLARY_ACCESS_TOKEN: str = ""
    AI_MODEL_URL: str = "http://localhost:8001/detect-pothole"
    AI_MODEL_ENABLED: bool = True
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:5174,http://localhost:3000"
    APP_ENV: str = "development"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
