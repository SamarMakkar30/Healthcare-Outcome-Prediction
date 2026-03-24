"""
Production Configuration Module
Centralized configuration management with environment-based settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache
from typing import Optional, Dict, Any
from enum import Enum
import os


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Production-grade configuration with validation.
    Loads from environment variables with sensible defaults.
    """
    
    # Application
    app_name: str = "Healthcare Prediction System"
    app_version: str = "2.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_rate_limit: int = 100  # requests per minute
    
    # Security
    secret_key: str = Field(
        default="CHANGE-THIS-IN-PRODUCTION-USE-SECRETS-MANAGER",
        description="JWT signing key - MUST be changed in production"
    )
    encryption_key: Optional[str] = None
    allowed_origins: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    
    # Database
    database_url: str = "sqlite:///./healthcare_ml.db"
    redis_url: str = "redis://localhost:6379/0"
    
    # ML Configuration
    model_registry_path: str = "./models/registry"
    experiment_tracking_uri: str = "sqlite:///./mlflow.db"
    default_model_version: str = "production"
    
    # Model Training
    random_state: int = 42
    test_size: float = 0.15
    validation_size: float = 0.15
    cv_folds: int = 5
    
    # Healthcare-Specific Thresholds
    diabetes_threshold: float = 0.35  # Optimized for recall
    heart_disease_threshold: float = 0.40
    stroke_threshold: float = 0.30  # Lower threshold due to severity
    
    # False Negative Cost Multipliers (relative to FP cost of 1)
    diabetes_fn_cost: float = 5.0
    heart_disease_fn_cost: float = 8.0
    stroke_fn_cost: float = 10.0  # Stroke has highest FN cost
    
    # Monitoring
    enable_monitoring: bool = True
    drift_detection_threshold: float = 0.1
    prediction_logging: bool = True
    
    # Paths
    data_raw_path: str = "./data/raw"
    data_processed_path: str = "./data/processed"
    models_path: str = "./models"
    logs_path: str = "./logs"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @field_validator("debug", mode="before")
    @classmethod
    def _parse_debug_flag(cls, value):
        """Accept common environment strings for debug mode."""
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "dev", "development"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False

        return value
    
    def get_threshold(self, disease_type: str) -> float:
        """Get optimized threshold for disease type."""
        thresholds = {
            "diabetes": self.diabetes_threshold,
            "heart_disease": self.heart_disease_threshold,
            "stroke": self.stroke_threshold
        }
        return thresholds.get(disease_type, 0.5)
    
    def get_fn_cost(self, disease_type: str) -> float:
        """Get false negative cost for disease type."""
        costs = {
            "diabetes": self.diabetes_fn_cost,
            "heart_disease": self.heart_disease_fn_cost,
            "stroke": self.stroke_fn_cost
        }
        return costs.get(disease_type, 1.0)


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()


# Feature configurations per disease
FEATURE_CONFIG = {
    "diabetes": {
        "numeric_features": [
            "age", "bmi", "blood_pressure", "glucose", "insulin",
            "physical_activity", "sleep_hours", "stress_level"
        ],
        "categorical_features": [
            "gender", "family_history", "smoking", "alcohol"
        ],
        "target": "diabetes",
        "clinical_ranges": {
            "age": (0, 120),
            "bmi": (10, 70),
            "blood_pressure": (60, 250),
            "glucose": (30, 600),
            "insulin": (0, 1000),
            "physical_activity": (0, 7),
            "sleep_hours": (0, 24),
            "stress_level": (1, 10)
        }
    },
    "heart_disease": {
        "numeric_features": [
            "age", "resting_bp", "cholesterol", "max_heart_rate",
            "oldpeak", "bmi"
        ],
        "categorical_features": [
            "gender", "chest_pain_type", "fasting_blood_sugar",
            "resting_ecg", "exercise_angina", "smoking", "family_history"
        ],
        "target": "heart_disease",
        "clinical_ranges": {
            "age": (0, 120),
            "resting_bp": (60, 250),
            "cholesterol": (100, 600),
            "max_heart_rate": (40, 250),
            "oldpeak": (0, 10),
            "bmi": (10, 70)
        }
    },
    "stroke": {
        "numeric_features": [
            "age", "avg_glucose_level", "bmi", "physical_activity",
            "alcohol_intake"
        ],
        "categorical_features": [
            "gender", "hypertension", "heart_disease", "ever_married",
            "work_type", "residence_type", "smoking_status"
        ],
        "target": "stroke",
        "clinical_ranges": {
            "age": (0, 120),
            "avg_glucose_level": (30, 600),
            "bmi": (10, 70),
            "physical_activity": (0, 7),
            "alcohol_intake": (0, 10)
        }
    }
}

# Backward compatibility: some legacy code/tests expect a single `features` key.
for _cfg in FEATURE_CONFIG.values():
    _cfg.setdefault("features", _cfg["numeric_features"] + _cfg["categorical_features"])
