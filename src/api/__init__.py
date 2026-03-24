# API module for Healthcare Prediction System
from .schemas import (
    DiseaseType,
    RiskLevel,
    PredictionResponse,
    ErrorResponse,
    DiabetesPredictionRequest,
    HeartDiseasePredictionRequest,
    StrokePredictionRequest,
    GenericPredictionRequest
)

__all__ = [
    'DiseaseType',
    'RiskLevel',
    'PredictionResponse',
    'ErrorResponse',
    'DiabetesPredictionRequest',
    'HeartDiseasePredictionRequest',
    'StrokePredictionRequest',
    'GenericPredictionRequest'
]
