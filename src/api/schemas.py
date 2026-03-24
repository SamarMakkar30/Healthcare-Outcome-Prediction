"""
Production API Schemas
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class DiseaseType(str, Enum):
    """Supported disease types."""
    DIABETES = "diabetes"
    HEART_DISEASE = "heart_disease"
    STROKE = "stroke"


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Gender(str, Enum):
    """Gender options."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class SmokingStatus(str, Enum):
    """Smoking status options."""
    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"


# ==================== Base Models ====================

class HealthMetricsBase(BaseModel):
    """Base health metrics common to all predictions."""
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender: Gender = Field(..., description="Patient gender")
    bmi: float = Field(..., ge=10, le=70, description="Body Mass Index")
    
    @validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v


# ==================== Disease-Specific Request Models ====================

class DiabetesPredictionRequest(HealthMetricsBase):
    """Request model for diabetes prediction."""
    blood_pressure: float = Field(..., ge=60, le=250, description="Blood pressure (systolic)")
    glucose: float = Field(..., ge=30, le=600, description="Blood glucose level")
    insulin: float = Field(0, ge=0, le=1000, description="Insulin level")
    family_history: int = Field(0, ge=0, le=1, description="Family history of diabetes (0/1)")
    physical_activity: int = Field(3, ge=0, le=7, description="Days of exercise per week")
    smoking: int = Field(0, ge=0, le=1, description="Smoking status (0/1)")
    alcohol: int = Field(0, ge=0, le=2, description="Alcohol consumption (0=none, 1=moderate, 2=heavy)")
    sleep_hours: float = Field(7, ge=0, le=24, description="Average sleep hours")
    stress_level: int = Field(5, ge=1, le=10, description="Stress level (1-10)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 55,
                "gender": "male",
                "bmi": 32.5,
                "blood_pressure": 145,
                "glucose": 165,
                "insulin": 80,
                "family_history": 1,
                "physical_activity": 2,
                "smoking": 0,
                "alcohol": 1,
                "sleep_hours": 6,
                "stress_level": 7
            }
        }


class HeartDiseasePredictionRequest(HealthMetricsBase):
    """Request model for heart disease prediction."""
    chest_pain_type: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    resting_bp: float = Field(..., ge=60, le=250, description="Resting blood pressure")
    cholesterol: float = Field(..., ge=100, le=600, description="Cholesterol level")
    fasting_blood_sugar: int = Field(0, ge=0, le=1, description="Fasting blood sugar > 120 (0/1)")
    resting_ecg: int = Field(0, ge=0, le=2, description="Resting ECG results (0-2)")
    max_heart_rate: float = Field(..., ge=40, le=250, description="Maximum heart rate achieved")
    exercise_angina: int = Field(0, ge=0, le=1, description="Exercise induced angina (0/1)")
    oldpeak: float = Field(0, ge=0, le=10, description="ST depression")
    smoking: int = Field(0, ge=0, le=1, description="Smoking status (0/1)")
    family_history: int = Field(0, ge=0, le=1, description="Family history (0/1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 62,
                "gender": "male",
                "bmi": 28.5,
                "chest_pain_type": 3,
                "resting_bp": 150,
                "cholesterol": 250,
                "fasting_blood_sugar": 1,
                "resting_ecg": 1,
                "max_heart_rate": 140,
                "exercise_angina": 1,
                "oldpeak": 2.5,
                "smoking": 1,
                "family_history": 1
            }
        }


class StrokePredictionRequest(HealthMetricsBase):
    """Request model for stroke prediction."""
    hypertension: int = Field(0, ge=0, le=1, description="Hypertension (0/1)")
    heart_disease: int = Field(0, ge=0, le=1, description="Heart disease (0/1)")
    ever_married: str = Field("Yes", description="Ever married (Yes/No)")
    work_type: str = Field("Private", description="Work type")
    residence_type: str = Field("Urban", description="Residence type (Urban/Rural)")
    avg_glucose_level: float = Field(..., ge=30, le=600, description="Average glucose level")
    smoking_status: str = Field("never", description="Smoking status")
    physical_activity: int = Field(3, ge=0, le=7, description="Days of exercise per week")
    alcohol_intake: int = Field(0, ge=0, le=10, description="Alcohol intake level")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 67,
                "gender": "female",
                "bmi": 36.6,
                "hypertension": 1,
                "heart_disease": 1,
                "ever_married": "Yes",
                "work_type": "Private",
                "residence_type": "Urban",
                "avg_glucose_level": 228.69,
                "smoking_status": "formerly",
                "physical_activity": 1,
                "alcohol_intake": 2
            }
        }


class GenericPredictionRequest(BaseModel):
    """Generic prediction request for any disease type."""
    disease_type: DiseaseType
    features: Dict[str, Any] = Field(..., description="Feature dictionary")
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError('Features cannot be empty')
        return v


# ==================== Response Models ====================

class RiskFactor(BaseModel):
    """Individual risk factor explanation."""
    feature: str
    value: Any
    impact: str  # "increases" or "decreases"
    importance: float
    clinical_note: Optional[str] = None


class Recommendation(BaseModel):
    """Health recommendation."""
    category: str
    advice: str
    priority: str  # "urgent", "high", "medium", "low"
    evidence_level: Optional[str] = None


class PredictionResponse(BaseModel):
    """Standard prediction response."""
    request_id: str
    disease_type: DiseaseType
    
    # Core prediction
    risk_probability: float = Field(..., ge=0, le=1)
    # Backward-compatible alias used by legacy frontend/tests
    probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    # Backward-compatible alias used by legacy frontend/tests
    risk_category: str
    prediction: int  # 0 or 1
    
    # Confidence
    confidence_interval: Dict[str, float]  # {"lower": 0.x, "upper": 0.x}
    
    # Threshold info
    threshold_used: float
    threshold_rationale: str
    
    # Explainability
    top_risk_factors: List[RiskFactor]
    contributing_factors: List[str]
    protective_factors: List[str]
    
    # Clinical guidance
    recommendations: List[Recommendation]
    urgency: str
    follow_up_suggested: bool
    
    # Metadata
    model_version: str
    model_name: str
    calibrated: bool
    timestamp: str
    latency_ms: float
    
    # REQUIRED DISCLAIMER
    disclaimer: str = (
        "⚠️ IMPORTANT: This prediction is for informational and educational purposes only. "
        "It does NOT constitute medical advice, diagnosis, or treatment recommendation. "
        "Always consult qualified healthcare professionals for medical decisions. "
        "Do not delay seeking medical attention based on this prediction."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "abc123",
                "disease_type": "diabetes",
                "risk_probability": 0.72,
                "risk_level": "high",
                "prediction": 1,
                "confidence_interval": {"lower": 0.65, "upper": 0.79},
                "threshold_used": 0.35,
                "threshold_rationale": "Optimized for 85% recall to minimize missed diagnoses",
                "top_risk_factors": [],
                "contributing_factors": ["elevated glucose", "high BMI"],
                "protective_factors": ["regular exercise"],
                "recommendations": [],
                "urgency": "high",
                "follow_up_suggested": True,
                "model_version": "2.0.0",
                "model_name": "xgboost",
                "calibrated": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "latency_ms": 45.2,
                "disclaimer": "..."
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    request_id: str
    total_records: int
    processed: int
    failed: int
    predictions: List[PredictionResponse]
    errors: List[Dict[str, str]]
    processing_time_ms: float


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ModelInfoResponse(BaseModel):
    """Model information response."""
    disease_type: str
    model_name: str
    version: str
    threshold: float
    metrics: Dict[str, float]
    last_trained: str
    feature_names: List[str]
    calibrated: bool


# ==================== Validation Helpers ====================

def validate_clinical_ranges(disease_type: DiseaseType, features: Dict[str, Any]) -> List[str]:
    """Validate that features are within clinical ranges."""
    warnings = []
    
    clinical_ranges = {
        "age": (0, 120),
        "bmi": (10, 70),
        "glucose": (30, 600),
        "blood_pressure": (60, 250),
        "cholesterol": (100, 600),
        "heart_rate": (40, 250)
    }
    
    for feature, (min_val, max_val) in clinical_ranges.items():
        if feature in features:
            value = features[feature]
            if value < min_val or value > max_val:
                warnings.append(
                    f"{feature}={value} is outside typical clinical range [{min_val}, {max_val}]"
                )
    
    return warnings
