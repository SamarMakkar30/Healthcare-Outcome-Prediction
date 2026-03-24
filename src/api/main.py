"""
Production FastAPI Application
Healthcare Prediction REST API with full production features.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any
import time
from datetime import datetime
import uuid

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_settings
from core.logging_config import setup_logging, get_logger
from api.schemas import (
    DiseaseType,
    PredictionResponse,
    ErrorResponse,
    HealthCheckResponse,
    ModelInfoResponse,
    DiabetesPredictionRequest,
    HeartDiseasePredictionRequest,
    StrokePredictionRequest,
    GenericPredictionRequest,
    validate_clinical_ranges
)
from services.inference_service import get_inference_service, HealthcareInferenceService

# Setup logging
setup_logging(log_level="INFO", json_format=False)
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Healthcare Prediction API...")
    
    # Load all models
    service = get_inference_service()
    load_results = service.load_all_models()
    
    for disease, loaded in load_results.items():
        status = "✓" if loaded else "✗"
        logger.info(f"  {status} {disease} model")
    
    logger.info("API ready to serve predictions")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Healthcare Prediction API...")


# Create FastAPI app
app = FastAPI(
    title="Healthcare Prediction API",
    description="""
    ## Production ML System for Multi-Disease Risk Assessment
    
    ### Features
    - **Multi-Disease Prediction**: Diabetes, Heart Disease, Stroke
    - **Calibrated Probabilities**: Reliable probability estimates
    - **Optimized Thresholds**: Tuned for recall to minimize missed diagnoses
    - **SHAP Explainability**: Understand which features drive predictions
    - **Clinical Recommendations**: Evidence-based health guidance
    
    ### Important Disclaimer
    ⚠️ This API provides predictions for **informational purposes only**.
    It does NOT constitute medical advice. Always consult healthcare professionals.
    
    ### API Version
    Production-grade implementation with proper ML pipeline, calibration, and monitoring.
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Middleware ====================

@app.middleware("http")
async def add_request_tracking(request: Request, call_next):
    """Add request ID and timing to all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    
    return response


# ==================== Exception Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An internal error occurred. Please try again.",
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


# ==================== Dependencies ====================

def get_service() -> HealthcareInferenceService:
    """Dependency to get inference service."""
    return get_inference_service()


# ==================== Health Endpoints ====================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check(service: HealthcareInferenceService = Depends(get_service)):
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        version=settings.app_version,
        models_loaded={
            "diabetes": service.is_model_loaded("diabetes"),
            "heart_disease": service.is_model_loaded("heart_disease"),
            "stroke": service.is_model_loaded("stroke")
        },
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


# ==================== Prediction Endpoints ====================

@app.post(
    "/predict/diabetes",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict Diabetes Risk"
)
async def predict_diabetes(
    request: DiabetesPredictionRequest,
    service: HealthcareInferenceService = Depends(get_service)
):
    """
    Predict diabetes risk based on patient features.
    
    The model uses calibrated probabilities and an optimized threshold
    tuned for high recall (minimizing missed diagnoses).
    """
    features = request.dict()
    features['gender'] = request.gender.value
    
    # Validate clinical ranges
    warnings = validate_clinical_ranges(DiseaseType.DIABETES, features)
    if warnings:
        logger.warning(f"Clinical range warnings: {warnings}")
    
    try:
        response = service.predict(
            disease_type="diabetes",
            features=features,
            request_id=str(uuid.uuid4())
        )
        return response
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/heart_disease",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict Heart Disease Risk"
)
async def predict_heart_disease(
    request: HeartDiseasePredictionRequest,
    service: HealthcareInferenceService = Depends(get_service)
):
    """
    Predict heart disease risk based on patient features.
    
    Includes cardiovascular-specific risk factors and recommendations.
    """
    features = request.dict()
    features['gender'] = request.gender.value
    
    try:
        response = service.predict(
            disease_type="heart_disease",
            features=features,
            request_id=str(uuid.uuid4())
        )
        return response
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/stroke",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict Stroke Risk"
)
async def predict_stroke(
    request: StrokePredictionRequest,
    service: HealthcareInferenceService = Depends(get_service)
):
    """
    Predict stroke risk based on patient features.
    
    Uses a lower threshold due to the severity of stroke,
    prioritizing sensitivity over specificity.
    """
    features = request.dict()
    features['gender'] = request.gender.value
    
    try:
        response = service.predict(
            disease_type="stroke",
            features=features,
            request_id=str(uuid.uuid4())
        )
        return response
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Generic Prediction Endpoint"
)
async def predict_generic(
    request: GenericPredictionRequest,
    service: HealthcareInferenceService = Depends(get_service)
):
    """
    Generic prediction endpoint for any disease type.
    
    Pass the disease_type and features dictionary.
    """
    try:
        response = service.predict(
            disease_type=request.disease_type.value,
            features=request.features,
            request_id=request.request_id
        )
        return response
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Model Info Endpoints ====================

@app.get(
    "/models/{disease_type}",
    response_model=ModelInfoResponse,
    tags=["Model Info"],
    summary="Get Model Information"
)
async def get_model_info(
    disease_type: DiseaseType,
    service: HealthcareInferenceService = Depends(get_service)
):
    """Get information about a specific model."""
    dt = disease_type.value
    
    if not service.is_model_loaded(dt):
        raise HTTPException(status_code=404, detail=f"Model not loaded: {dt}")
    
    from core.config import FEATURE_CONFIG
    config = FEATURE_CONFIG.get(dt, {})
    
    return ModelInfoResponse(
        disease_type=dt,
        model_name=service._model_names.get(dt, "unknown"),
        version=service._model_versions.get(dt, "unknown"),
        threshold=service._thresholds.get(dt, 0.5),
        metrics=service._threshold_configs.get(dt, {}).get('metrics', {}),
        last_trained="N/A",
        feature_names=config.get('numeric_features', []) + config.get('categorical_features', []),
        calibrated=True
    )


@app.get("/models", tags=["Model Info"], summary="List All Models")
async def list_models(service: HealthcareInferenceService = Depends(get_service)):
    """List all available models and their status."""
    return {
        "models": {
            "diabetes": {
                "loaded": service.is_model_loaded("diabetes"),
                "threshold": service._thresholds.get("diabetes", 0.5)
            },
            "heart_disease": {
                "loaded": service.is_model_loaded("heart_disease"),
                "threshold": service._thresholds.get("heart_disease", 0.5)
            },
            "stroke": {
                "loaded": service.is_model_loaded("stroke"),
                "threshold": service._thresholds.get("stroke", 0.5)
            }
        }
    }


# ==================== Run Configuration ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api_workers
    )
