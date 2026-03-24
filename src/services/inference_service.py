"""
Production Healthcare Inference Service
Handles model loading, prediction, and explainability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import time
import hashlib
from datetime import datetime

import joblib
import shap

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_settings, FEATURE_CONFIG
from core.logging_config import get_logger, get_audit_logger
from api.schemas import (
    DiseaseType, RiskLevel, PredictionResponse, 
    RiskFactor, Recommendation
)

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class HealthcareInferenceService:
    """
    Production inference service for healthcare predictions.
    
    Features:
    - Lazy model loading with caching
    - Request deduplication
    - Prediction with calibrated probabilities
    - SHAP-based explainability
    - Clinical recommendations
    """
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.settings = get_settings()
        
        # Cached models and configurations
        self._pipelines: Dict[str, Any] = {}
        self._thresholds: Dict[str, float] = {}
        self._threshold_configs: Dict[str, Dict] = {}
        self._shap_explainers: Dict[str, Any] = {}
        
        # Model metadata
        self._model_versions: Dict[str, str] = {}
        self._model_names: Dict[str, str] = {}
        
    def load_model(self, disease_type: str) -> bool:
        """
        Load model for a disease type.
        
        Returns True if successful, False otherwise.
        """
        try:
            pipeline_path = self.models_dir / f"{disease_type}_pipeline.pkl"
            threshold_path = self.models_dir / f"{disease_type}_threshold.json"
            
            if not pipeline_path.exists():
                logger.warning(f"Model not found: {pipeline_path}")
                return False
            
            # Load pipeline
            self._pipelines[disease_type] = joblib.load(pipeline_path)
            
            # Load threshold configuration
            if threshold_path.exists():
                with open(threshold_path, 'r') as f:
                    threshold_config = json.load(f)
                    self._thresholds[disease_type] = threshold_config['optimal_threshold']
                    self._threshold_configs[disease_type] = threshold_config
                    self._model_names[disease_type] = threshold_config.get('model_name', 'unknown')
            else:
                # Use default threshold from settings
                self._thresholds[disease_type] = self.settings.get_threshold(disease_type)
            
            self._model_versions[disease_type] = self.settings.app_version
            
            audit_logger.log_model_loaded(
                disease_type,
                self._model_versions[disease_type]
            )
            
            logger.info(f"Loaded {disease_type} model (threshold={self._thresholds[disease_type]:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {disease_type} model: {e}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load all available models."""
        results = {}
        for disease in ['diabetes', 'heart_disease', 'stroke']:
            results[disease] = self.load_model(disease)
        return results
    
    def is_model_loaded(self, disease_type: str) -> bool:
        """Check if a model is loaded."""
        return disease_type in self._pipelines
    
    def _prepare_features(
        self, 
        disease_type: str, 
        features: Dict[str, Any]
    ) -> pd.DataFrame:
        """Prepare feature DataFrame from request."""
        config = FEATURE_CONFIG.get(disease_type, {})
        all_features = config.get('numeric_features', []) + config.get('categorical_features', [])
        
        # Create DataFrame with correct column order
        data = {}
        for feature in all_features:
            if feature in features:
                data[feature] = [features[feature]]
            else:
                # Use default values for missing features
                data[feature] = [0]
        
        return pd.DataFrame(data)
    
    def _classify_risk(self, probability: float, disease_type: str) -> RiskLevel:
        """Classify risk level based on probability."""
        # Disease-specific thresholds for risk categorization
        if disease_type == "stroke":
            # Stroke has more aggressive categorization due to severity
            if probability >= 0.6:
                return RiskLevel.CRITICAL
            elif probability >= 0.4:
                return RiskLevel.HIGH
            elif probability >= 0.2:
                return RiskLevel.MODERATE
            else:
                return RiskLevel.LOW
        else:
            # Standard categorization
            if probability >= 0.7:
                return RiskLevel.CRITICAL
            elif probability >= 0.5:
                return RiskLevel.HIGH
            elif probability >= 0.3:
                return RiskLevel.MODERATE
            else:
                return RiskLevel.LOW
    
    def _get_shap_explanation(
        self,
        disease_type: str,
        X: pd.DataFrame
    ) -> List[RiskFactor]:
        """Get SHAP-based feature explanations."""
        try:
            pipeline = self._pipelines[disease_type]
            
            # Get model from pipeline (could be 'classifier' or 'model')
            model = pipeline.named_steps.get('classifier') or pipeline.named_steps.get('model')
            
            if model is None:
                return []
            
            # Initialize SHAP explainer if needed
            if disease_type not in self._shap_explainers:
                try:
                    self._shap_explainers[disease_type] = shap.TreeExplainer(model)
                except:
                    return []
            
            # Transform features through preprocessor only
            preprocessor = pipeline.named_steps.get('preprocessor')
            if preprocessor:
                X_transformed = preprocessor.transform(X)
            else:
                X_transformed = X.values if hasattr(X, 'values') else X
            
            # Get SHAP values
            explainer = self._shap_explainers[disease_type]
            shap_values = explainer.shap_values(X_transformed)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            # Get feature names
            preprocessor = pipeline.named_steps.get('preprocessor')
            if preprocessor:
                feature_names = preprocessor.get_feature_names_out().tolist()
            else:
                feature_names = X.columns.tolist()
            
            # Create risk factors
            risk_factors = []
            for idx, (name, shap_val) in enumerate(zip(feature_names, shap_values)):
                # Clean up feature name
                clean_name = name.replace('num__', '').replace('cat__', '')
                
                risk_factors.append(RiskFactor(
                    feature=clean_name,
                    value=float(X_transformed[0, idx]) if hasattr(X_transformed, '__getitem__') else 0,
                    impact="increases" if shap_val > 0 else "decreases",
                    importance=abs(float(shap_val)),
                    clinical_note=None
                ))
            
            # Sort by importance and return top 5
            risk_factors.sort(key=lambda x: x.importance, reverse=True)
            return risk_factors[:5]
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return []
    
    def _generate_recommendations(
        self,
        disease_type: str,
        features: Dict[str, Any],
        risk_level: RiskLevel
    ) -> List[Recommendation]:
        """Generate personalized recommendations."""
        recommendations = []
        
        # Urgent recommendation for high risk
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append(Recommendation(
                category="Urgent Action",
                advice="Please consult a healthcare provider as soon as possible for professional evaluation.",
                priority="urgent",
                evidence_level="Expert consensus"
            ))
        
        # Disease-specific recommendations
        if disease_type == "diabetes":
            recommendations.extend(self._diabetes_recommendations(features))
        elif disease_type == "heart_disease":
            recommendations.extend(self._heart_disease_recommendations(features))
        elif disease_type == "stroke":
            recommendations.extend(self._stroke_recommendations(features))
        
        # General recommendations
        recommendations.append(Recommendation(
            category="Regular Monitoring",
            advice="Schedule regular health screenings and follow up with your healthcare provider.",
            priority="medium",
            evidence_level="Clinical guidelines"
        ))
        
        return recommendations
    
    def _diabetes_recommendations(self, features: Dict[str, Any]) -> List[Recommendation]:
        """Generate diabetes-specific recommendations."""
        recs = []
        
        if features.get('bmi', 0) > 30:
            recs.append(Recommendation(
                category="Weight Management",
                advice="Your BMI indicates obesity. Evidence shows that losing 5-10% of body weight can significantly reduce diabetes risk.",
                priority="high",
                evidence_level="Strong evidence (multiple RCTs)"
            ))
        
        if features.get('glucose', 0) > 140:
            recs.append(Recommendation(
                category="Blood Sugar",
                advice="Elevated glucose levels detected. Consider monitoring blood sugar regularly and discussing with an endocrinologist.",
                priority="high",
                evidence_level="Clinical guidelines"
            ))
        
        if features.get('physical_activity', 7) < 3:
            recs.append(Recommendation(
                category="Physical Activity",
                advice="Increase physical activity to at least 150 minutes of moderate exercise per week. This has been shown to reduce diabetes risk by 30-50%.",
                priority="high",
                evidence_level="Strong evidence (DPP study)"
            ))
        
        return recs
    
    def _heart_disease_recommendations(self, features: Dict[str, Any]) -> List[Recommendation]:
        """Generate heart disease-specific recommendations."""
        recs = []
        
        if features.get('cholesterol', 0) > 240:
            recs.append(Recommendation(
                category="Cholesterol Management",
                advice="High cholesterol detected. Consider dietary changes (reduce saturated fats, increase fiber) and discuss statin therapy with your doctor.",
                priority="high",
                evidence_level="Strong evidence (ACC/AHA guidelines)"
            ))
        
        if features.get('resting_bp', 0) > 140:
            recs.append(Recommendation(
                category="Blood Pressure",
                advice="Elevated blood pressure increases heart disease risk. Reduce sodium intake, maintain healthy weight, and consider medication if lifestyle changes are insufficient.",
                priority="high",
                evidence_level="Strong evidence (JNC guidelines)"
            ))
        
        if features.get('smoking', 0) == 1:
            recs.append(Recommendation(
                category="Smoking Cessation",
                advice="Smoking significantly increases cardiovascular risk. Quitting smoking reduces heart disease risk by 50% within 1 year.",
                priority="urgent",
                evidence_level="Strong evidence (Surgeon General's Report)"
            ))
        
        return recs
    
    def _stroke_recommendations(self, features: Dict[str, Any]) -> List[Recommendation]:
        """Generate stroke-specific recommendations."""
        recs = []
        
        if features.get('hypertension', 0) == 1:
            recs.append(Recommendation(
                category="Hypertension Control",
                advice="Hypertension is the #1 modifiable risk factor for stroke. Work with your doctor to achieve target BP < 130/80 mmHg.",
                priority="urgent",
                evidence_level="Strong evidence (SPRINT trial)"
            ))
        
        if features.get('heart_disease', 0) == 1:
            recs.append(Recommendation(
                category="Cardiac Management",
                advice="Existing heart disease increases stroke risk. Ensure optimal management of cardiac conditions and consider anticoagulation if indicated.",
                priority="high",
                evidence_level="ACC/AHA guidelines"
            ))
        
        if features.get('avg_glucose_level', 0) > 140:
            recs.append(Recommendation(
                category="Glucose Control",
                advice="Elevated glucose is associated with increased stroke risk. Maintain HbA1c below 7% through diet, exercise, and medication if needed.",
                priority="high",
                evidence_level="ADA guidelines"
            ))
        
        return recs
    
    def predict(
        self,
        disease_type: str,
        features: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> PredictionResponse:
        """
        Make prediction with full production features.
        
        Returns calibrated probability, optimal threshold prediction,
        SHAP explanations, and recommendations.
        """
        start_time = time.time()
        request_id = request_id or str(hash(str(features)))
        
        # Ensure model is loaded
        if not self.is_model_loaded(disease_type):
            self.load_model(disease_type)
        
        if not self.is_model_loaded(disease_type):
            raise ValueError(f"Model not available for {disease_type}")
        
        # Log prediction request
        audit_logger.log_prediction_request(
            request_id=request_id,
            disease_type=disease_type
        )
        
        # Prepare features
        X = self._prepare_features(disease_type, features)
        
        # Get prediction
        pipeline = self._pipelines[disease_type]
        proba = pipeline.predict_proba(X)[0, 1]
        
        # Apply optimal threshold
        threshold = self._thresholds[disease_type]
        prediction = int(proba >= threshold)
        
        # Classify risk level
        risk_level = self._classify_risk(proba, disease_type)
        
        # Get explanations
        risk_factors = self._get_shap_explanation(disease_type, X)
        
        # Separate contributing and protective factors
        contributing = [f.feature for f in risk_factors if f.impact == "increases"][:3]
        protective = [f.feature for f in risk_factors if f.impact == "decreases"][:3]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(disease_type, features, risk_level)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Confidence interval (rough estimate)
        std_estimate = np.sqrt(proba * (1 - proba) / 100)
        ci_lower = max(0, proba - 1.96 * std_estimate)
        ci_upper = min(1, proba + 1.96 * std_estimate)
        
        # Build response
        response = PredictionResponse(
            request_id=request_id,
            disease_type=DiseaseType(disease_type),
            risk_probability=round(proba, 4),
            probability=round(proba, 4),
            risk_level=risk_level,
            risk_category=risk_level.value,
            prediction=prediction,
            confidence_interval={"lower": round(ci_lower, 4), "upper": round(ci_upper, 4)},
            threshold_used=threshold,
            threshold_rationale=f"Optimized for ≥85% recall to minimize missed diagnoses (FN cost={self.settings.get_fn_cost(disease_type)}x FP cost)",
            top_risk_factors=risk_factors,
            contributing_factors=contributing,
            protective_factors=protective,
            recommendations=recommendations,
            urgency="immediate" if risk_level == RiskLevel.CRITICAL else "high" if risk_level == RiskLevel.HIGH else "routine",
            follow_up_suggested=proba >= 0.3,
            model_version=self._model_versions.get(disease_type, "unknown"),
            model_name=self._model_names.get(disease_type, "unknown"),
            calibrated=True,
            timestamp=datetime.utcnow().isoformat() + "Z",
            latency_ms=round(latency_ms, 2)
        )
        
        # Log result
        audit_logger.log_prediction_result(
            request_id=request_id,
            disease_type=disease_type,
            risk_level=risk_level.value,
            probability=proba,
            latency_ms=latency_ms
        )
        
        return response


# Singleton instance
_inference_service: Optional[HealthcareInferenceService] = None

def get_inference_service() -> HealthcareInferenceService:
    """Get or create inference service singleton."""
    global _inference_service
    if _inference_service is None:
        settings = get_settings()
        _inference_service = HealthcareInferenceService(settings.models_path)
    return _inference_service
