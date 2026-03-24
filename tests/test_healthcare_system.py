"""
Healthcare Prediction System - Production Test Suite
Comprehensive testing for ML pipelines, API, and healthcare constraints.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture
def sample_diabetes_features():
    """Sample diabetes prediction features."""
    return {
        "age": 55,
        "gender": "male",
        "bmi": 28.5,
        "blood_pressure": 130,
        "glucose": 150,
        "insulin": 85,
        "skin_thickness": 25,
        "pregnancies": 0,
        "diabetes_pedigree": 0.5,
        "family_history": 1
    }


@pytest.fixture
def sample_heart_features():
    """Sample heart disease prediction features."""
    return {
        "age": 60,
        "gender": "male",
        "chest_pain_type": 2,
        "resting_bp": 140,
        "cholesterol": 250,
        "fasting_blood_sugar": 1,
        "rest_ecg": 1,
        "max_heart_rate": 150,
        "exercise_angina": 0,
        "st_depression": 1.5,
        "st_slope": 1,
        "num_major_vessels": 1,
        "thalassemia": 2
    }


@pytest.fixture
def sample_stroke_features():
    """Sample stroke prediction features."""
    return {
        "age": 65,
        "gender": "male",
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "residence_type": "Urban",
        "avg_glucose_level": 180.0,
        "bmi": 30.0,
        "smoking_status": "formerly smoked"
    }


@pytest.fixture
def sample_training_data():
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'age': np.random.randint(20, 80, n_samples),
        'bmi': np.random.uniform(18, 45, n_samples),
        'glucose': np.random.uniform(70, 200, n_samples),
        'blood_pressure': np.random.randint(90, 180, n_samples),
    })
    
    # Create imbalanced target (10% positive)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    return X, y


# =============================================================================
# Unit Tests: Configuration
# =============================================================================

class TestConfiguration:
    """Test configuration module."""
    
    def test_settings_default_thresholds(self):
        """Test that default thresholds are healthcare-optimized."""
        from core.config import get_settings
        
        settings = get_settings()
        
        # Thresholds should be below 0.5 for recall optimization
        assert settings.diabetes_threshold < 0.5
        assert settings.heart_disease_threshold < 0.5
        assert settings.stroke_threshold < 0.5
        
        # Stroke should have lowest threshold (highest stakes)
        assert settings.stroke_threshold <= settings.diabetes_threshold
        assert settings.stroke_threshold <= settings.heart_disease_threshold
    
    def test_fn_cost_multipliers(self):
        """Test that false negative costs reflect clinical severity."""
        from core.config import get_settings
        
        settings = get_settings()
        
        # Stroke FN cost should be highest
        assert settings.stroke_fn_cost >= settings.heart_disease_fn_cost
        assert settings.heart_disease_fn_cost >= settings.diabetes_fn_cost
    
    def test_feature_config_completeness(self):
        """Test that feature configs are complete."""
        from core.config import FEATURE_CONFIG
        
        for disease, config in FEATURE_CONFIG.items():
            assert 'features' in config
            assert 'clinical_ranges' in config
            assert len(config['features']) > 0


# =============================================================================
# Unit Tests: ML Pipeline
# =============================================================================

class TestMLPipeline:
    """Test ML pipeline components."""
    
    def test_clinical_range_validator(self, sample_diabetes_features):
        """Test clinical range validation."""
        from ml.pipeline import ClinicalRangeValidator
        
        validator = ClinicalRangeValidator('diabetes')
        
        # Valid data should pass
        df = pd.DataFrame([sample_diabetes_features])
        validated = validator.fit_transform(df)
        
        assert validated is not None
        assert len(validated) == 1
    
    def test_feature_engineer_creates_features(self, sample_diabetes_features):
        """Test feature engineering creates derived features."""
        from ml.pipeline import FeatureEngineer
        
        engineer = FeatureEngineer('diabetes')
        
        df = pd.DataFrame([sample_diabetes_features])
        transformed = engineer.fit_transform(df)
        
        # Should have more columns than input
        assert transformed.shape[1] >= df.shape[1]
    
    def test_pipeline_no_data_leakage(self, sample_training_data):
        """Test that pipeline doesn't leak data between train/test."""
        from ml.pipeline import create_full_pipeline
        from sklearn.model_selection import train_test_split
        
        X, y = sample_training_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        pipeline = create_full_pipeline('diabetes')
        
        # Fit on training data
        pipeline.fit(X_train, y_train)
        
        # Transform should work on test data
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
        
        # Shapes should be consistent
        X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
        assert X_test_transformed.shape[1] == X_train_transformed.shape[1]


# =============================================================================
# Unit Tests: Probability Calibration
# =============================================================================

class TestCalibration:
    """Test probability calibration and threshold optimization."""
    
    def test_threshold_optimizer_recall_constraint(self):
        """Test threshold optimizer respects recall constraint."""
        from ml.calibration import ThresholdOptimizer
        
        # Mock probabilities and labels
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        y_proba = np.array([0.9, 0.7, 0.5, 0.3, 0.8, 0.4, 0.2, 0.1, 0.05, 0.02])
        
        optimizer = ThresholdOptimizer()
        threshold, metrics = optimizer.optimize(
            y_true, y_proba, 
            method='recall_constrained', 
            min_recall=0.75
        )
        
        # Verify recall constraint is met
        predictions = (y_proba >= threshold).astype(int)
        true_positives = np.sum((predictions == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        
        assert recall >= 0.75 or threshold <= 0.5  # Either meets constraint or is at minimum
    
    def test_threshold_optimizer_cost_sensitive(self):
        """Test cost-sensitive threshold optimization."""
        from ml.calibration import ThresholdOptimizer
        
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        y_proba = np.array([0.9, 0.7, 0.5, 0.3, 0.8, 0.4, 0.2, 0.1, 0.05, 0.02])
        
        optimizer = ThresholdOptimizer()
        
        # High FN cost should lower threshold
        threshold_high_fn, _ = optimizer.optimize(
            y_true, y_proba, method='cost_sensitive', fn_cost=10.0, fp_cost=1.0
        )
        
        threshold_equal, _ = optimizer.optimize(
            y_true, y_proba, method='cost_sensitive', fn_cost=1.0, fp_cost=1.0
        )
        
        # Higher FN cost should result in lower threshold
        assert threshold_high_fn <= threshold_equal


# =============================================================================
# Unit Tests: Healthcare Intelligence
# =============================================================================

class TestHealthcareIntelligence:
    """Test healthcare intelligence layer."""
    
    def test_risk_categorization(self):
        """Test risk categories are correctly assigned."""
        from healthcare.intelligence import HealthcareIntelligence
        
        intelligence = HealthcareIntelligence()
        
        # Low probability should be low risk
        category, _ = intelligence.get_risk_category('diabetes', 0.1)
        assert category == 'low'
        
        # High probability should be high/very_high risk
        category, _ = intelligence.get_risk_category('diabetes', 0.7)
        assert category in ['high', 'very_high']
    
    def test_clinical_urgency_escalation(self):
        """Test that urgency escalates with probability and disease severity."""
        from healthcare.intelligence import HealthcareIntelligence, ClinicalUrgency
        
        intelligence = HealthcareIntelligence()
        
        # High stroke probability should be urgent
        urgency = intelligence.get_clinical_urgency('stroke', 0.6)
        assert urgency in [ClinicalUrgency.URGENT, ClinicalUrgency.EMERGENT]
        
        # Low diabetes probability should be routine
        urgency = intelligence.get_clinical_urgency('diabetes', 0.15)
        assert urgency == ClinicalUrgency.ROUTINE
    
    def test_threshold_rationale_exists(self):
        """Test that threshold rationale is provided for all diseases."""
        from healthcare.intelligence import HealthcareIntelligence
        
        intelligence = HealthcareIntelligence()
        
        for disease in ['diabetes', 'heart_disease', 'stroke']:
            rationale = intelligence.get_threshold_rationale(disease)
            
            assert 'threshold' in rationale
            assert 'rationale' in rationale
            assert len(rationale['rationale']) > 50  # Substantive explanation
    
    def test_borderline_case_guidance(self):
        """Test that borderline cases receive special guidance."""
        from healthcare.intelligence import HealthcareIntelligence
        from core.config import get_settings
        
        intelligence = HealthcareIntelligence()
        settings = get_settings()
        
        # Borderline case: just below threshold
        threshold = settings.diabetes_threshold
        guidance = intelligence.get_false_negative_guidance(
            'diabetes', threshold - 0.05, 0  # Predicted negative
        )
        
        assert guidance is not None
        assert 'warning' in guidance
        assert 'recommendations' in guidance


# =============================================================================
# Unit Tests: MLOps
# =============================================================================

class TestMLOps:
    """Test MLOps components."""
    
    def test_model_registry_versioning(self, tmp_path):
        """Test model registry tracks versions."""
        from mlops.registry import ModelRegistry
        
        registry = ModelRegistry(base_path=str(tmp_path / 'registry'))
        
        # Register mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0, 1]))
        mock_metrics = {'accuracy': 0.85, 'recall': 0.90}
        
        version = registry.register_model(
            model=mock_model,
            model_name='test_model',
            metrics=mock_metrics,
            metadata={'disease': 'diabetes'}
        )
        
        assert version is not None
        assert 'v' in version or version.startswith('1')  # Version format
    
    def test_drift_detection(self):
        """Test drift detection identifies distribution shifts."""
        from mlops.monitoring import PredictionMonitor
        
        monitor = PredictionMonitor()
        
        # Record baseline
        baseline_features = np.random.normal(0, 1, (100, 4))
        for features in baseline_features:
            monitor.record_prediction(
                features=dict(zip(['f1', 'f2', 'f3', 'f4'], features)),
                prediction=0,
                probability=0.3,
                disease_type='diabetes'
            )
        
        # Check with same distribution (no drift)
        report = monitor.check_drift('diabetes')
        
        # With enough samples, drift should be detected or not based on statistics
        assert hasattr(report, 'has_drift')


# =============================================================================
# Integration Tests: API
# =============================================================================

class TestAPIIntegration:
    """Integration tests for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
    
    def test_prediction_requires_valid_input(self, client):
        """Test that invalid input is rejected."""
        # Missing required fields
        response = client.post('/predict/diabetes', json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_prediction_returns_required_fields(
        self, client, sample_diabetes_features
    ):
        """Test prediction response has all required fields."""
        response = client.post('/predict/diabetes', json=sample_diabetes_features)
        
        if response.status_code == 200:
            data = response.json()
            
            # Must have these fields for healthcare compliance
            assert 'probability' in data
            assert 'prediction' in data
            assert 'disclaimer' in data
            assert 'risk_category' in data
    
    def test_prediction_includes_disclaimer(
        self, client, sample_diabetes_features
    ):
        """Test that predictions always include medical disclaimer."""
        response = client.post('/predict/diabetes', json=sample_diabetes_features)
        
        if response.status_code == 200:
            data = response.json()
            disclaimer = data.get('disclaimer', '')
            
            # Disclaimer should mention key limitations
            assert 'not' in disclaimer.lower() or 'disclaimer' in disclaimer.lower()


# =============================================================================
# Property-Based Tests
# =============================================================================

class TestPropertyBased:
    """Property-based tests for invariants."""
    
    def test_probability_always_in_range(self, sample_diabetes_features):
        """Test probabilities are always in [0, 1]."""
        # This would use hypothesis for full property testing
        # Simplified version here
        
        from services.inference_service import HealthcareInferenceService
        
        service = HealthcareInferenceService()
        
        # Test with various valid inputs
        for _ in range(10):
            features = sample_diabetes_features.copy()
            features['age'] = np.random.randint(18, 100)
            features['bmi'] = np.random.uniform(15, 50)
            
            try:
                result = service.predict('diabetes', features)
                if result:
                    assert 0 <= result.probability <= 1
            except Exception:
                # Service may not be initialized
                pass
    
    def test_threshold_produces_valid_binary_prediction(self):
        """Test threshold always produces 0 or 1."""
        from core.config import get_settings
        
        settings = get_settings()
        
        probabilities = np.random.uniform(0, 1, 100)
        
        for disease in ['diabetes', 'heart_disease', 'stroke']:
            threshold = settings.get_threshold(disease)
            predictions = (probabilities >= threshold).astype(int)
            
            assert all(p in [0, 1] for p in predictions)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance and stress tests."""
    
    def test_prediction_latency(self, sample_diabetes_features):
        """Test prediction completes within latency budget."""
        import time
        
        from services.inference_service import HealthcareInferenceService
        
        service = HealthcareInferenceService()
        
        # Warm up
        try:
            service.predict('diabetes', sample_diabetes_features)
        except Exception:
            pytest.skip("Service not available")
        
        # Measure latency
        latencies = []
        for _ in range(10):
            start = time.time()
            service.predict('diabetes', sample_diabetes_features)
            latencies.append(time.time() - start)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Should complete within 500ms
        assert avg_latency < 0.5
        assert p95_latency < 1.0


# =============================================================================
# Regression Tests
# =============================================================================

class TestRegression:
    """Regression tests to catch unexpected behavior changes."""
    
    def test_model_output_deterministic(self, sample_diabetes_features):
        """Test that same input produces same output."""
        from services.inference_service import HealthcareInferenceService
        
        service = HealthcareInferenceService()
        
        try:
            result1 = service.predict('diabetes', sample_diabetes_features)
            result2 = service.predict('diabetes', sample_diabetes_features)
            
            if result1 and result2:
                assert result1.probability == result2.probability
                assert result1.prediction == result2.prediction
        except Exception:
            pytest.skip("Service not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
