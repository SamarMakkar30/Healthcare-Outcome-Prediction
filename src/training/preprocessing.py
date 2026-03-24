"""
Preprocessing module - ColumnTransformer builders for each disease.
All preprocessing is encapsulated in sklearn transformers for pipeline integration.
"""

from typing import Dict, List, Tuple, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np


# =============================================================================
# Feature Configuration per Disease
# =============================================================================

FEATURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "diabetes": {
        "numeric_features": [
            "age", "bmi", "blood_pressure", "glucose", "insulin",
            "physical_activity", "sleep_hours", "stress_level"
        ],
        "categorical_features": ["gender", "family_history", "smoking", "alcohol"],
        "target": "diabetes"
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
        "target": "heart_disease"
    },
    "stroke": {
        "numeric_features": [
            "age", "avg_glucose_level", "bmi", "physical_activity", "alcohol_intake"
        ],
        "categorical_features": [
            "gender", "hypertension", "heart_disease", "ever_married",
            "work_type", "residence_type", "smoking_status"
        ],
        "target": "stroke"
    }
}


def get_feature_config(disease: str) -> Dict[str, Any]:
    """Get feature configuration for a disease."""
    if disease not in FEATURE_CONFIG:
        raise ValueError(f"Unknown disease: {disease}. Available: {list(FEATURE_CONFIG.keys())}")
    return FEATURE_CONFIG[disease]


def get_feature_names(disease: str) -> List[str]:
    """Get all feature names (numeric + categorical) for a disease."""
    config = get_feature_config(disease)
    return config["numeric_features"] + config["categorical_features"]


# =============================================================================
# Preprocessing Transformers
# =============================================================================

def build_numeric_transformer() -> Pipeline:
    """
    Build numeric feature transformer.
    
    Steps:
    1. Impute missing values with median
    2. Scale to zero mean, unit variance
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def build_categorical_transformer() -> Pipeline:
    """
    Build categorical feature transformer.
    
    Steps:
    1. Impute missing values with most frequent
    2. One-hot encode with unknown handling
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])


def build_preprocessor(disease: str) -> ColumnTransformer:
    """
    Build complete preprocessing ColumnTransformer for a disease.
    
    Args:
        disease: One of 'diabetes', 'heart_disease', 'stroke'
        
    Returns:
        ColumnTransformer with numeric and categorical pipelines
    """
    config = get_feature_config(disease)
    
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    
    transformers = []
    
    # Add numeric transformer if we have numeric features
    if numeric_features:
        transformers.append(
            ("numeric", build_numeric_transformer(), numeric_features)
        )
    
    # Add categorical transformer if we have categorical features
    if categorical_features:
        transformers.append(
            ("categorical", build_categorical_transformer(), categorical_features)
        )
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop any columns not specified
        verbose_feature_names_out=False
    )
    
    return preprocessor


def validate_input_features(df, disease: str) -> None:
    """
    Validate that input DataFrame has required features.
    
    Raises:
        ValueError: If required features are missing
    """
    required = get_feature_names(disease)
    missing = set(required) - set(df.columns)
    
    if missing:
        raise ValueError(f"Missing required features for {disease}: {missing}")
