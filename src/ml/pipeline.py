"""
Production ML Pipeline Module
Implements sklearn pipelines to prevent data leakage and ensure reproducibility.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import joblib

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import FEATURE_CONFIG, get_settings


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline."""
    numeric_features: List[str]
    categorical_features: List[str]
    target: str
    impute_strategy_numeric: str = "median"
    impute_strategy_categorical: str = "constant"
    categorical_fill_value: str = "missing"
    use_smote: bool = True
    smote_k_neighbors: int = 5


class ClinicalRangeValidator(BaseEstimator, TransformerMixin):
    """
    Validates and clips features to clinical ranges.
    Prevents out-of-distribution inputs from causing erratic predictions.
    """
    
    def __init__(self, clinical_ranges: Dict[str, Tuple[float, float]] | str):
        if isinstance(clinical_ranges, str):
            self.clinical_ranges = FEATURE_CONFIG.get(clinical_ranges, {}).get("clinical_ranges", {})
        else:
            self.clinical_ranges = clinical_ranges
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_validated = X.copy()
        
        for feature, (min_val, max_val) in self.clinical_ranges.items():
            if feature in X_validated.columns:
                X_validated[feature] = X_validated[feature].clip(min_val, max_val)
        
        return X_validated


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates clinically-relevant derived features.
    All transformations are fit on training data only.
    """
    
    def __init__(self, disease_type: str):
        self.disease_type = disease_type
        self.fitted_ = False
    
    def fit(self, X, y=None):
        self.fitted_ = True
        return self
    
    def transform(self, X):
        X_eng = X.copy()
        
        if self.disease_type == "diabetes":
            X_eng = self._engineer_diabetes_features(X_eng)
        elif self.disease_type == "heart_disease":
            X_eng = self._engineer_heart_features(X_eng)
        elif self.disease_type == "stroke":
            X_eng = self._engineer_stroke_features(X_eng)
        
        return X_eng
    
    def _engineer_diabetes_features(self, df):
        """Create diabetes-specific features."""
        if "bmi" in df.columns:
            # BMI risk categories (clinical standards)
            df["bmi_risk_score"] = pd.cut(
                df["bmi"],
                bins=[0, 18.5, 25, 30, 35, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(float).fillna(1)
        
        if "age" in df.columns:
            # Age-based risk factor
            df["age_risk"] = (df["age"] > 45).astype(int)
        
        if "glucose" in df.columns and "insulin" in df.columns:
            # HOMA-IR approximation (insulin resistance indicator)
            df["glucose_insulin_ratio"] = df["glucose"] / (df["insulin"] + 1)
        
        return df
    
    def _engineer_heart_features(self, df):
        """Create heart disease-specific features."""
        if "age" in df.columns and "max_heart_rate" in df.columns:
            # Heart rate reserve
            predicted_max_hr = 220 - df["age"]
            df["hr_reserve"] = predicted_max_hr - df["max_heart_rate"]
            df["hr_achievement_ratio"] = df["max_heart_rate"] / predicted_max_hr
        
        if "resting_bp" in df.columns:
            # Hypertension stages
            df["hypertension_stage"] = pd.cut(
                df["resting_bp"],
                bins=[0, 120, 130, 140, 180, 300],
                labels=[0, 1, 2, 3, 4]
            ).astype(float).fillna(0)
        
        return df
    
    def _engineer_stroke_features(self, df):
        """Create stroke-specific features."""
        if "age" in df.columns:
            # Age is strongest predictor for stroke
            df["age_squared"] = df["age"] ** 2 / 1000  # Scaled for numerical stability
        
        if "hypertension" in df.columns and "heart_disease" in df.columns:
            # Cardiovascular comorbidity score
            df["cv_comorbidity"] = (
                df["hypertension"].astype(int) + 
                df["heart_disease"].astype(int)
            )
        
        return df


def create_preprocessing_pipeline(
    config: PipelineConfig,
    disease_type: str
) -> ColumnTransformer:
    """
    Create preprocessing pipeline that:
    - Fits ONLY on training data (no leakage)
    - Handles unseen categories gracefully
    - Serializes with the model
    
    Args:
        config: Pipeline configuration
        disease_type: Type of disease for feature engineering
    
    Returns:
        ColumnTransformer with numeric and categorical pipelines
    """
    
    # Numeric preprocessing: impute -> scale
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy=config.impute_strategy_numeric)),
        ("scaler", StandardScaler())
    ])
    
    # Categorical preprocessing: impute -> encode
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(
            strategy=config.impute_strategy_categorical,
            fill_value=config.categorical_fill_value
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",  # Critical: handles unseen categories
            sparse_output=False,
            drop="if_binary"  # Avoid multicollinearity for binary features
        ))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.numeric_features),
            ("cat", categorical_transformer, config.categorical_features)
        ],
        remainder="drop",  # Explicitly drop unlisted features
        verbose_feature_names_out=True
    )
    
    return preprocessor


def create_full_pipeline(
    model: BaseEstimator | str,
    config: Optional[PipelineConfig] = None,
    disease_type: Optional[str] = None,
    use_resampling: bool = True,
    resampling_strategy: str = "smote"
) -> Pipeline:
    """
    Create complete ML pipeline with preprocessing and model.
    
    For imbalanced data (common in healthcare), includes optional resampling.
    Resampling happens AFTER preprocessing but ONLY on training data.
    
    Args:
        model: Scikit-learn compatible estimator
        config: Pipeline configuration
        disease_type: Disease type for feature engineering
        use_resampling: Whether to use resampling for class imbalance
        resampling_strategy: 'smote', 'adasyn', or 'undersample'
    
    Returns:
        Complete pipeline ready for fit/predict
    """
    
    # Legacy compatibility: create_full_pipeline("diabetes")
    if isinstance(model, str) and config is None and disease_type is None:
        disease_type = model
        config = get_pipeline_config(disease_type)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=get_settings().random_state,
            class_weight="balanced"
        )
        use_resampling = False

    if config is None or disease_type is None:
        raise ValueError("config and disease_type are required")

    # Get clinical ranges for validation
    clinical_ranges = FEATURE_CONFIG.get(disease_type, {}).get("clinical_ranges", {})
    
    preprocessor = create_preprocessing_pipeline(config, disease_type)
    
    if use_resampling:
        # Use imblearn Pipeline for resampling support
        if resampling_strategy == "smote":
            sampler = SMOTE(
                random_state=get_settings().random_state,
                k_neighbors=config.smote_k_neighbors
            )
        elif resampling_strategy == "adasyn":
            sampler = ADASYN(random_state=get_settings().random_state)
        elif resampling_strategy == "undersample":
            sampler = RandomUnderSampler(random_state=get_settings().random_state)
        else:
            raise ValueError(f"Unknown resampling strategy: {resampling_strategy}")
        
        pipeline = ImbPipeline([
            ("clinical_validator", ClinicalRangeValidator(clinical_ranges)),
            ("feature_engineer", FeatureEngineer(disease_type)),
            ("preprocessor", preprocessor),
            ("sampler", sampler),
            ("model", model)
        ])
    else:
        pipeline = Pipeline([
            ("clinical_validator", ClinicalRangeValidator(clinical_ranges)),
            ("feature_engineer", FeatureEngineer(disease_type)),
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    
    return pipeline


def get_pipeline_config(disease_type: str) -> PipelineConfig:
    """Get pipeline configuration for a disease type."""
    
    if disease_type not in FEATURE_CONFIG:
        raise ValueError(f"Unknown disease type: {disease_type}")
    
    config = FEATURE_CONFIG[disease_type]
    
    return PipelineConfig(
        numeric_features=config["numeric_features"],
        categorical_features=config["categorical_features"],
        target=config["target"]
    )


def save_pipeline(pipeline: Pipeline, filepath: str) -> None:
    """Save pipeline with compression."""
    joblib.dump(pipeline, filepath, compress=3)


def load_pipeline(filepath: str) -> Pipeline:
    """Load saved pipeline."""
    return joblib.load(filepath)
