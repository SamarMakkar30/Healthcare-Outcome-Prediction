"""
Model Training Module - Production Pipeline Training System

Exports complete sklearn Pipelines with preprocessing + model.
Each disease produces: models/{disease}_pipeline.pkl
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression

# Handle imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.preprocessing import (
    build_preprocessor,
    get_feature_config,
    get_feature_names,
    FEATURE_CONFIG
)
from training.evaluation import (
    evaluate_model,
    print_evaluation_report,
    EvaluationResult
)

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingResult:
    """Container for training results."""
    disease: str
    pipeline: Pipeline
    train_metrics: EvaluationResult
    test_metrics: EvaluationResult
    cv_scores: np.ndarray
    model_path: str


# =============================================================================
# Model Builders
# =============================================================================

def build_base_model(model_type: str = "ensemble") -> Any:
    """
    Build the classification model.
    
    Args:
        model_type: 'rf', 'gb', 'lr', or 'ensemble'
        
    Returns:
        Sklearn classifier
    """
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    
    elif model_type == "gb":
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42
        )
    
    elif model_type == "lr":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    
    elif model_type == "ensemble":
        return VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )),
                ("gb", GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )),
                ("lr", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                ))
            ],
            voting="soft"
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Pipeline Builder
# =============================================================================

def build_pipeline(disease: str, model_type: str = "ensemble") -> Pipeline:
    """
    Build complete preprocessing + model pipeline.
    
    Args:
        disease: Disease name
        model_type: Model type to use
        
    Returns:
        sklearn Pipeline ready for fitting
    """
    preprocessor = build_preprocessor(disease)
    model = build_base_model(model_type)
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    return pipeline


# =============================================================================
# Data Loading
# =============================================================================

def load_disease_data(disease: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare data for a disease.
    
    Args:
        disease: Disease name
        
    Returns:
        (X, y) tuple
    """
    config = get_feature_config(disease)
    
    # Map disease to data file
    data_files = {
        "diabetes": DATA_DIR / "raw" / "diabetes_data.csv",
        "heart_disease": DATA_DIR / "raw" / "heart_disease_data.csv",
        "stroke": DATA_DIR / "raw" / "stroke_data.csv"
    }
    
    data_path = data_files.get(disease)
    
    if not data_path or not data_path.exists():
        raise FileNotFoundError(f"Data file not found for {disease}: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Get feature names and target
    feature_names = get_feature_names(disease)
    target = config["target"]
    
    # Handle missing columns gracefully
    available_features = [f for f in feature_names if f in df.columns]
    
    if len(available_features) < len(feature_names):
        missing = set(feature_names) - set(available_features)
        print(f"  Warning: Missing features in data: {missing}")
        print(f"  Available: {available_features}")
    
    # Ensure target exists
    if target not in df.columns:
        # Try common alternative names
        alt_targets = ["target", "label", "outcome", "class"]
        for alt in alt_targets:
            if alt in df.columns:
                df[target] = df[alt]
                break
        else:
            raise ValueError(f"Target column '{target}' not found in data")
    
    X = df[available_features].copy()
    y = df[target].copy()
    
    # Convert target to int
    y = y.astype(int)
    
    return X, y


# =============================================================================
# Training Function
# =============================================================================

def train_disease_model(
    disease: str,
    model_type: str = "ensemble",
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42
) -> TrainingResult:
    """
    Train and save a complete pipeline for a disease.
    
    Args:
        disease: Disease name ('diabetes', 'heart_disease', 'stroke')
        model_type: Model type to use
        test_size: Test set proportion
        cv_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        TrainingResult with pipeline and metrics
    """
    print(f"\n{'#'*60}")
    print(f"#  Training {disease.upper()} Model")
    print(f"{'#'*60}")
    
    # Load data
    print(f"\n[1/5] Loading data...")
    X, y = load_disease_data(disease)
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"  Class distribution: {dict(y.value_counts())}")
    
    # Train/test split
    print(f"\n[2/5] Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build pipeline
    print(f"\n[3/5] Building pipeline...")
    pipeline = build_pipeline(disease, model_type)
    print(f"  Preprocessor: ColumnTransformer")
    print(f"  Model: {model_type}")
    
    # Cross-validation
    print(f"\n[4/5] Cross-validation ({cv_folds} folds)...")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_folds, scoring='recall'
    )
    print(f"  CV Recall: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Fit final model
    print(f"\n[5/5] Training final model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on train set
    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_metrics = evaluate_model(y_train, y_train_pred, y_train_proba)
    
    # Evaluate on test set
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba)
    
    # Print reports
    print_evaluation_report(disease, train_metrics, "Train")
    print_evaluation_report(disease, test_metrics, "Test")
    
    # Check for overfitting
    overfit_gap = train_metrics.accuracy - test_metrics.accuracy
    if overfit_gap > 0.1:
        print(f"  ⚠️  Warning: Possible overfitting (gap={overfit_gap:.3f})")
    
    # Save pipeline
    model_path = MODELS_DIR / f"{disease}_pipeline.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\n✅ Pipeline saved: {model_path}")
    
    return TrainingResult(
        disease=disease,
        pipeline=pipeline,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        cv_scores=cv_scores,
        model_path=str(model_path)
    )


# =============================================================================
# Save Pipeline (Standalone)
# =============================================================================

def save_pipeline(pipeline: Pipeline, disease: str) -> str:
    """
    Save a trained pipeline to disk.
    
    Args:
        pipeline: Fitted sklearn Pipeline
        disease: Disease name
        
    Returns:
        Path to saved file
    """
    model_path = MODELS_DIR / f"{disease}_pipeline.pkl"
    joblib.dump(pipeline, model_path)
    return str(model_path)


def load_pipeline(disease: str) -> Pipeline:
    """
    Load a trained pipeline from disk.
    
    Args:
        disease: Disease name
        
    Returns:
        Fitted sklearn Pipeline
    """
    model_path = MODELS_DIR / f"{disease}_pipeline.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Pipeline not found: {model_path}")
    
    return joblib.load(model_path)


# =============================================================================
# Train All Models
# =============================================================================

def train_all_models(model_type: str = "ensemble") -> Dict[str, TrainingResult]:
    """
    Train pipelines for all diseases.
    
    Args:
        model_type: Model type to use
        
    Returns:
        Dictionary of disease -> TrainingResult
    """
    results = {}
    
    print("\n" + "="*70)
    print("  HEALTHCARE PREDICTION SYSTEM - MODEL TRAINING")
    print("="*70)
    
    for disease in FEATURE_CONFIG.keys():
        try:
            result = train_disease_model(disease, model_type)
            results[disease] = result
        except Exception as e:
            print(f"\n❌ Error training {disease}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("  TRAINING SUMMARY")
    print("="*70)
    
    for disease, result in results.items():
        print(f"\n  {disease.upper()}")
        print(f"    Test Accuracy:  {result.test_metrics.accuracy:.4f}")
        print(f"    Test Recall:    {result.test_metrics.recall:.4f}")
        print(f"    Test ROC-AUC:   {result.test_metrics.roc_auc:.4f}")
        print(f"    Saved to:       {result.model_path}")
    
    print("\n" + "="*70)
    print("  All pipelines saved to models/ directory")
    print("="*70 + "\n")
    
    return results


# =============================================================================
# Inference Helper
# =============================================================================

def predict_from_dict(disease: str, input_data: Dict[str, Any]) -> Tuple[int, float]:
    """
    Make prediction from raw input dictionary.
    
    Args:
        disease: Disease name
        input_data: Raw feature dictionary
        
    Returns:
        (prediction, probability) tuple
    """
    pipeline = load_pipeline(disease)
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Predict
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0, 1]
    
    return int(prediction), float(probability)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train healthcare prediction models")
    parser.add_argument(
        "--disease",
        choices=["diabetes", "heart_disease", "stroke", "all"],
        default="all",
        help="Disease to train (default: all)"
    )
    parser.add_argument(
        "--model-type",
        choices=["rf", "gb", "lr", "ensemble"],
        default="ensemble",
        help="Model type (default: ensemble)"
    )
    
    args = parser.parse_args()
    
    if args.disease == "all":
        train_all_models(args.model_type)
    else:
        train_disease_model(args.disease, args.model_type)
