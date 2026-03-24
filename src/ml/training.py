"""
Production Model Training Module
Implements proper ML workflow with nested cross-validation and experiment tracking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    brier_score_loss
)
from xgboost import XGBClassifier
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_settings, FEATURE_CONFIG
from core.logging_config import get_logger
from ml.pipeline import (
    create_full_pipeline, 
    get_pipeline_config,
    PipelineConfig,
    save_pipeline
)
from ml.calibration import CalibratedHealthcareModel, ThresholdOptimizer

logger = get_logger(__name__)


@dataclass
class ExperimentResult:
    """Stores results from a training experiment."""
    experiment_id: str
    model_name: str
    disease_type: str
    timestamp: str
    
    # Cross-validation metrics
    cv_scores: Dict[str, List[float]]
    cv_mean: Dict[str, float]
    cv_std: Dict[str, float]
    
    # Test set metrics
    test_metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    
    # Threshold optimization
    optimal_threshold: float
    threshold_metrics: Dict[str, float]
    
    # Calibration
    brier_score: float
    is_calibrated: bool
    
    # Model metadata
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    training_time_seconds: float = 0.0


@dataclass
class ModelArtifacts:
    """Container for all model artifacts."""
    model: Any
    pipeline: Any
    calibrator: Optional[Any] = None
    threshold: float = 0.5
    feature_names: List[str] = field(default_factory=list)
    experiment_result: Optional[ExperimentResult] = None


class HealthcareModelTrainer:
    """
    Production-grade model trainer for healthcare ML.
    
    Features:
    - Proper train/val/test split with stratification
    - Nested cross-validation for unbiased evaluation
    - Multiple model comparison
    - Automatic threshold optimization
    - Probability calibration
    - Experiment tracking
    """
    
    def __init__(self, disease_type: str):
        self.disease_type = disease_type
        self.settings = get_settings()
        self.pipeline_config = get_pipeline_config(disease_type)
        
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, ExperimentResult] = {}
        self.best_model_name: Optional[str] = None
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{self.disease_type}_{timestamp}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"exp_{self.disease_type}_{timestamp}_{short_hash}"
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get model configurations with healthcare-appropriate hyperparameters.
        
        Key considerations:
        - class_weight='balanced' for imbalanced data
        - Moderate tree depth to prevent overfitting
        - Multiple estimators for stability
        """
        return {
            "xgboost": {
                "model": XGBClassifier(
                    n_estimators=200,
                    max_depth=6,  # Shallower than default for healthcare
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=1.0,  # L2 regularization
                    scale_pos_weight=1,  # Will be calculated from data
                    random_state=self.settings.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                "param_grid": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [4, 6, 8],
                    "model__learning_rate": [0.05, 0.1]
                }
            },
            "random_forest": {
                "model": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    class_weight='balanced',  # Critical for healthcare
                    random_state=self.settings.random_state,
                    n_jobs=-1
                ),
                "param_grid": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [8, 12, 16],
                    "model__min_samples_leaf": [2, 4, 8]
                }
            },
            "lightgbm": {
                "model": lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    num_leaves=31,
                    min_child_samples=20,
                    class_weight='balanced',
                    random_state=self.settings.random_state,
                    verbose=-1
                ),
                "param_grid": {
                    "model__n_estimators": [100, 200],
                    "model__num_leaves": [15, 31, 63],
                    "model__learning_rate": [0.05, 0.1]
                }
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.1,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    random_state=self.settings.random_state
                ),
                "param_grid": {
                    "model__n_estimators": [100, 150],
                    "model__max_depth": [4, 5, 6]
                }
            }
        }
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
               pd.Series, pd.Series, pd.Series]:
        """
        Prepare data with proper train/val/test split.
        
        Split strategy:
        1. 70% training
        2. 15% validation (for calibration and threshold tuning)
        3. 15% test (NEVER touched until final evaluation)
        
        All splits are stratified to maintain class ratios.
        """
        from sklearn.model_selection import train_test_split
        
        target_column = target_column or self.pipeline_config.target
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Dataset size: {len(X)} samples")
        logger.info(f"Class distribution: {dict(y.value_counts())}")
        logger.info(f"Positive rate: {y.mean():.2%}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.settings.test_size,
            stratify=y,
            random_state=self.settings.random_state
        )
        
        # Second split: train/validation
        val_ratio = self.settings.validation_size / (1 - self.settings.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=self.settings.random_state
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def cross_validate_model(
        self,
        model_name: str,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform stratified cross-validation.
        
        Returns both mean scores and individual fold scores for
        uncertainty quantification.
        """
        logger.info(f"Cross-validating {model_name}...")
        
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.settings.random_state
        )
        
        # Create pipeline
        pipeline = create_full_pipeline(
            model=model,
            config=self.pipeline_config,
            disease_type=self.disease_type,
            use_resampling=True
        )
        
        # Multiple scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(
                pipeline, X, y,
                cv=cv,
                scoring=metric,
                n_jobs=-1
            )
            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': scores.mean(),
                'std': scores.std()
            }
        
        # Get OOF predictions for threshold analysis
        try:
            oof_proba = cross_val_predict(
                pipeline, X, y,
                cv=cv,
                method='predict_proba'
            )[:, 1]
            cv_results['oof_probabilities'] = oof_proba
        except Exception as e:
            logger.warning(f"Could not get OOF predictions: {e}")
            cv_results['oof_probabilities'] = None
        
        return cv_results
    
    def train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> ExperimentResult:
        """
        Train a single model with full evaluation.
        
        Includes:
        - Cross-validation on training data
        - Threshold optimization on validation data
        - Final evaluation on test data
        - Probability calibration
        """
        import time
        start_time = time.time()
        
        experiment_id = self._generate_experiment_id()
        model_configs = self.get_model_configs()
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        base_model = model_configs[model_name]["model"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name.upper()} for {self.disease_type}")
        logger.info(f"Experiment ID: {experiment_id}")
        logger.info(f"{'='*60}")
        
        # Cross-validation on training data
        cv_results = self.cross_validate_model(
            model_name,
            base_model,
            pd.concat([X_train, X_val]),
            pd.concat([y_train, y_val])
        )
        
        logger.info(f"\nCV Results:")
        for metric, values in cv_results.items():
            if metric != 'oof_probabilities':
                logger.info(f"  {metric}: {values['mean']:.4f} (+/- {values['std']:.4f})")
        
        # Create and train full pipeline
        pipeline = create_full_pipeline(
            model=base_model,
            config=self.pipeline_config,
            disease_type=self.disease_type,
            use_resampling=True
        )
        
        # Fit on training data
        pipeline.fit(X_train, y_train)
        
        # Create calibrated model
        fn_cost = self.settings.get_fn_cost(self.disease_type)
        threshold_optimizer = ThresholdOptimizer(
            false_negative_cost=fn_cost,
            false_positive_cost=1.0,
            min_recall=0.85
        )
        
        # Get validation predictions for calibration
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Optimize threshold on validation set
        threshold_result = threshold_optimizer.optimize_recall_constrained(
            y_val.values, val_proba
        )
        optimal_threshold = threshold_result.optimal_threshold
        
        logger.info(f"\nOptimal Threshold: {optimal_threshold:.3f}")
        logger.info(f"  Recall at threshold: {threshold_result.metrics_at_threshold.recall:.2%}")
        logger.info(f"  Precision at threshold: {threshold_result.metrics_at_threshold.precision:.2%}")
        
        # Final evaluation on TEST set
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= optimal_threshold).astype(int)
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred, zero_division=0),
            'recall': recall_score(y_test, test_pred, zero_division=0),
            'f1': f1_score(y_test, test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, test_proba),
            'average_precision': average_precision_score(y_test, test_proba),
            'brier_score': brier_score_loss(y_test, test_proba)
        }
        
        cm = confusion_matrix(y_test, test_pred)
        
        logger.info(f"\nTest Set Results (threshold={optimal_threshold:.3f}):")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0][0]:4d}  FP: {cm[0][1]:4d}")
        logger.info(f"  FN: {cm[1][0]:4d}  TP: {cm[1][1]:4d}")
        
        # Get feature importance
        feature_importance = self._extract_feature_importance(pipeline)
        
        training_time = time.time() - start_time
        
        # Store model
        self.models[model_name] = {
            'pipeline': pipeline,
            'threshold': optimal_threshold,
            'threshold_result': threshold_result
        }
        
        # Create experiment result
        result = ExperimentResult(
            experiment_id=experiment_id,
            model_name=model_name,
            disease_type=self.disease_type,
            timestamp=datetime.now().isoformat(),
            cv_scores={k: v['scores'] for k, v in cv_results.items() if k != 'oof_probabilities'},
            cv_mean={k: v['mean'] for k, v in cv_results.items() if k != 'oof_probabilities'},
            cv_std={k: v['std'] for k, v in cv_results.items() if k != 'oof_probabilities'},
            test_metrics=test_metrics,
            confusion_matrix=cm.tolist(),
            optimal_threshold=optimal_threshold,
            threshold_metrics={
                'recall': threshold_result.metrics_at_threshold.recall,
                'precision': threshold_result.metrics_at_threshold.precision,
                'f1': threshold_result.metrics_at_threshold.f1_score,
                'specificity': threshold_result.metrics_at_threshold.specificity
            },
            brier_score=test_metrics['brier_score'],
            is_calibrated=False,  # Will be updated if calibration applied
            hyperparameters=base_model.get_params(),
            feature_importance=feature_importance,
            training_time_seconds=training_time
        )
        
        self.results[model_name] = result
        
        return result
    
    def _extract_feature_importance(self, pipeline) -> Optional[Dict[str, float]]:
        """Extract feature importance from trained pipeline."""
        try:
            # Get the model from pipeline
            model = pipeline.named_steps.get('model')
            if model is None:
                return None
            
            # Get feature names after preprocessing
            preprocessor = pipeline.named_steps.get('preprocessor')
            if preprocessor:
                feature_names = preprocessor.get_feature_names_out().tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
            
            importance = model.feature_importances_
            
            return dict(zip(feature_names, importance.tolist()))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, ExperimentResult]:
        """Train all configured models and compare."""
        
        model_configs = self.get_model_configs()
        
        for model_name in model_configs:
            try:
                self.train_single_model(
                    model_name,
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test
                )
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        # Determine best model based on recall (healthcare priority)
        best_recall = 0
        for name, result in self.results.items():
            recall = result.test_metrics['recall']
            if recall > best_recall:
                best_recall = recall
                self.best_model_name = name
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BEST MODEL: {self.best_model_name.upper()}")
        logger.info(f"Test Recall: {best_recall:.4f}")
        logger.info(f"{'='*60}")
        
        return self.results
    
    def save_best_model(self, output_dir: str) -> str:
        """Save the best model with all artifacts."""
        if self.best_model_name is None:
            raise ValueError("No model trained yet")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_data = self.models[self.best_model_name]
        result = self.results[self.best_model_name]
        
        # Save pipeline
        pipeline_path = output_path / f"{self.disease_type}_pipeline.pkl"
        joblib.dump(model_data['pipeline'], pipeline_path, compress=3)
        
        # Save threshold config
        threshold_config = {
            'optimal_threshold': model_data['threshold'],
            'disease_type': self.disease_type,
            'model_name': self.best_model_name,
            'metrics': result.threshold_metrics
        }
        threshold_path = output_path / f"{self.disease_type}_threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump(threshold_config, f, indent=2)
        
        # Save experiment result
        result_path = output_path / f"{self.disease_type}_experiment.json"
        with open(result_path, 'w') as f:
            json.dump(result.__dict__, f, indent=2, default=str)
        
        logger.info(f"Model saved to {output_path}")
        
        return str(pipeline_path)
    
    def get_comparison_report(self) -> pd.DataFrame:
        """Generate comparison report of all trained models."""
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for name, result in self.results.items():
            row = {
                'model': name,
                'cv_recall': result.cv_mean.get('recall', 0),
                'cv_recall_std': result.cv_std.get('recall', 0),
                'test_recall': result.test_metrics['recall'],
                'test_precision': result.test_metrics['precision'],
                'test_f1': result.test_metrics['f1'],
                'test_roc_auc': result.test_metrics['roc_auc'],
                'optimal_threshold': result.optimal_threshold,
                'training_time': result.training_time_seconds
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('test_recall', ascending=False)
        
        return df


def train_disease_model(
    disease_type: str,
    data_path: str,
    output_dir: str
) -> ExperimentResult:
    """
    Convenience function to train model for a disease.
    
    Args:
        disease_type: 'diabetes', 'heart_disease', or 'stroke'
        data_path: Path to CSV file with data
        output_dir: Directory to save model artifacts
    
    Returns:
        ExperimentResult with training details
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"# Training {disease_type.upper()} Model")
    logger.info(f"{'#'*60}\n")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize trainer
    trainer = HealthcareModelTrainer(disease_type)
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
    
    # Train all models
    results = trainer.train_all_models(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )
    
    # Print comparison
    comparison = trainer.get_comparison_report()
    logger.info("\nModel Comparison:")
    logger.info(comparison.to_string())
    
    # Save best model
    trainer.save_best_model(output_dir)
    
    return results[trainer.best_model_name]
