"""
Probability Calibration and Threshold Optimization Module
Critical for healthcare ML where probability reliability matters.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve, 
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
import joblib

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_settings
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ThresholdMetrics:
    """Metrics at a specific threshold."""
    threshold: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    total_cost: float  # Weighted cost of errors


@dataclass
class OptimalThresholdResult:
    """Result of threshold optimization."""
    optimal_threshold: float
    optimization_method: str
    metrics_at_threshold: ThresholdMetrics
    all_thresholds: List[ThresholdMetrics]
    rationale: str


class ThresholdOptimizer:
    """
    Optimize classification threshold for healthcare applications.
    
    Healthcare context requires different thresholds based on:
    - Disease severity (stroke > heart disease > diabetes typically)
    - Cost of false negatives vs false positives
    - Clinical workflow constraints
    
    Default medical ML: OPTIMIZE FOR RECALL (minimize missed diagnoses)
    """
    
    def __init__(
        self,
        false_negative_cost: float = 10.0,
        false_positive_cost: float = 1.0,
        min_recall: float = 0.85  # Healthcare default: catch 85%+ of positive cases
    ):
        """
        Args:
            false_negative_cost: Cost weight for missing a positive case
            false_positive_cost: Cost weight for false alarm
            min_recall: Minimum acceptable recall (sensitivity)
        """
        self.fn_cost = false_negative_cost
        self.fp_cost = false_positive_cost
        self.min_recall = min_recall
    
    def optimize_cost_sensitive(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> OptimalThresholdResult:
        """
        Find threshold minimizing total misclassification cost.
        
        Cost = FN * fn_cost + FP * fp_cost
        
        For healthcare: fn_cost >> fp_cost
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        metrics_list = []
        
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            metrics = self._calculate_metrics(y_true, y_pred, y_proba, t)
            metrics_list.append(metrics)
        
        # Find minimum cost threshold
        costs = [m.total_cost for m in metrics_list]
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        
        return OptimalThresholdResult(
            optimal_threshold=optimal_threshold,
            optimization_method="cost_sensitive",
            metrics_at_threshold=metrics_list[optimal_idx],
            all_thresholds=metrics_list,
            rationale=f"Minimizes cost function with FN:FP ratio of {self.fn_cost}:{self.fp_cost}"
        )
    
    def optimize_recall_constrained(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        min_recall: Optional[float] = None
    ) -> OptimalThresholdResult:
        """
        Find threshold achieving minimum recall while maximizing precision.
        
        Strategy: "Catch at least X% of positive cases, then maximize precision"
        """
        min_recall = min_recall or self.min_recall
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Filter to thresholds meeting recall constraint
        valid_mask = recall[:-1] >= min_recall
        
        if not valid_mask.any():
            # Can't achieve desired recall, return lowest threshold
            logger.warning(
                f"Cannot achieve {min_recall:.0%} recall. "
                f"Max possible recall: {recall.max():.2%}"
            )
            optimal_idx = 0
            rationale = f"Target recall {min_recall:.0%} not achievable. Using lowest threshold."
        else:
            # Among valid thresholds, maximize precision
            valid_precisions = precision[:-1][valid_mask]
            valid_thresholds = thresholds[valid_mask]
            best_idx = np.argmax(valid_precisions)
            optimal_idx = np.where(valid_mask)[0][best_idx]
            rationale = f"Achieves >={min_recall:.0%} recall while maximizing precision"
        
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate full metrics at optimal threshold
        y_pred = (y_proba >= optimal_threshold).astype(int)
        metrics = self._calculate_metrics(y_true, y_pred, y_proba, optimal_threshold)
        
        return OptimalThresholdResult(
            optimal_threshold=optimal_threshold,
            optimization_method="recall_constrained",
            metrics_at_threshold=metrics,
            all_thresholds=[],  # Not computing all for this method
            rationale=rationale
        )
    
    def optimize_f1(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> OptimalThresholdResult:
        """
        Find threshold maximizing F1 score.
        
        Note: F1 may not be optimal for healthcare (doesn't account for
        asymmetric costs), but useful as a baseline.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Calculate F1 for each threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred = (y_proba >= optimal_threshold).astype(int)
        metrics = self._calculate_metrics(y_true, y_pred, y_proba, optimal_threshold)
        
        return OptimalThresholdResult(
            optimal_threshold=optimal_threshold,
            optimization_method="f1_maximization",
            metrics_at_threshold=metrics,
            all_thresholds=[],
            rationale="Maximizes F1 score (harmonic mean of precision and recall)"
        )
    
    def optimize_youden_j(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> OptimalThresholdResult:
        """
        Find threshold maximizing Youden's J statistic.
        
        J = Sensitivity + Specificity - 1 = TPR - FPR
        
        Useful when both sensitivity and specificity matter equally.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Youden's J
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred = (y_proba >= optimal_threshold).astype(int)
        metrics = self._calculate_metrics(y_true, y_pred, y_proba, optimal_threshold)
        
        return OptimalThresholdResult(
            optimal_threshold=optimal_threshold,
            optimization_method="youden_j",
            metrics_at_threshold=metrics,
            all_thresholds=[],
            rationale="Maximizes Youden's J (balanced sensitivity + specificity)"
        )

    # Backward compatibility with older API/tests.
    def optimize(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        method: str = "recall_constrained",
        min_recall: Optional[float] = None,
        fn_cost: Optional[float] = None,
        fp_cost: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        if fn_cost is not None:
            self.fn_cost = fn_cost
        if fp_cost is not None:
            self.fp_cost = fp_cost

        if method == "recall_constrained":
            result = self.optimize_recall_constrained(y_true, y_proba, min_recall=min_recall)
        elif method == "cost_sensitive":
            result = self.optimize_cost_sensitive(y_true, y_proba)
        elif method in {"f1", "optimize_f1"}:
            result = self.optimize_f1(y_true, y_proba)
        elif method == "youden_j":
            result = self.optimize_youden_j(y_true, y_proba)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        metrics = {
            "precision": result.metrics_at_threshold.precision,
            "recall": result.metrics_at_threshold.recall,
            "f1_score": result.metrics_at_threshold.f1_score,
            "specificity": result.metrics_at_threshold.specificity,
            "false_positive_rate": result.metrics_at_threshold.false_positive_rate,
            "false_negative_rate": result.metrics_at_threshold.false_negative_rate,
            "total_cost": result.metrics_at_threshold.total_cost,
        }
        return float(result.optimal_threshold), metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        threshold: float
    ) -> ThresholdMetrics:
        """Calculate comprehensive metrics at a threshold."""
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        total_cost = fn * self.fn_cost + fp * self.fp_cost
        
        return ThresholdMetrics(
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            total_cost=total_cost
        )


class CalibratedHealthcareModel(BaseEstimator, ClassifierMixin):
    """
    Calibrated classifier wrapper for healthcare applications.
    
    Features:
    - Probability calibration (isotonic or sigmoid)
    - Optimized threshold based on clinical requirements
    - Confidence intervals via bootstrapping
    
    Why calibration matters in healthcare:
    - Tree ensembles (XGBoost, RF) push probabilities toward 0/1
    - A "70% probability" from XGBoost might actually be 55%
    - Clinicians need reliable probabilities for shared decision-making
    """
    
    def __init__(
        self,
        base_estimator: BaseEstimator,
        calibration_method: str = "isotonic",
        cv: int = 5,
        threshold_optimizer: Optional[ThresholdOptimizer] = None,
        disease_type: str = "diabetes"
    ):
        """
        Args:
            base_estimator: Trained sklearn-compatible classifier
            calibration_method: 'isotonic' (flexible) or 'sigmoid' (Platt scaling)
            cv: Cross-validation folds for calibration
            threshold_optimizer: ThresholdOptimizer instance
            disease_type: For loading appropriate cost configuration
        """
        self.base_estimator = base_estimator
        self.calibration_method = calibration_method
        self.cv = cv
        self.threshold_optimizer = threshold_optimizer
        self.disease_type = disease_type
        
        self.calibrated_model_ = None
        self.optimal_threshold_ = 0.5
        self.threshold_result_ = None
        self.calibration_metrics_ = {}
    
    def fit(self, X, y, X_cal=None, y_cal=None):
        """
        Fit the calibrated model.
        
        If X_cal/y_cal provided, use them for calibration (recommended).
        Otherwise, use cross-validation calibration.
        
        Args:
            X: Training features
            y: Training labels
            X_cal: Optional separate calibration set features
            y_cal: Optional separate calibration set labels
        """
        settings = get_settings()
        
        # Fit base model
        logger.info(f"Fitting base model: {type(self.base_estimator).__name__}")
        self.base_estimator.fit(X, y)
        
        # Calibrate
        if X_cal is not None and y_cal is not None:
            # Use separate calibration set (preferred)
            logger.info("Calibrating with separate calibration set")
            self.calibrated_model_ = CalibratedClassifierCV(
                self.base_estimator,
                method=self.calibration_method,
                cv="prefit"  # Model already fitted
            )
            self.calibrated_model_.fit(X_cal, y_cal)
            
            # Get calibrated probabilities for threshold optimization
            cal_proba = self.calibrated_model_.predict_proba(X_cal)[:, 1]
            y_for_threshold = y_cal
        else:
            # Use cross-validation calibration
            logger.info(f"Calibrating with {self.cv}-fold CV")
            self.calibrated_model_ = CalibratedClassifierCV(
                self.base_estimator,
                method=self.calibration_method,
                cv=self.cv
            )
            self.calibrated_model_.fit(X, y)
            
            # Get OOF predictions for threshold optimization
            cal_proba = cross_val_predict(
                self.base_estimator, X, y,
                cv=self.cv, method='predict_proba'
            )[:, 1]
            y_for_threshold = y
        
        # Optimize threshold
        if self.threshold_optimizer is None:
            fn_cost = settings.get_fn_cost(self.disease_type)
            self.threshold_optimizer = ThresholdOptimizer(
                false_negative_cost=fn_cost,
                false_positive_cost=1.0
            )
        
        self.threshold_result_ = self.threshold_optimizer.optimize_recall_constrained(
            y_for_threshold, cal_proba
        )
        self.optimal_threshold_ = self.threshold_result_.optimal_threshold
        
        logger.info(
            f"Optimal threshold: {self.optimal_threshold_:.3f} "
            f"(recall: {self.threshold_result_.metrics_at_threshold.recall:.2%}, "
            f"precision: {self.threshold_result_.metrics_at_threshold.precision:.2%})"
        )
        
        # Calculate calibration metrics
        self._calculate_calibration_metrics(y_for_threshold, cal_proba)
        
        return self
    
    def predict_proba(self, X) -> np.ndarray:
        """Return calibrated probabilities."""
        return self.calibrated_model_.predict_proba(X)
    
    def predict(self, X) -> np.ndarray:
        """Return predictions using optimal threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.optimal_threshold_).astype(int)
    
    def predict_with_confidence(self, X, n_bootstrap: int = 100) -> Dict[str, np.ndarray]:
        """
        Return predictions with confidence intervals.
        
        Uses bootstrap sampling of the calibrated model's estimators.
        """
        proba = self.predict_proba(X)[:, 1]
        
        # Bootstrap confidence intervals
        # For CalibratedClassifierCV, we can sample from base estimators
        bootstrap_probas = []
        
        for estimator in self.calibrated_model_.calibrated_classifiers_:
            try:
                p = estimator.predict_proba(X)[:, 1]
                bootstrap_probas.append(p)
            except:
                continue
        
        if len(bootstrap_probas) > 1:
            bootstrap_probas = np.array(bootstrap_probas)
            ci_lower = np.percentile(bootstrap_probas, 2.5, axis=0)
            ci_upper = np.percentile(bootstrap_probas, 97.5, axis=0)
        else:
            # Fallback: estimate CI from probability itself
            std_estimate = np.sqrt(proba * (1 - proba) / 100)  # Rough estimate
            ci_lower = np.clip(proba - 1.96 * std_estimate, 0, 1)
            ci_upper = np.clip(proba + 1.96 * std_estimate, 0, 1)
        
        return {
            "probability": proba,
            "prediction": (proba >= self.optimal_threshold_).astype(int),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "threshold": self.optimal_threshold_
        }
    
    def _calculate_calibration_metrics(self, y_true, y_proba):
        """Calculate calibration quality metrics."""
        
        # Brier score (lower is better, 0 is perfect)
        brier = brier_score_loss(y_true, y_proba)
        
        # Expected Calibration Error (ECE)
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        self.calibration_metrics_ = {
            "brier_score": brier,
            "expected_calibration_error": ece,
            "calibration_curve": {
                "predicted": prob_pred.tolist(),
                "actual": prob_true.tolist()
            }
        }
    
    def get_model_card(self) -> Dict[str, Any]:
        """Return model card with all relevant information."""
        return {
            "base_model": type(self.base_estimator).__name__,
            "calibration_method": self.calibration_method,
            "disease_type": self.disease_type,
            "optimal_threshold": self.optimal_threshold_,
            "threshold_rationale": self.threshold_result_.rationale if self.threshold_result_ else None,
            "metrics_at_threshold": {
                "precision": self.threshold_result_.metrics_at_threshold.precision,
                "recall": self.threshold_result_.metrics_at_threshold.recall,
                "f1_score": self.threshold_result_.metrics_at_threshold.f1_score,
                "specificity": self.threshold_result_.metrics_at_threshold.specificity
            } if self.threshold_result_ else None,
            "calibration_metrics": self.calibration_metrics_
        }
