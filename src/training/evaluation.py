"""
Evaluation module - Metrics calculation and reporting.
"""

from typing import Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (excludes confusion matrix)."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc
        }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> EvaluationResult:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
        
    Returns:
        EvaluationResult with all metrics
    """
    # Handle edge case where only one class in y_true
    unique_classes = np.unique(y_true)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Use zero_division=0 for edge cases
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC requires both classes and probabilities
    if y_proba is not None and len(unique_classes) > 1:
        roc_auc = roc_auc_score(y_true, y_proba)
    else:
        roc_auc = 0.0
    
    cm = confusion_matrix(y_true, y_pred)
    
    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=cm
    )


def print_evaluation_report(
    disease: str,
    result: EvaluationResult,
    dataset_split: str = "Test"
) -> None:
    """
    Print formatted evaluation report.
    
    Args:
        disease: Disease name
        result: EvaluationResult object
        dataset_split: 'Train' or 'Test'
    """
    print(f"\n{'='*60}")
    print(f"  {disease.upper()} - {dataset_split} Set Evaluation")
    print(f"{'='*60}")
    print(f"  Accuracy:   {result.accuracy:.4f}")
    print(f"  Precision:  {result.precision:.4f}")
    print(f"  Recall:     {result.recall:.4f}")
    print(f"  F1 Score:   {result.f1:.4f}")
    print(f"  ROC-AUC:    {result.roc_auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {result.confusion_matrix}")
    print(f"{'='*60}\n")


def compare_train_test(
    train_result: EvaluationResult,
    test_result: EvaluationResult
) -> Dict[str, float]:
    """
    Compare train and test metrics to detect overfitting.
    
    Returns:
        Dictionary with metric differences (train - test)
    """
    return {
        "accuracy_diff": train_result.accuracy - test_result.accuracy,
        "precision_diff": train_result.precision - test_result.precision,
        "recall_diff": train_result.recall - test_result.recall,
        "f1_diff": train_result.f1 - test_result.f1,
        "roc_auc_diff": train_result.roc_auc - test_result.roc_auc
    }
