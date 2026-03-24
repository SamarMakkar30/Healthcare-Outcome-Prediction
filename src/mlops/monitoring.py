"""
Model Monitoring and Drift Detection
Production monitoring for healthcare ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path
from scipy import stats

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_settings
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionLog:
    """Log entry for a single prediction."""
    timestamp: str
    disease_type: str
    features: Dict[str, Any]
    probability: float
    prediction: int
    threshold: float
    latency_ms: float
    model_version: str


@dataclass
class DriftReport:
    """Report of drift detection results."""
    timestamp: str
    disease_type: str
    feature_drift: Dict[str, Dict[str, float]]  # feature -> {statistic, p_value, drift_detected}
    prediction_drift: Dict[str, float]
    overall_drift_detected: bool
    recommendations: List[str]

    @property
    def has_drift(self) -> bool:
        """Backward-compatible alias for legacy tests/code."""
        return self.overall_drift_detected


class PredictionMonitor:
    """
    Monitor predictions for data drift and model performance.
    
    Features:
    - Log predictions for analysis
    - Detect feature distribution drift
    - Detect prediction distribution drift
    - Alert on significant changes
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        window_size: int = 1000,
        drift_threshold: float = 0.05,
        monitoring_path: str = "./monitoring"
    ):
        """
        Args:
            reference_data: Baseline data for drift comparison
            window_size: Number of recent predictions to keep
            drift_threshold: P-value threshold for drift detection
            monitoring_path: Path to store monitoring data
        """
        self.reference_data = reference_data
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.monitoring_path = Path(monitoring_path)
        self.monitoring_path.mkdir(parents=True, exist_ok=True)
        
        # Recent predictions buffer (per disease type)
        self._prediction_buffers: Dict[str, deque] = {}
        self._feature_buffers: Dict[str, deque] = {}
        
        # Reference statistics
        self._reference_stats: Dict[str, Dict[str, Any]] = {}
    
    def set_reference_data(self, disease_type: str, data: pd.DataFrame):
        """Set reference data for a disease type."""
        self._reference_stats[disease_type] = self._compute_statistics(data)
        logger.info(f"Set reference data for {disease_type}: {len(data)} samples")
    
    def _compute_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics for drift comparison."""
        stats_dict = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                "mean": data[col].mean(),
                "std": data[col].std(),
                "median": data[col].median(),
                "q25": data[col].quantile(0.25),
                "q75": data[col].quantile(0.75),
                "min": data[col].min(),
                "max": data[col].max()
            }
        
        return stats_dict
    
    def log_prediction(self, log: PredictionLog):
        """Log a prediction for monitoring."""
        disease_type = log.disease_type
        
        # Initialize buffers if needed
        if disease_type not in self._prediction_buffers:
            self._prediction_buffers[disease_type] = deque(maxlen=self.window_size)
            self._feature_buffers[disease_type] = deque(maxlen=self.window_size)
        
        # Add to buffers
        self._prediction_buffers[disease_type].append({
            "timestamp": log.timestamp,
            "probability": log.probability,
            "prediction": log.prediction
        })
        
        self._feature_buffers[disease_type].append(log.features)
        
        # Periodically save to disk
        if len(self._prediction_buffers[disease_type]) % 100 == 0:
            self._save_monitoring_data(disease_type)

    def record_prediction(
        self,
        features: Dict[str, Any],
        prediction: int,
        probability: float,
        disease_type: str,
        threshold: float = 0.5,
        latency_ms: float = 0.0,
        model_version: str = "unknown",
    ):
        """Backward-compatible wrapper for older monitor API."""
        log = PredictionLog(
            timestamp=datetime.now().isoformat(),
            disease_type=disease_type,
            features=features,
            probability=float(probability),
            prediction=int(prediction),
            threshold=float(threshold),
            latency_ms=float(latency_ms),
            model_version=model_version,
        )
        self.log_prediction(log)

    def check_drift(self, disease_type: str) -> DriftReport:
        """Backward-compatible wrapper for older monitor API."""
        return self.generate_drift_report(disease_type)
    
    def _save_monitoring_data(self, disease_type: str):
        """Save monitoring data to disk."""
        data_path = self.monitoring_path / f"{disease_type}_predictions.json"
        
        data = {
            "predictions": list(self._prediction_buffers[disease_type]),
            "updated_at": datetime.now().isoformat()
        }
        
        with open(data_path, 'w') as f:
            json.dump(data, f)
    
    def detect_feature_drift(
        self,
        disease_type: str,
        current_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Detect drift in feature distributions using KS test.
        
        Returns:
            Dictionary of {feature: {statistic, p_value, drift_detected}}
        """
        if disease_type not in self._reference_stats:
            logger.warning(f"No reference data for {disease_type}")
            return {}
        
        # Use buffered data if current_data not provided
        if current_data is None:
            if disease_type not in self._feature_buffers:
                return {}
            current_data = pd.DataFrame(list(self._feature_buffers[disease_type]))
        
        if len(current_data) < 30:
            logger.warning("Insufficient data for drift detection")
            return {}
        
        drift_results = {}
        reference = self._reference_stats[disease_type]
        
        for feature in current_data.select_dtypes(include=[np.number]).columns:
            if feature not in reference:
                continue
            
            # Kolmogorov-Smirnov test
            ref_mean = reference[feature]["mean"]
            ref_std = reference[feature]["std"]
            
            if ref_std > 0:
                # Normalize current data using reference statistics
                current_normalized = (current_data[feature] - ref_mean) / ref_std
                
                # Compare to standard normal
                ks_stat, p_value = stats.kstest(current_normalized, 'norm')
                
                drift_results[feature] = {
                    "statistic": float(ks_stat),
                    "p_value": float(p_value),
                    "drift_detected": p_value < self.drift_threshold,
                    "current_mean": float(current_data[feature].mean()),
                    "reference_mean": float(ref_mean)
                }
        
        return drift_results
    
    def detect_prediction_drift(self, disease_type: str) -> Dict[str, float]:
        """
        Detect drift in prediction distribution.
        
        Monitors:
        - Mean prediction probability
        - Positive prediction rate
        - Probability distribution shift
        """
        if disease_type not in self._prediction_buffers:
            return {}
        
        buffer = list(self._prediction_buffers[disease_type])
        if len(buffer) < 100:
            return {}
        
        # Split into recent and older predictions
        split_point = len(buffer) // 2
        older = [p["probability"] for p in buffer[:split_point]]
        recent = [p["probability"] for p in buffer[split_point:]]
        
        # Mann-Whitney U test for distribution shift
        mw_stat, mw_p = stats.mannwhitneyu(older, recent, alternative='two-sided')
        
        # Calculate prediction rates
        older_pos_rate = np.mean([p["prediction"] for p in buffer[:split_point]])
        recent_pos_rate = np.mean([p["prediction"] for p in buffer[split_point:]])
        
        return {
            "mann_whitney_statistic": float(mw_stat),
            "mann_whitney_p_value": float(mw_p),
            "prediction_drift_detected": mw_p < self.drift_threshold,
            "older_mean_probability": float(np.mean(older)),
            "recent_mean_probability": float(np.mean(recent)),
            "older_positive_rate": float(older_pos_rate),
            "recent_positive_rate": float(recent_pos_rate),
            "positive_rate_change": float(recent_pos_rate - older_pos_rate)
        }
    
    def generate_drift_report(self, disease_type: str) -> DriftReport:
        """Generate comprehensive drift report."""
        feature_drift = self.detect_feature_drift(disease_type)
        prediction_drift = self.detect_prediction_drift(disease_type)
        
        # Determine overall drift
        feature_drift_detected = any(
            d.get("drift_detected", False) 
            for d in feature_drift.values()
        )
        prediction_drift_detected = prediction_drift.get("prediction_drift_detected", False)
        
        overall_drift = feature_drift_detected or prediction_drift_detected
        
        # Generate recommendations
        recommendations = []
        
        if feature_drift_detected:
            drifted_features = [
                f for f, d in feature_drift.items() 
                if d.get("drift_detected")
            ]
            recommendations.append(
                f"Feature drift detected in: {', '.join(drifted_features)}. "
                "Consider investigating data collection changes."
            )
        
        if prediction_drift_detected:
            recommendations.append(
                "Prediction distribution has shifted. "
                "Consider model retraining or threshold adjustment."
            )
        
        if overall_drift:
            recommendations.append(
                "Model performance may have degraded. "
                "Recommend validation on recent labeled data."
            )
        
        return DriftReport(
            timestamp=datetime.now().isoformat(),
            disease_type=disease_type,
            feature_drift=feature_drift,
            prediction_drift=prediction_drift,
            overall_drift_detected=overall_drift,
            recommendations=recommendations
        )
    
    def get_prediction_statistics(self, disease_type: str) -> Dict[str, Any]:
        """Get statistics about recent predictions."""
        if disease_type not in self._prediction_buffers:
            return {}
        
        buffer = list(self._prediction_buffers[disease_type])
        if not buffer:
            return {}
        
        probabilities = [p["probability"] for p in buffer]
        predictions = [p["prediction"] for p in buffer]
        
        return {
            "total_predictions": len(buffer),
            "mean_probability": float(np.mean(probabilities)),
            "std_probability": float(np.std(probabilities)),
            "positive_rate": float(np.mean(predictions)),
            "probability_percentiles": {
                "p10": float(np.percentile(probabilities, 10)),
                "p25": float(np.percentile(probabilities, 25)),
                "p50": float(np.percentile(probabilities, 50)),
                "p75": float(np.percentile(probabilities, 75)),
                "p90": float(np.percentile(probabilities, 90))
            }
        }


class PerformanceMonitor:
    """
    Monitor model performance over time.
    
    Tracks:
    - Latency statistics
    - Error rates
    - Throughput
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latency_buffer: Dict[str, deque] = {}
        self._error_buffer: Dict[str, deque] = {}
    
    def log_latency(self, disease_type: str, latency_ms: float):
        """Log prediction latency."""
        if disease_type not in self._latency_buffer:
            self._latency_buffer[disease_type] = deque(maxlen=self.window_size)
        self._latency_buffer[disease_type].append(latency_ms)
    
    def log_error(self, disease_type: str, error_type: str):
        """Log prediction error."""
        if disease_type not in self._error_buffer:
            self._error_buffer[disease_type] = deque(maxlen=self.window_size)
        self._error_buffer[disease_type].append({
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type
        })
    
    def get_latency_stats(self, disease_type: str) -> Dict[str, float]:
        """Get latency statistics."""
        if disease_type not in self._latency_buffer:
            return {}
        
        latencies = list(self._latency_buffer[disease_type])
        if not latencies:
            return {}
        
        return {
            "count": len(latencies),
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "max_ms": float(np.max(latencies))
        }
    
    def get_error_rate(self, disease_type: str, window_hours: int = 24) -> float:
        """Get error rate in the last N hours."""
        if disease_type not in self._error_buffer:
            return 0.0
        
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent_errors = [
            e for e in self._error_buffer[disease_type]
            if datetime.fromisoformat(e["timestamp"]) > cutoff
        ]
        
        # Approximate total requests
        total_requests = len(self._latency_buffer.get(disease_type, []))
        if total_requests == 0:
            return 0.0
        
        return len(recent_errors) / total_requests
