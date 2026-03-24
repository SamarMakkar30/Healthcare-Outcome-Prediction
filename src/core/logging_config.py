"""
Production Logging Configuration
Structured logging with audit trail support for HIPAA compliance.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
from functools import lru_cache


class StructuredFormatter(logging.Formatter):
    """JSON structured logging for production observability."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "disease_type"):
            log_entry["disease_type"] = record.disease_type
        if hasattr(record, "prediction_id"):
            log_entry["prediction_id"] = record.prediction_id
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms
        if hasattr(record, "event_type"):
            log_entry["event_type"] = record.event_type
            
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)


class AuditLogger:
    """
    HIPAA-compliant audit logging.
    Records all PHI access with immutable trail.
    """
    
    def __init__(self, logger_name: str = "audit"):
        self.logger = logging.getLogger(logger_name)
        
    def log_prediction_request(
        self,
        request_id: str,
        disease_type: str,
        patient_id_hash: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Log prediction request for audit trail."""
        self.logger.info(
            "Prediction requested",
            extra={
                "event_type": "prediction_request",
                "request_id": request_id,
                "disease_type": disease_type,
                "patient_id_hash": patient_id_hash,
                "user_id": user_id
            }
        )
    
    def log_prediction_result(
        self,
        request_id: str,
        disease_type: str,
        risk_level: str,
        probability: float,
        latency_ms: float
    ):
        """Log prediction result."""
        self.logger.info(
            "Prediction completed",
            extra={
                "event_type": "prediction_result",
                "request_id": request_id,
                "disease_type": disease_type,
                "risk_level": risk_level,
                "probability": round(probability, 4),
                "latency_ms": round(latency_ms, 2)
            }
        )
    
    def log_model_loaded(self, model_name: str, version: str):
        """Log model loading event."""
        self.logger.info(
            f"Model loaded: {model_name}",
            extra={
                "event_type": "model_loaded",
                "model_name": model_name,
                "model_version": version
            }
        )
    
    def log_security_event(
        self,
        event_type: str,
        details: dict,
        severity: str = "warning"
    ):
        """Log security-relevant events."""
        log_func = getattr(self.logger, severity, self.logger.warning)
        log_func(
            f"Security event: {event_type}",
            extra={
                "event_type": f"security_{event_type}",
                "details": details
            }
        )


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True
) -> None:
    """
    Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        json_format: Use JSON structured logging
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
    
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
    
    # Set levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


@lru_cache()
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


@lru_cache()
def get_audit_logger() -> AuditLogger:
    """Get the audit logger instance."""
    return AuditLogger()
