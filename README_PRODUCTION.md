# Healthcare Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

**Production-grade healthcare ML system for disease risk prediction**

[Quick Start](#quick-start) • [Architecture](#architecture) • [API Reference](#api-reference) • [Development](#development)

</div>

---

## Overview

A production-ready machine learning system for predicting healthcare risks across three conditions:
- **Diabetes** - Early detection for lifestyle intervention
- **Heart Disease** - Cardiovascular risk assessment
- **Stroke** - Critical risk identification for prevention

### Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Recall-Optimized** | Clinical thresholds tuned for high sensitivity (low false negatives) |
| 🔬 **Calibrated Probabilities** | Isotonic regression calibration for reliable risk estimates |
| 📊 **SHAP Explanations** | Feature importance for clinical interpretability |
| 🏥 **Healthcare Intelligence** | Clinical urgency levels and risk categorization |
| 📈 **Drift Detection** | Statistical monitoring for model degradation |
| 🔐 **HIPAA-Aware Logging** | Structured audit trails without PHI exposure |

---

## Quick Start

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd healthcare_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-production.txt

# Copy environment config
cp .env.example .env
```

### Run the API

```bash
# Development mode
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predict/diabetes \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "bmi": 28.5,
    "glucose": 150,
    "blood_pressure": 130,
    "insulin": 85,
    "family_history": 1
  }'
```

---

## Architecture

```
healthcare_prediction/
├── src/
│   ├── api/                 # FastAPI application
│   │   ├── main.py         # API endpoints
│   │   └── schemas.py      # Pydantic models
│   ├── core/               # Configuration & logging
│   │   ├── config.py       # Settings & thresholds
│   │   └── logging_config.py
│   ├── ml/                 # ML pipeline
│   │   ├── pipeline.py     # sklearn Pipeline + SMOTE
│   │   ├── calibration.py  # Probability calibration
│   │   └── training.py     # Model training
│   ├── healthcare/         # Clinical intelligence
│   │   └── intelligence.py # Risk categorization
│   ├── services/           # Business logic
│   │   └── inference_service.py
│   └── mlops/              # MLOps components
│       ├── registry.py     # Model versioning
│       └── monitoring.py   # Drift detection
├── tests/                  # Test suite
├── models/                 # Trained models
├── data/                   # Data files
├── Dockerfile
├── docker-compose.yml
└── requirements-production.txt
```

### System Flow

```
Request → API Layer → Validation → Inference Service → ML Pipeline
                                          ↓
                                    Calibration
                                          ↓
                              Healthcare Intelligence
                                          ↓
                                     Response
```

---

## API Reference

### Health Check
```
GET /health
```
Returns system health status and loaded models.

### Predictions

#### Diabetes Prediction
```
POST /predict/diabetes
```
**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.72,
  "confidence_interval": [0.65, 0.79],
  "risk_category": "high",
  "risk_factors": [
    {"feature": "glucose", "impact": "increases", "importance": 0.35}
  ],
  "recommendations": [
    "Schedule HbA1c test with healthcare provider"
  ],
  "disclaimer": "This prediction is for informational purposes only..."
}
```

#### Heart Disease Prediction
```
POST /predict/heart_disease
```

#### Stroke Prediction
```
POST /predict/stroke
```

### Model Information
```
GET /models
```
Returns information about loaded models and their metadata.

---

## ML Engineering Details

### Threshold Selection

Healthcare ML requires **recall-optimized thresholds** because missing disease (false negatives) is more costly than extra testing (false positives).

| Disease | Threshold | FN Cost Multiplier | Rationale |
|---------|-----------|-------------------|-----------|
| Diabetes | 0.35 | 3x | Early intervention prevents complications |
| Heart Disease | 0.40 | 5x | Missed CAD can lead to MI/death |
| Stroke | 0.30 | 8x | Catastrophic outcomes, time-sensitive |

### Pipeline Architecture

```python
Pipeline([
    ('validator', ClinicalRangeValidator()),    # Validate inputs
    ('engineer', FeatureEngineer()),            # Derive features
    ('preprocessor', ColumnTransformer([        # Transform
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])),
    ('smote', SMOTE()),                         # Handle imbalance
    ('classifier', CalibratedClassifier())      # Calibrated model
])
```

### Probability Calibration

Raw model outputs are calibrated using **isotonic regression** to ensure predicted probabilities match observed frequencies:

- Calibration error < 0.05 (Brier score component)
- Confidence intervals via bootstrap resampling

---

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test class
pytest tests/test_healthcare_system.py::TestCalibration -v
```

### Docker Deployment

```bash
# Build image
docker build -t healthcare-ml:latest .

# Run container
docker run -p 8000:8000 healthcare-ml:latest

# Or use docker-compose
docker-compose up -d
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

---

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | Environment name |
| `API_PORT` | 8000 | API server port |
| `LOG_LEVEL` | INFO | Logging verbosity |
| `DIABETES_THRESHOLD` | 0.35 | Classification threshold |
| `ENABLE_SHAP_EXPLANATIONS` | true | Generate SHAP values |

---

## Monitoring

### Drift Detection

The system monitors for:
- **Feature drift**: Kolmogorov-Smirnov test on input distributions
- **Prediction drift**: Mann-Whitney U test on output distributions
- **Performance drift**: Rolling accuracy/recall metrics

### Metrics

Prometheus metrics available at `/metrics`:
- `prediction_latency_seconds` - Request latency histogram
- `predictions_total` - Prediction count by disease/result
- `drift_score` - Current drift detection score

---

## Healthcare Compliance

### Limitations

⚠️ This system is for **educational and screening purposes only**.

- NOT a diagnostic tool
- NOT a substitute for professional medical advice
- Should be used with clinical oversight

### Ethical Considerations

- Model performance may vary across demographic groups
- Regular bias auditing recommended
- Human oversight required for all clinical decisions

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<div align="center">
Built with ❤️ for healthcare ML education
</div>
