"""
Bootstrap model artifacts compatible with current environment.

This script trains lightweight pipelines from datasets in data/raw and writes
artifacts expected by the production inference service.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.config import FEATURE_CONFIG


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"

DATASETS = {
    "diabetes": DATA_DIR / "diabetes_data.csv",
    "heart_disease": DATA_DIR / "heart_disease_data.csv",
    "stroke": DATA_DIR / "stroke_data.csv",
}


def train_one(disease: str, csv_path: Path) -> None:
    config = FEATURE_CONFIG[disease]
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    target = config["target"]

    df = pd.read_csv(csv_path)
    X = df[numeric_features + categorical_features]
    y = df[target]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_path = MODELS_DIR / f"{disease}_pipeline.pkl"
    threshold_path = MODELS_DIR / f"{disease}_threshold.json"

    joblib.dump(pipeline, pipeline_path)

    threshold_config = {
        "disease_type": disease,
        "model_name": "bootstrap_random_forest",
        "optimal_threshold": 0.5,
        "notes": "Bootstrap model generated for runtime compatibility",
    }
    threshold_path.write_text(json.dumps(threshold_config, indent=2), encoding="utf-8")

    print(f"Saved: {pipeline_path}")
    print(f"Saved: {threshold_path}")


def main() -> None:
    for disease, path in DATASETS.items():
        print(f"Training {disease} from {path}...")
        train_one(disease, path)

    print("Done.")


if __name__ == "__main__":
    main()
