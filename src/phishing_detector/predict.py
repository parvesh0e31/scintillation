from __future__ import annotations
from pathlib import Path
import joblib
from .features import feature_matrix, validate_url


def load_model(path: str):
    return joblib.load(Path(path))


def predict_url(model, url: str) -> dict:
    value = validate_url(url)
    probability = float(model.predict_proba(feature_matrix([value]))[0, 1])
    return {"url": value, "label": "phishing" if probability >= 0.5 else "legitimate", "phishing_probability": probability}
