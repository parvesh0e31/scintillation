from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .predict import load_model, predict_url
from pathlib import Path
import os

app = FastAPI(title="URL Phishing Detector", version="1.0")
MODEL_PATH = os.getenv("PHISHING_MODEL", "models/model.joblib")
_model = None

class URLRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict")
def predict(request: URLRequest):
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(status_code=503, detail="model artifact not found; train a model first")
        _model = load_model(MODEL_PATH)
    try:
        return predict_url(_model, request.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
