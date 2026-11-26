"""Online serving: a small REST API around the exported model."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

N_FEATURES = 13


class Wine(BaseModel):
    features: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)


def create_app(model_dir=None):
    model_dir = Path(model_dir or os.getenv("MODEL_DIR", "model"))
    model = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    app = FastAPI(title="Wine classifier")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info():
        return {**meta, "git_sha": os.getenv("GIT_SHA", "local")}

    @app.post("/predict")
    def predict(wine: Wine):
        row = np.array([wine.features])
        probabilities = model.predict_proba(row)[0]
        index = int(np.argmax(probabilities))
        return {
            "prediction": index,
            "class_name": meta["target_names"][index],
            "probabilities": [round(float(p), 4) for p in probabilities],
        }

    return app
