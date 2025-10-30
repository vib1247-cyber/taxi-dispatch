from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from taxi_dispatch.io_utils import load_joblib
from taxi_dispatch.demand_model import DemandModelWrapper, build_api_feature_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

app = FastAPI(title="Smart Taxi Dispatch API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_wrapper() -> DemandModelWrapper:
    path = OUTPUTS_DIR / "demand_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return load_joblib(path)


def _load_centers() -> np.ndarray:
    path = OUTPUTS_DIR / "centers.npy"
    if not path.exists():
        raise FileNotFoundError(f"Centers not found: {path}")
    return np.load(path)


def _load_distance_matrix() -> np.ndarray:
    path = OUTPUTS_DIR / "distance_matrix.npy"
    if not path.exists():
        raise FileNotFoundError(f"Distance matrix not found: {path}")
    return np.load(path)


class PredictRequest(BaseModel):
    zone: int
    timestamp: str  # ISO8601
    extra: Optional[Dict[str, float]] = None


class DistanceRequest(BaseModel):
    from_zone: int
    to_zone: int


@app.get("/")
def root():
    return {"message": "Smart Taxi API is live 🚖"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/zones")
async def zones():
    centers = _load_centers()
    zones = list(range(len(centers)))
    return {"zones": zones, "centers": centers.tolist()}


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        wrapper: DemandModelWrapper = _load_wrapper()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        ts = pd.to_datetime(req.timestamp)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
    X = build_api_feature_row(zone=int(req.zone), timestamp_hour=ts, extra_numeric=req.extra, wrapper=wrapper)
    yhat = float(wrapper.predict_vector(X)[0])
    return {"zone": req.zone, "timestamp": ts.isoformat(), "predicted_demand": yhat, "model_type": wrapper.model_type}


@app.post("/distance")
async def distance(req: DistanceRequest):
    try:
        D = _load_distance_matrix()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    Z = D.shape[0]
    if not (0 <= req.from_zone < Z and 0 <= req.to_zone < Z):
        raise HTTPException(status_code=400, detail=f"Zones must be in [0, {Z-1}]")
    return {"from_zone": req.from_zone, "to_zone": req.to_zone, "distance_km": float(D[req.from_zone, req.to_zone])}
