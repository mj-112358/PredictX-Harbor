"""HTTP prediction service for the five PredictX endpoints.

Run:   uvicorn service.app:app --host 0.0.0.0 --port 8000

    GET  /health                      service status and loaded endpoints
    GET  /schema/{endpoint}           form fields, units, admissible ranges and options
    POST /predict/{endpoint}          body: form fields of the endpoint (see /schema/{endpoint})
                                      returns {"endpoint", "risk_score", "risk_percent", "attribution"}
    POST /predict/heart/onehot        body: the website's one-hot coronary form

``risk_score`` is the PredictX ensemble probability of the endpoint's disease label. Every response carries
the attribution required by the TabPFN licence.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predictx import cohorts
from predictx.website import SCHEMAS, heart_from_onehot, to_model_frame

ARTIFACTS = os.environ.get("PREDICTX_ARTIFACTS", os.path.join(os.path.dirname(__file__), "..", "artifacts"))


def _artifact(endpoint: str) -> str:
    return os.path.join(ARTIFACTS, f"{endpoint}.joblib")


@lru_cache(maxsize=None)
def model(endpoint: str):
    """Load an endpoint's ensemble once and prepare it for low-latency inference."""
    if not os.path.exists(_artifact(endpoint)):
        raise HTTPException(503, f"model for {endpoint} has not been trained (run python -m service.train_models)")
    return joblib.load(_artifact(endpoint)).prepare_for_serving()


@asynccontextmanager
async def lifespan(_: FastAPI):
    for endpoint in cohorts.ENDPOINTS:
        if os.path.exists(_artifact(endpoint)):
            model(endpoint)
    yield


ATTRIBUTION = "Built with PriorLabs-TabPFN"

app = FastAPI(title="PredictX", version="1.0", description=ATTRIBUTION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def risk(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if endpoint not in cohorts.ENDPOINTS:
        raise HTTPException(404, f"unknown endpoint {endpoint!r}")
    try:
        row = to_model_frame(endpoint, payload)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err
    score = float(model(endpoint).predict_proba(row)[0, 1])
    return {"endpoint": endpoint, "risk_score": round(score, 4), "risk_percent": round(100 * score, 1),
            "attribution": ATTRIBUTION}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "endpoints": [e for e in cohorts.ENDPOINTS if os.path.exists(_artifact(e))],
            "attribution": ATTRIBUTION}


@app.get("/schema/{endpoint}")
def schema(endpoint: str) -> Dict[str, Any]:
    if endpoint not in SCHEMAS:
        raise HTTPException(404, f"unknown endpoint {endpoint!r}")
    return {name: {"label": f.label, "unit": f.unit, "range": f.range,
                   "options": None if f.options is None else list(f.options)}
            for name, f in SCHEMAS[endpoint].items()}


@app.post("/predict/heart/onehot")
def predict_heart_onehot(form: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = heart_from_onehot(form)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err
    return risk("heart", payload)


@app.post("/predict/{endpoint}")
def predict(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return risk(endpoint, payload)
