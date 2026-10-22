"""HTTP prediction service for the five PredictX endpoints.

Run:   uvicorn service.app:app --host 0.0.0.0 --port 8000

    GET  /health                      service status and loaded endpoints
    GET  /schema/{endpoint}           form fields, units, admissible ranges and options
    POST /predict/{endpoint}          body: form fields of the endpoint (see /schema/{endpoint})
                                      returns {"endpoint", "risk_score", "risk_percent", "attribution"}
    POST /predict/heart/onehot        body: the website's one-hot coronary form
    POST /acquire/heart?lam=0         body: coronary form fields recorded so far
                                      returns the risk score and the net value of every test not yet performed

``risk_score`` is the PredictX ensemble probability of the endpoint's disease label. Every prediction is
appended to a JSON-lines inference record (``PREDICTX_RECORDS``, default ``records/inference.jsonl``) with
the request, the model version (SHA-256 prefix of the artifact) and the score. Every response carries the
attribution required by the TabPFN licence.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predictx import cohorts
from predictx.acquisition import HEART_TESTS, mask_tests
from predictx.website import SCHEMAS, heart_from_onehot, to_model_frame

ROOT = os.path.join(os.path.dirname(__file__), "..")
ARTIFACTS = os.environ.get("PREDICTX_ARTIFACTS", os.path.join(ROOT, "artifacts"))
ACQUISITION = "heart_acquisition"


def _artifact(name: str) -> str:
    return os.path.join(ARTIFACTS, f"{name}.joblib")


def _require(name: str) -> str:
    path = _artifact(name)
    if not os.path.exists(path):
        raise HTTPException(503, f"artifact {name} has not been trained (run python -m service.train_models)")
    return path


@lru_cache(maxsize=None)
def model_version(name: str) -> str:
    """First twelve hexadecimal digits of the SHA-256 digest of an artifact."""
    with open(_require(name), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


@lru_cache(maxsize=None)
def model(endpoint: str):
    """Load an endpoint's ensemble once and prepare it for low-latency inference."""
    return joblib.load(_require(endpoint)).prepare_for_serving()


@lru_cache(maxsize=None)
def acquisition_policy():
    """Load the coronary value-of-information controller."""
    return joblib.load(_require(ACQUISITION))


def record(entry: Dict[str, Any]) -> None:
    """Append one prediction to the JSON-lines inference record."""
    path = os.environ.get("PREDICTX_RECORDS", os.path.join(ROOT, "records", "inference.jsonl"))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **entry}) + "\n")


@asynccontextmanager
async def lifespan(_: FastAPI):
    for endpoint in cohorts.ENDPOINTS:
        if os.path.exists(_artifact(endpoint)):
            model(endpoint)
    if os.path.exists(_artifact(ACQUISITION)):
        acquisition_policy()
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
    record({"endpoint": endpoint, "model_version": model_version(endpoint), "request": payload,
            "risk_score": score})
    return {"endpoint": endpoint, "risk_score": round(score, 4), "risk_percent": round(100 * score, 1),
            "model_version": model_version(endpoint), "attribution": ATTRIBUTION}


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


@app.post("/acquire/heart")
def acquire_heart(payload: Dict[str, Any], lam: float = 0.0) -> Dict[str, Any]:
    """Rank the coronary tests not yet performed by net value of information (information utility).

    A test counts as performed when every biomarker it reveals is present in ``payload``. ``lam`` is the
    price of one unit of testing cost; the recommended test is the one with the largest positive net value.
    """
    if lam < 0:
        raise HTTPException(422, "lam must be non-negative")
    try:
        row = to_model_frame("heart", payload)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err
    performed = {t for t, (cols, _) in HEART_TESTS.items() if row[cols].notna().all(axis=None)}
    values = acquisition_policy().net_values(mask_tests(row, performed), performed, lam)
    ranked = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    recommended = ranked[0][0] if ranked and ranked[0][1] > 0 else None
    response = risk("heart", payload)
    response.update({"performed_tests": sorted(performed), "recommended_test": recommended,
                     "test_values": [{"test": t, "cost": HEART_TESTS[t][1], "net_value": round(v, 4)}
                                     for t, v in ranked]})
    return response
