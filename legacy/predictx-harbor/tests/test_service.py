"""End-to-end tests of the HTTP service. Skipped until ``python -m service.train_models`` has been run."""
import json
import os

import pytest

from predictx import cohorts
from tests.test_website import EXAMPLES, HEART_FORM

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "artifacts")
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(ARTIFACTS, f"{e}.joblib")) for e in cohorts.ENDPOINTS),
    reason="trained artifacts not found")


HEART_TEST_FIELDS = ("max_heart_rate", "exercise_angina", "st_depression", "st_slope", "major_vessels", "thallium")


@pytest.fixture(autouse=True)
def records(tmp_path, monkeypatch):
    path = tmp_path / "inference.jsonl"
    monkeypatch.setenv("PREDICTX_RECORDS", str(path))
    return path


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from service.app import app
    return TestClient(app)


@pytest.mark.parametrize("endpoint", cohorts.ENDPOINTS)
def test_predict_returns_a_probability(client, endpoint):
    r = client.post(f"/predict/{endpoint}", json=EXAMPLES[endpoint])
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["risk_score"] <= 1 and body["risk_percent"] == pytest.approx(100 * body["risk_score"], abs=0.06)
    assert body["attribution"] == "Built with PriorLabs-TabPFN"


def test_onehot_route_matches_named_route(client):
    a = client.post("/predict/heart/onehot", json=HEART_FORM).json()
    b = client.post("/predict/heart", json=EXAMPLES["heart"]).json()
    assert a["risk_score"] == b["risk_score"]


@pytest.mark.parametrize("endpoint, field, low, high", [
    ("heart", "major_vessels", 0, 3),
    ("diabetes", "glucose", 85, 190),
    ("breast", "bare_nuclei", 1, 10),
    ("kidney", "albumin_grade", 0, 4),
    ("liver", "total_bilirubin", 0.6, 8.0),
])
def test_risk_increases_with_a_dominant_marker(client, endpoint, field, low, high):
    lo = client.post(f"/predict/{endpoint}", json={**EXAMPLES[endpoint], field: low}).json()["risk_score"]
    hi = client.post(f"/predict/{endpoint}", json={**EXAMPLES[endpoint], field: high}).json()["risk_score"]
    assert hi > lo


def test_invalid_payload_returns_422(client):
    assert client.post("/predict/heart", json={**EXAMPLES["heart"], "sex": "x"}).status_code == 422


def test_schema_lists_every_field(client):
    assert set(client.get("/schema/kidney").json()) == set(EXAMPLES["kidney"])


def test_every_prediction_is_recorded_with_its_model_version(client, records):
    body = client.post("/predict/diabetes", json=EXAMPLES["diabetes"]).json()
    entry = json.loads(records.read_text().splitlines()[-1])
    assert entry["endpoint"] == "diabetes" and entry["request"] == EXAMPLES["diabetes"]
    assert entry["model_version"] == body["model_version"] and len(entry["model_version"]) == 12
    assert entry["risk_score"] == pytest.approx(body["risk_score"], abs=5e-5)


def test_acquisition_ranks_every_test_not_yet_performed(client):
    baseline = {k: v for k, v in EXAMPLES["heart"].items() if k not in HEART_TEST_FIELDS}
    body = client.post("/acquire/heart", json=baseline).json()
    assert body["performed_tests"] == []
    values = [t["net_value"] for t in body["test_values"]]
    assert {t["test"] for t in body["test_values"]} == {"exercise_ecg", "fluoroscopy", "thallium"}
    assert values == sorted(values, reverse=True)
    assert body["recommended_test"] == (body["test_values"][0]["test"] if values[0] > 0 else None)


def test_acquisition_stops_when_testing_is_expensive_or_complete(client):
    baseline = {k: v for k, v in EXAMPLES["heart"].items() if k not in HEART_TEST_FIELDS}
    assert client.post("/acquire/heart?lam=1", json=baseline).json()["recommended_test"] is None
    complete = client.post("/acquire/heart", json=EXAMPLES["heart"]).json()
    assert complete["recommended_test"] is None and complete["test_values"] == []
