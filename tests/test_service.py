"""End-to-end tests of the HTTP service. Skipped until ``python -m service.train_models`` has been run."""
import os

import pytest

from predictx import cohorts
from tests.test_website import EXAMPLES, HEART_FORM

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "artifacts")
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(ARTIFACTS, f"{e}.joblib")) for e in cohorts.ENDPOINTS),
    reason="trained artifacts not found")


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
