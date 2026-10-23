"""Service tests with a model fitted on fixture data and a tiny randomly initialised encoder."""
import json
import shutil

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sweepscreen.data import build as build_mod
from sweepscreen.features import FeatureStore, extract
from sweepscreen.models.encoder import Encoder
from sweepscreen.pipeline import AggregationConfig, SweepScreen
from sweepscreen.train import TrainConfig, fit
from tests.conftest import synthetic_frame


@pytest.fixture
def client(tmp_path, busi_tree, busbra_tree, breastvid_tree):
    raw = tmp_path / "raw"
    raw.mkdir()
    for tree, name in ((busi_tree, "busi"), (busbra_tree, "busbra"), (breastvid_tree, "breastvid")):
        shutil.move(str(tree), str(raw / name))
    table = build_mod.build(raw=raw, extracted=tmp_path / "ext", processed=tmp_path / "proc")
    enc = Encoder(config=Encoder.tiny_config(), threads=1)
    store = FeatureStore(extract(table, enc, out_root=tmp_path / "feat", chunk=16, qfail_every=2))
    models = fit(store, table, np.arange(len(table)), TrainConfig(detector_epochs=2, video_stride=1, malignancy_video_stride=1))
    model = SweepScreen(models, enc, AggregationConfig(window=2, min_frames=3, min_fraction=0.5))
    model.save(tmp_path / "m" / "bundle.joblib")
    from service.app import create_app
    app = create_app(model=model, log_path=tmp_path / "log.jsonl")
    c = TestClient(app)
    c.log_path = tmp_path / "log.jsonl"
    return c


def _png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def _video(tmp_path, frames):
    path = tmp_path / "sweep.avi"
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()
    return path.read_bytes()


def test_health_and_ui(client):
    assert client.get("/health").json()["model_loaded"]
    assert "SweepScreen" in client.get("/").text
    card = client.get("/model").json()
    assert len(card["version"]) == 12


def test_analyze_image(client):
    img, _ = synthetic_frame(np.random.default_rng(0))
    r = client.post("/analyze/image", files={"file": ("f.png", _png(img), "image/png")})
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["lesion_probability"] <= 1 and j["heatmap_png"] and j["annotated_png"]


def test_analyze_sweep_video_and_log(client, tmp_path):
    rng = np.random.default_rng(1)
    frames = [synthetic_frame(rng, center=(120, 100))[0] for _ in range(8)]
    vid = _video(tmp_path, frames)
    r = client.post("/analyze/sweep", data={"woman_id": "W-001"},
                    files=[("left", ("l.avi", vid, "video/x-msvideo")), ("right", ("r.avi", vid, "video/x-msvideo"))])
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["decision"] in {"NORMAL", "RECALL", "UNREADABLE"}
    assert set(j["breasts"]) == {"left", "right"}
    entry = json.loads(client.log_path.read_text().splitlines()[-1])
    assert entry["endpoint"] == "/analyze/sweep" and entry["decision"] == j["decision"]
    assert entry["woman_id_sha256"] and "W-001" not in json.dumps(entry)
    assert entry["frames"] == {"left": 8, "right": 8}


def test_bad_upload_rejected(client):
    r = client.post("/analyze/image", files={"file": ("f.png", b"not an image", "image/png")})
    assert r.status_code == 400
