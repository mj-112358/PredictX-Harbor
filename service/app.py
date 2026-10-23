"""SweepScreen HTTP service.

    uvicorn service.app:app --port 8000

Routes
  GET  /                 health-worker web page
  GET  /health           status, loaded model version
  GET  /model            model card of the loaded version
  POST /analyze/image    one image → lesion probability, boxes, suspicion, quality, annotated image, heatmap
  POST /analyze/sweep    left and right sweep (video or image files) → NORMAL / RECALL / UNREADABLE

Environment: ``SWEEPSCREEN_MODELS`` (folder with ``registry.json``, default ``models/``),
``SWEEPSCREEN_ENCODER`` (DINOv2 weight folder), ``SWEEPSCREEN_LOG`` (JSONL inference log, default
``logs/inference.jsonl``).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from sweepscreen.features import prepare_record
from sweepscreen.pipeline import SweepScreen

from .media import decode_upload, draw_boxes, heatmap_overlay, png_b64

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / "static"


class InferenceLog:
    """Append-only JSON-lines record of every request (thread-safe)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, entry: dict) -> None:
        with self._lock, open(self.path, "a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")


def load_current(models_dir: Path, encoder_dir: Optional[str] = None) -> SweepScreen:
    from sweepscreen.models.encoder import Encoder

    reg = json.loads((models_dir / "registry.json").read_text())
    entry = [e for e in reg if e.get("current")][-1]
    return SweepScreen.load(models_dir / entry["file"], Encoder(weights=encoder_dir))


def create_app(model: Optional[SweepScreen] = None, log_path: Optional[Path] = None,
               models_dir: Optional[Path] = None) -> FastAPI:
    models_dir = Path(models_dir or os.environ.get("SWEEPSCREEN_MODELS", ROOT / "models"))
    state = {"model": model}
    log = InferenceLog(Path(log_path or os.environ.get("SWEEPSCREEN_LOG", ROOT / "logs" / "inference.jsonl")))
    app = FastAPI(title="SweepScreen", version="0.1.0")

    def get_model() -> SweepScreen:
        if state["model"] is None:
            try:
                state["model"] = load_current(models_dir, os.environ.get("SWEEPSCREEN_ENCODER"))
            except FileNotFoundError as exc:
                raise HTTPException(503, f"model not available: {exc}")
        return state["model"]

    def record(endpoint, digests, result, started, extra=None):
        log.write({"id": str(uuid.uuid4()), "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                   "endpoint": endpoint, "model_version": get_model().version, "inputs_sha256": digests,
                   "latency_ms": round((time.perf_counter() - started) * 1000, 1), **result, **(extra or {})})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        m = state["model"]
        return {"status": "ok", "model_loaded": m is not None, "model_version": m.version if m else None}

    @app.get("/model")
    def model_card():
        m = get_model()
        return {"version": m.version, "encoder": m.encoder.name, "frame_threshold": m.models.frame_threshold,
                "quality_threshold": m.models.quality.threshold_, "aggregation": m.aggregation.__dict__}

    @app.post("/analyze/image")
    async def analyze_image(file: UploadFile = File(...)):
        started = time.perf_counter()
        data = await file.read()
        try:
            frames = decode_upload(file.filename, data)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        m = get_model()
        r = m.analyze_frames(frames[:1], heatmaps=True)[0]
        prepared = prepare_record(frames[0], []).image
        result = {"lesion_probability": r.lesion_probability, "suspicious": r.lesion_probability >= m.models.frame_threshold,
                  "adequate": r.adequate, "quality": r.quality, "boxes": r.boxes, "box_scores": r.box_scores,
                  "suspicion": r.suspicion}
        record("/analyze/image", [hashlib.sha256(data).hexdigest()], result, started)
        return {**result, "model_version": m.version, "threshold": m.models.frame_threshold,
                "annotated_png": png_b64(draw_boxes(frames[0], r.boxes, r.box_scores)),
                "heatmap_png": png_b64(heatmap_overlay(prepared, r.heatmap))}

    @app.post("/analyze/sweep")
    async def analyze_sweep(left: List[UploadFile] = File(...), right: List[UploadFile] = File(...),
                            woman_id: str = Form("")):
        started = time.perf_counter()
        sweeps, digests = {}, []
        for side, files in (("left", left), ("right", right)):
            frames = []
            for f in files:
                data = await f.read()
                digests.append(hashlib.sha256(data).hexdigest())
                try:
                    frames += decode_upload(f.filename, data)
                except ValueError as exc:
                    raise HTTPException(400, f"{side}: {exc}")
            sweeps[side] = frames
        m = get_model()
        out = m.analyze_sweep(sweeps)
        key = {}
        for side, b in out["breasts"].items():
            k = b["peak_frame"]
            if k is not None:
                fr = out["frames"][side][k]
                key[side] = {"frame": k, "score": b["score"], "boxes": fr.boxes,
                             "annotated_png": png_b64(draw_boxes(sweeps[side][k], fr.boxes, fr.box_scores))}
        summary = {"decision": out["decision"], "score": out["score"], "reasons": out["reasons"],
                   "breasts": {s: {k: v for k, v in b.items() if k != "coverage"} | {"coverage": b["coverage"]}
                               for s, b in out["breasts"].items()}}
        woman_hash = hashlib.sha256(woman_id.encode()).hexdigest()[:16] if woman_id else None
        record("/analyze/sweep", digests, summary, started,
               {"woman_id_sha256": woman_hash, "frames": {s: len(v) for s, v in sweeps.items()}})
        return {**summary, "key_frames": key, "threshold": out["threshold"], "model_version": m.version}

    return app


app = create_app()
