"""End-to-end inference: raw frames → per-frame analysis → per-woman decision.

A model bundle (``models/sweepscreen-<version>.joblib``) holds the fitted quality gate, detector head,
frame calibrator, malignancy head, operating threshold and aggregation settings. The frozen encoder is
loaded from its local weight folder. :class:`SweepScreen` exposes

* :meth:`analyze_frames`: quality, lesion probability, boxes (original pixel coordinates), heatmap and
  malignancy suspicion per frame;
* :meth:`analyze_sweep`: per-breast persistence aggregation and the per-woman NORMAL / RECALL /
  UNREADABLE decision.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np

from .aggregate import aggregate_breast, decide_woman
from .features import prepare_record
from .models.detector import frame_stats
from .models.malignancy import lesion_features
from .quality import frame_features
from .train import Models

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AggregationConfig:
    window: int = 5
    min_frames: int = 30
    min_fraction: float = 0.6


@dataclass
class FrameResult:
    index: int
    adequate: bool
    quality: float
    lesion_probability: float
    boxes: List[List[int]]              # original image coordinates
    box_scores: List[float]
    suspicion: Optional[float]          # malignancy score of the top box, if any
    heatmap: Optional[np.ndarray] = field(default=None, repr=False)   # prepared 224x224 map


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SweepScreen:
    def __init__(self, models: Models, encoder, aggregation: AggregationConfig = AggregationConfig(),
                 version: str = "dev"):
        self.models, self.encoder, self.aggregation, self.version = models, encoder, aggregation, version

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> str:
        """Write the bundle and a JSON card; returns the version (first 12 hex digits of the SHA-256)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "aggregation": asdict(self.aggregation), "encoder": self.encoder.name},
                    path, compress=3)
        version = file_sha256(path)[:12]
        card = {"version": version, "file": path.name, "encoder": self.encoder.name,
                "frame_threshold": self.models.frame_threshold, "aggregation": asdict(self.aggregation),
                "train_info": self.models.info}
        path.with_suffix(".json").write_text(json.dumps(card, indent=1, default=str))
        self.version = version
        return version

    @classmethod
    def load(cls, path: Path, encoder) -> "SweepScreen":
        blob = joblib.load(path)
        return cls(blob["models"], encoder, AggregationConfig(**blob["aggregation"]), file_sha256(Path(path))[:12])

    # ------------------------------------------------------------------ inference
    def analyze_frames(self, frames: Sequence[np.ndarray], heatmaps: bool = False) -> List[FrameResult]:
        prepared = [prepare_record(f, []) for f in frames]
        q = self.models.quality.score_features(np.stack([frame_features(p.cropped) for p in prepared]))
        cls, tokens = self.encoder(np.stack([p.image for p in prepared]))
        det = self.models.detector
        prob = det.patch_probabilities(tokens)
        fp = det.calibrator_.predict_proba(frame_stats(prob))[:, 1]
        results = []
        for i, p in enumerate(prepared):
            boxes, scores = det.boxes(prob[i])
            suspicion = None
            if boxes and self.models.malignancy is not None:
                X = lesion_features(cls[i:i + 1], tokens[i:i + 1], [boxes[0]], [None])
                suspicion = float(self.models.malignancy.predict(X)[0])
            orig = []
            x0c, y0c = p.crop[0], p.crop[1]
            for (bx0, by0, bx1, by1) in boxes:
                px, py = p.pad
                orig.append([int((bx0 - px) / p.scale) + x0c, int((by0 - py) / p.scale) + y0c,
                             int((bx1 - px) / p.scale) + x0c, int((by1 - py) / p.scale) + y0c])
            results.append(FrameResult(i, bool(q[i] >= self.models.quality.threshold_), float(q[i]), float(fp[i]),
                                       orig, scores, suspicion, det.heatmap(prob[i]) if heatmaps else None))
        return results

    def analyze_sweep(self, sweeps: Dict[str, Sequence[np.ndarray]]) -> Dict[str, object]:
        """``sweeps`` maps breast side ("left"/"right") to its frame sequence."""
        a = self.aggregation
        per_side, frames_out = {}, {}
        for side, frames in sweeps.items():
            res = self.analyze_frames(frames)
            per_side[side] = aggregate_breast([r.lesion_probability for r in res], [r.adequate for r in res],
                                              a.window, a.min_frames, a.min_fraction)
            frames_out[side] = res
        woman = decide_woman(per_side, self.models.frame_threshold)
        return {"decision": woman.decision, "score": woman.score, "reasons": woman.reasons,
                "breasts": {s: asdict(b) for s, b in per_side.items()}, "frames": frames_out,
                "threshold": self.models.frame_threshold, "model_version": self.version}
