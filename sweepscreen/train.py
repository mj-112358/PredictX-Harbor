"""Fitting all learned components on a set of training rows of a feature store.

Used identically by every experiment fold and by the final deployable model, so evaluation and
deployment share one code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .evaluation.metrics import threshold_for_spec
from .features import FeatureStore
from .models.detector import PatchDetector, frame_stats
from .models.malignancy import MalignancyHead, lesion_features
from .quality import QualityGate


@dataclass
class TrainConfig:
    video_stride: int = 3          # use every k-th video frame for detector training
    malignancy_video_stride: int = 10
    frame_spec: float = 0.95       # frame threshold: this fraction of lesion-free validation frames below it
    detector_epochs: int = 60
    quality_target_pass: float = 0.97
    seed: int = 0


@dataclass
class Models:
    detector: PatchDetector
    malignancy: Optional[MalignancyHead]
    quality: QualityGate
    frame_threshold: float
    config: TrainConfig
    info: Dict[str, object] = field(default_factory=dict)


def _subsample(table: pd.DataFrame, rows: np.ndarray, stride: int) -> np.ndarray:
    keep = []
    for i in rows:
        fi = table.frame_index.iloc[i]
        if pd.isna(table.video_id.iloc[i]) or int(fi) % stride == 0:
            keep.append(i)
    return np.asarray(keep, int)


def lesion_label(table: pd.DataFrame) -> np.ndarray:
    return table.lesion_present.astype(bool).values.astype(int)


def fit(store: FeatureStore, table: pd.DataFrame, train_rows: np.ndarray, cfg: TrainConfig = TrainConfig()) -> Models:
    train_rows = np.asarray([i for i in train_rows if not bool(table.discordant.iloc[i])], int)
    det_rows = _subsample(table, train_rows, cfg.video_stride)
    tokens = store.array("tokens")
    targets = store.array("targets")
    y_frame = lesion_label(table)

    X = np.asarray(tokens[det_rows], np.float32)
    det = PatchDetector(dim=X.shape[-1], grid=X.shape[1], epochs=cfg.detector_epochs, seed=cfg.seed)
    det.fit(X, np.asarray(targets[det_rows], np.float32), y_frame[det_rows], table.group_id.values[det_rows])
    p_train = det.frame_probability(X)
    neg = y_frame[det_rows] == 0
    frame_threshold = threshold_for_spec(p_train[neg], cfg.frame_spec) if neg.any() else 0.5

    mal_rows = _subsample(table, train_rows, cfg.malignancy_video_stride)
    mal_rows = mal_rows[(table.label.values[mal_rows] != "normal") & (y_frame[mal_rows] == 1)]
    mal = None
    if len(np.unique(table.label.values[mal_rows])) == 2:
        boxes = [max(b, key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1])) if b else None for b in store.boxes(mal_rows)]
        Xm = lesion_features(np.asarray(store.array("cls")[mal_rows]), np.asarray(tokens[mal_rows], np.float32), boxes,
                             table.age.values[mal_rows])
        mal = MalignancyHead(cfg.seed).fit(Xm, (table.label.values[mal_rows] == "malignant").astype(int),
                                           table.group_id.values[mal_rows])

    qrows = np.load(store.root / "qfail_rows.npy")
    qfail = np.load(store.root / "qfail.npy")
    in_train = np.isin(qrows, train_rows)
    qfeat = np.asarray(store.array("qfeat"))
    gate = QualityGate(target_pass=cfg.quality_target_pass, seed=cfg.seed).fit_features(
        qfeat[_subsample(table, train_rows, cfg.video_stride)], qfail[in_train].reshape(-1, qfail.shape[-1]))
    return Models(det, mal, gate, float(frame_threshold), cfg,
                  {"n_detector_rows": int(len(det_rows)), "n_malignancy_rows": int(len(mal_rows))})


def predict(models: Models, store: FeatureStore, table: pd.DataFrame, rows: np.ndarray, batch: int = 512) -> pd.DataFrame:
    """Per-row frame probability, candidate boxes, malignancy (GT box and detected box) and quality score."""
    tokens, cls = store.array("tokens"), store.array("cls")
    out = []
    for s in range(0, len(rows), batch):
        r = rows[s:s + batch]
        X = np.asarray(tokens[r], np.float32)
        prob = models.detector.patch_probabilities(X)
        fp = models.detector.calibrator_.predict_proba(frame_stats(prob))[:, 1]
        cands = [models.detector.boxes(prob[k], threshold=0.3, max_boxes=10) for k in range(len(r))]
        gt = store.boxes(r)
        mal_gt, mal_det = np.full(len(r), np.nan), np.full(len(r), np.nan)
        if models.malignancy is not None:
            gtb = [max(b, key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1])) if b else None for b in gt]
            detb = [c[0][0] if c[0] else None for c in cands]
            ages = table.age.values[r]
            mal_gt = models.malignancy.predict(lesion_features(np.asarray(cls[r]), X, gtb, ages))
            mal_det = models.malignancy.predict(lesion_features(np.asarray(cls[r]), X, detb, ages))
        q = models.quality.score_features(np.asarray(store.array("qfeat")[r]))
        for k, i in enumerate(r):
            out.append({"row": int(i), "uid": table.uid.iloc[i], "site": table.site.iloc[i],
                        "group_id": table.group_id.iloc[i], "label": table.label.iloc[i],
                        "lesion": int(table.lesion_present.iloc[i]), "frame_prob": float(fp[k]),
                        "mal_gt": float(mal_gt[k]), "mal_det": float(mal_det[k]), "quality": float(q[k]),
                        "video_id": table.video_id.iloc[i], "frame_index": table.frame_index.iloc[i],
                        "candidates": [[list(b), sc] for b, sc in zip(*cands[k])], "gt_boxes": [list(b) for b in gt[k]]})
    return pd.DataFrame(out)
