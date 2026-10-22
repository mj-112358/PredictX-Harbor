"""Detection metrics: IoU matching, FROC and average precision.

* **Matching.** Per image, predictions are taken in decreasing score order; each matches the
  unmatched ground-truth box with the highest IoU if that IoU ≥ ``iou``. Every other prediction is a
  false positive.
* **FROC.** Lesion-level sensitivity against the mean number of false positives per image, over
  all score thresholds (IoU ≥ 0.3 hit criterion, the convention for lesion detection). Lesion-free
  images contribute false positives only.
* **AP@0.5.** Pascal-VOC all-point interpolated average precision at IoU ≥ 0.5.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds: Sequence[Tuple[Box, float]], gts: Sequence[Box], thr: float) -> List[Tuple[float, bool]]:
    """(score, is_true_positive) for each prediction of one image."""
    used = set()
    out = []
    for box, score in sorted(preds, key=lambda x: -x[1]):
        best, best_j = 0.0, None
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(box, g)
            if v > best:
                best, best_j = v, j
        if best_j is not None and best >= thr:
            used.add(best_j)
            out.append((score, True))
        else:
            out.append((score, False))
    return out


def _scored(all_preds, all_gts, thr):
    scores, tps = [], []
    for preds, gts in zip(all_preds, all_gts):
        for s, tp in match(preds, gts, thr):
            scores.append(s)
            tps.append(tp)
    order = np.argsort(-np.asarray(scores)) if scores else np.array([], int)
    return np.asarray(scores)[order], np.asarray(tps, bool)[order]


def froc(all_preds, all_gts, iou_thr: float = 0.3, fppi_points=(0.125, 0.25, 0.5, 1.0, 2.0)) -> Dict[str, object]:
    """FROC curve and sensitivity at the requested false positives per image."""
    n_img, n_les = len(all_gts), sum(len(g) for g in all_gts)
    scores, tps = _scored(all_preds, all_gts, iou_thr)
    tp_cum, fp_cum = np.cumsum(tps), np.cumsum(~tps)
    sens = tp_cum / max(n_les, 1)
    fppi = fp_cum / max(n_img, 1)
    at = {}
    for f in fppi_points:
        ok = np.flatnonzero(fppi <= f)
        at[f] = float(sens[ok[-1]]) if len(ok) else 0.0
    return {"fppi": fppi.tolist(), "sensitivity": sens.tolist(), "at": at, "n_lesions": n_les, "n_images": n_img,
            "max_sensitivity": float(sens[-1]) if len(sens) else 0.0}


def average_precision(all_preds, all_gts, iou_thr: float = 0.5) -> float:
    n_les = sum(len(g) for g in all_gts)
    if n_les == 0:
        return float("nan")
    scores, tps = _scored(all_preds, all_gts, iou_thr)
    if not len(tps):
        return 0.0
    tp_cum, fp_cum = np.cumsum(tps), np.cumsum(~tps)
    recall = tp_cum / n_les
    precision = tp_cum / (tp_cum + fp_cum)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
