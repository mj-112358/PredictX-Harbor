"""D: patch-level lesion detector on frozen encoder tokens.

Head: a three-layer convolutional network over the G x G patch-token grid (1x1 projection, 3x3
context convolution, 1x1 output), producing a lesion logit per patch. Training target per patch is
the fraction of the patch covered by the lesion (mask when available, otherwise box), with soft
binary cross-entropy and a positive-class weight (inverse patch prevalence, capped at 10).

From the patch probability map the detector derives

* a **frame lesion probability**: a logistic calibrator on (max patch probability, mean of the
  top-4 patches, fraction of patches above 0.5), fitted on frames held out from head training;
* **boxes**: connected components of the up-sampled map above a threshold, each scored by its
  peak probability;
* a **heatmap** (the up-sampled map) for display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

Box = Tuple[int, int, int, int]


def coverage_targets(boxes_list: Sequence[Sequence[Box]], masks: Optional[Sequence[Optional[np.ndarray]]],
                     size: int, grid: int) -> np.ndarray:
    """(N, grid, grid) fraction of each patch covered by lesion pixels (mask if given, else boxes)."""
    out = np.zeros((len(boxes_list), grid, grid), np.float32)
    for i, boxes in enumerate(boxes_list):
        m = None if masks is None else masks[i]
        if m is None:
            m = np.zeros((size, size), np.float32)
            for x0, y0, x1, y1 in boxes:
                m[max(0, y0):min(size, y1), max(0, x0):min(size, x1)] = 1
        out[i] = cv2.resize(m.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    return out


class PatchHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:   # (N, G, G, D) -> (N, G, G)
        return self.net(tokens.permute(0, 3, 1, 2)).squeeze(1)


def frame_stats(prob: np.ndarray) -> np.ndarray:
    """(N, 3) max, top-4 mean and fraction > 0.5 of each patch probability map."""
    flat = prob.reshape(len(prob), -1)
    top = np.sort(flat, axis=1)[:, -4:]
    return np.column_stack([flat.max(1), top.mean(1), (flat > 0.5).mean(1)])


@dataclass
class Detection:
    probability: float                  # frame-level lesion probability
    boxes: List[Box]                    # in the prepared (letterboxed) frame
    box_scores: List[float]
    heatmap: np.ndarray                 # (size, size) float32 in [0, 1]


@dataclass
class PatchDetector:
    dim: int
    grid: int = 16
    size: int = 224
    hidden: int = 128
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 64
    patience: int = 8
    box_threshold: float = 0.5
    max_pos_weight: float = 10.0
    seed: int = 0
    history_: list = field(default_factory=list, init=False)

    def _loss(self, logits, target, pos_weight):
        return nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    def fit(self, tokens: np.ndarray, targets: np.ndarray, frame_labels: np.ndarray, groups: np.ndarray,
            val_fraction: float = 0.2) -> "PatchDetector":
        """Train the head; the held-out ``val_fraction`` of groups is used for early stopping and to fit the frame calibrator."""
        rng = np.random.default_rng(self.seed)
        ug = np.unique(groups)
        val_groups = set(rng.choice(ug, size=max(1, int(len(ug) * val_fraction)), replace=False).tolist())
        is_val = np.array([g in val_groups for g in groups])
        tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
        torch.manual_seed(self.seed)
        self.head_ = PatchHead(self.dim, self.hidden)
        opt = torch.optim.AdamW(self.head_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos = float(targets[tr].mean())
        pos_weight = torch.tensor(min(self.max_pos_weight, (1 - pos) / max(pos, 1e-4)))
        Xt, Yt = torch.from_numpy(tokens.astype(np.float32)), torch.from_numpy(targets.astype(np.float32))
        best, best_state, stall = np.inf, None, 0
        for epoch in range(self.epochs):
            self.head_.train()
            order = rng.permutation(tr)
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                opt.zero_grad()
                loss = self._loss(self.head_(Xt[idx]), Yt[idx], pos_weight)
                loss.backward()
                opt.step()
            self.head_.eval()
            with torch.no_grad():
                vl = float(self._loss(self.head_(Xt[va]), Yt[va], pos_weight)) if len(va) else 0.0
            self.history_.append(vl)
            if vl < best - 1e-4:
                best, best_state, stall = vl, {k: v.clone() for k, v in self.head_.state_dict().items()}, 0
            else:
                stall += 1
                if stall >= self.patience:
                    break
        if best_state is not None:
            self.head_.load_state_dict(best_state)
        calib_idx = va if len(np.unique(frame_labels[va])) == 2 else np.arange(len(frame_labels))
        self.calibrator_ = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=1000)).fit(
            frame_stats(self.patch_probabilities(tokens[calib_idx])), frame_labels[calib_idx])
        return self

    @torch.no_grad()
    def patch_probabilities(self, tokens: np.ndarray, batch: int = 256) -> np.ndarray:
        self.head_.eval()
        out = []
        for s in range(0, len(tokens), batch):
            out.append(torch.sigmoid(self.head_(torch.from_numpy(tokens[s:s + batch].astype(np.float32)))).numpy())
        return np.concatenate(out) if out else np.zeros((0, self.grid, self.grid), np.float32)

    def frame_probability(self, tokens: np.ndarray) -> np.ndarray:
        return self.calibrator_.predict_proba(frame_stats(self.patch_probabilities(tokens)))[:, 1]

    def heatmap(self, patch_prob: np.ndarray) -> np.ndarray:
        return cv2.resize(patch_prob.astype(np.float32), (self.size, self.size), interpolation=cv2.INTER_LINEAR)

    def boxes(self, patch_prob: np.ndarray, threshold: Optional[float] = None, max_boxes: int = 5):
        """Boxes (prepared coordinates) and scores from one patch probability map."""
        hm = self.heatmap(patch_prob)
        t = self.box_threshold if threshold is None else threshold
        n, labels, stats, _ = cv2.connectedComponentsWithStats((hm >= t).astype(np.uint8), connectivity=8)
        found = []
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            found.append(((int(x), int(y), int(x + w), int(y + h)), float(hm[labels == i].max())))
        found.sort(key=lambda b: -b[1])
        found = found[:max_boxes]
        return [b for b, _ in found], [s for _, s in found]

    def detect(self, tokens: np.ndarray) -> List[Detection]:
        prob = self.patch_probabilities(tokens)
        frame_p = self.calibrator_.predict_proba(frame_stats(prob))[:, 1]
        out = []
        for i in range(len(prob)):
            b, s = self.boxes(prob[i])
            out.append(Detection(float(frame_p[i]), b, s, self.heatmap(prob[i])))
        return out

    def candidates(self, tokens: np.ndarray, thresholds=(0.3,)) -> List[List[Tuple[Box, float]]]:
        """Per-frame candidate boxes with scores at a low threshold (for FROC / mAP)."""
        prob = self.patch_probabilities(tokens)
        out = []
        for i in range(len(prob)):
            b, s = self.boxes(prob[i], threshold=min(thresholds), max_boxes=10)
            out.append(list(zip(b, s)))
        return out
