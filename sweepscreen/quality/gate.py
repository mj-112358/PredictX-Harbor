"""Quality gate.

Training data: real frames from the public datasets are *adequate*; *inadequate* frames are
synthesised from real frames by the failure modes in :data:`FAILURE_MODES` (no public dataset labels
inadequate frames). A gradient-boosted classifier on :func:`~sweepscreen.quality.features.frame_features`
gives P(adequate). The decision threshold is set on adequate training frames so that a fixed fraction
(``target_pass``, default 0.97) of them pass.

Coverage: a breast is readable when it has at least ``min_frames`` adequate frames and an adequate
fraction of at least ``min_fraction``. Without probe tracking, coverage is measured by this
adequate-frame count over the fixed-duration sweep protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_NAMES, frame_features


def _blank(g, rng):
    """Probe in air / no gel: near-black frame with faint noise and a bright near-field band."""
    out = rng.normal(4, 2, g.shape).clip(0, 255)
    out[: max(2, g.shape[0] // 20)] = rng.normal(120, 20, (max(2, g.shape[0] // 20), g.shape[1]))
    return out.astype(np.uint8)


def _contact(g, rng):
    """Probe face in contact over at most half its width."""
    w = g.shape[1]
    keep = int(w * rng.uniform(0.15, 0.5))
    out = (g.astype(np.float32) * rng.uniform(0.02, 0.08)).astype(np.uint8)
    start = 0 if rng.random() < 0.5 else w - keep
    out[:, start:start + keep] = g[:, start:start + keep]
    return out


def _under_gain(g, rng):
    return (g.astype(np.float32) * rng.uniform(0.08, 0.2)).astype(np.uint8)


def _over_gain(g, rng):
    return np.clip(g.astype(np.float32) * rng.uniform(3.0, 5.0) + 40, 0, 255).astype(np.uint8)


def _motion_blur(g, rng):
    k = int(rng.integers(15, 31))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(g, -1, kernel)


def _reverberation(g, rng):
    """Low-information frame: horizontal reverberation bands on a dark background (no tissue)."""
    h, w = g.shape
    out = rng.normal(10, 4, (h, w))
    period = int(rng.integers(8, 20))
    for y in range(0, h, period):
        out[y:y + 2] += 120 * np.exp(-y / (0.4 * h))
    return out.clip(0, 255).astype(np.uint8)


FAILURE_MODES = {"blank": _blank, "contact": _contact, "under_gain": _under_gain, "over_gain": _over_gain,
                 "motion_blur": _motion_blur, "reverberation": _reverberation}


def synthesize_failure(gray: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    return FAILURE_MODES[mode](gray, rng)


@dataclass
class QualityGate:
    target_pass: float = 0.97
    seed: int = 0
    threshold_: float = field(default=0.5, init=False)

    def fit(self, adequate: Sequence[np.ndarray], failures_per_frame: int = 1) -> "QualityGate":
        """Fit on adequate frames plus ``failures_per_frame`` synthetic failures per frame (modes cycled)."""
        rng = np.random.default_rng(self.seed)
        modes = list(FAILURE_MODES)
        X, y = [], []
        for i, g in enumerate(adequate):
            X.append(frame_features(g))
            y.append(1)
            for j in range(failures_per_frame):
                X.append(frame_features(synthesize_failure(g, modes[(i * failures_per_frame + j) % len(modes)], rng)))
                y.append(0)
        X, y = np.asarray(X), np.asarray(y)
        self.model_ = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=self.seed)
        self.model_.fit(X, y)
        p_ok = self.model_.predict_proba(X[y == 1])[:, 1]
        self.threshold_ = float(np.quantile(p_ok, 1 - self.target_pass))
        return self

    def score(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """P(adequate) per frame."""
        X = np.asarray([frame_features(g) for g in frames])
        return self.model_.predict_proba(X)[:, 1]

    def adequate(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        return self.score(frames) >= self.threshold_

    @property
    def feature_names(self) -> tuple:
        return FEATURE_NAMES


def coverage_status(adequate_flags: Sequence[bool], min_frames: int = 30, min_fraction: float = 0.6) -> Dict[str, object]:
    """Readable / unreadable decision for one breast's sweep from its per-frame adequacy flags."""
    flags = np.asarray(adequate_flags, bool)
    n_ok = int(flags.sum())
    frac = float(flags.mean()) if len(flags) else 0.0
    return {"readable": bool(n_ok >= min_frames and frac >= min_fraction), "adequate_frames": n_ok,
            "adequate_fraction": frac, "frames": int(len(flags))}
