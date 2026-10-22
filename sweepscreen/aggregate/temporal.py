"""Temporal aggregation.

1. **Persistence.** Frame lesion probabilities are averaged over a sliding window of ``window``
   consecutive *adequate* frames. Inadequate frames break the run, so a window never spans a
   quality gap. A lesion therefore has to be seen in ``window`` consecutive usable frames to score
   highly, which suppresses single-frame artefacts.
2. **Breast.** Breast score = maximum persistent score over the sweep. The breast is unreadable when
   the coverage rule of :func:`sweepscreen.quality.coverage_status` fails.
3. **Woman.** RECALL if any breast score reaches the operating threshold (a detected lesion is
   reported even if the other breast is unreadable); otherwise UNREADABLE if any breast is
   unreadable; otherwise NORMAL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..quality.gate import coverage_status


def persistence_scores(prob: Sequence[float], adequate: Optional[Sequence[bool]] = None, window: int = 5) -> np.ndarray:
    """Per-frame persistent score: mean of the last ``window`` probabilities within the current run of
    adequate frames (NaN for inadequate frames and for the first ``window - 1`` frames of each run)."""
    p = np.asarray(prob, np.float64)
    ok = np.ones(len(p), bool) if adequate is None else np.asarray(adequate, bool)
    out = np.full(len(p), np.nan)
    run_start = None
    for t in range(len(p)):
        if not ok[t]:
            run_start = None
            continue
        if run_start is None:
            run_start = t
        if t - run_start + 1 >= window:
            out[t] = p[t - window + 1:t + 1].mean()
    return out


@dataclass
class BreastResult:
    score: float                 # max persistent score (0 when no full window exists)
    peak_frame: Optional[int]    # frame index of the max
    readable: bool
    coverage: Dict[str, object] = field(default_factory=dict)


def aggregate_breast(prob: Sequence[float], adequate: Optional[Sequence[bool]] = None, window: int = 5,
                     min_frames: int = 30, min_fraction: float = 0.6) -> BreastResult:
    s = persistence_scores(prob, adequate, window)
    cov = coverage_status(np.ones(len(s), bool) if adequate is None else adequate, min_frames, min_fraction)
    if np.all(np.isnan(s)):
        return BreastResult(0.0, None, cov["readable"], cov)
    k = int(np.nanargmax(s))
    return BreastResult(float(s[k]), k, cov["readable"], cov)


@dataclass
class WomanResult:
    decision: str                        # "NORMAL" | "RECALL" | "UNREADABLE"
    score: float                         # max breast score
    breasts: Dict[str, BreastResult]
    reasons: List[str] = field(default_factory=list)


def decide_woman(breasts: Dict[str, BreastResult], threshold: float) -> WomanResult:
    score = max((b.score for b in breasts.values()), default=0.0)
    recall = [side for side, b in breasts.items() if b.score >= threshold]
    if recall:
        return WomanResult("RECALL", score, breasts, [f"persistent lesion, {s} breast" for s in recall])
    unreadable = [side for side, b in breasts.items() if not b.readable]
    if unreadable:
        return WomanResult("UNREADABLE", score, breasts, [f"inadequate coverage, {s} breast" for s in unreadable])
    return WomanResult("NORMAL", score, breasts, [])
