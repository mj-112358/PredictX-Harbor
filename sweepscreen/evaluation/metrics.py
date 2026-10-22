"""Classification metrics with group-clustered bootstrap confidence intervals."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def sens_at_spec(y, p, spec: float) -> float:
    """Sensitivity at the threshold where specificity first reaches ``spec`` (threshold from the negatives)."""
    y, p = np.asarray(y).astype(bool), np.asarray(p, float)
    if y.all() or (~y).all():
        return float("nan")
    t = np.quantile(p[~y], spec, method="higher")
    return float((p[y] > t).mean())


def spec_at_sens(y, p, sens: float) -> float:
    y, p = np.asarray(y).astype(bool), np.asarray(p, float)
    if y.all() or (~y).all():
        return float("nan")
    t = np.quantile(p[y], 1 - sens, method="lower")
    return float((p[~y] < t).mean())


def threshold_for_spec(p_neg, spec: float) -> float:
    return float(np.quantile(np.asarray(p_neg, float), spec, method="higher"))


def bootstrap_ci(metric: Callable, y, p, groups=None, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Point estimate and percentile CI, resampling whole groups (patients/videos) with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    est = metric(y, p)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(ug))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([members[g] for g in pick])
        v = metric(y[idx], p[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return est, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(est), float(lo), float(hi)


def summary(y, p, groups=None, specs=(0.90, 0.95), n_boot: int = 2000) -> Dict[str, object]:
    out = {"n": int(len(y)), "n_pos": int(np.sum(y)), "auroc": bootstrap_ci(auroc, y, p, groups, n_boot)}
    for s in specs:
        out[f"sens@spec{int(s * 100)}"] = bootstrap_ci(lambda a, b, s=s: sens_at_spec(a, b, s), y, p, groups, n_boot)
    return out
