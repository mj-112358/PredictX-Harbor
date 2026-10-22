"""Threshold certification at a new site.

Given scores of ``n`` locally confirmed positives (cancers or lesions), choose a decision threshold
so that the sensitivity target is met with a stated guarantee:

* :func:`binomial_threshold`: the highest threshold whose one-sided Clopper–Pearson lower bound on
  sensitivity is at least ``target`` at confidence ``1 - alpha``. If it exists, the population
  sensitivity at that threshold is ≥ ``target`` with probability ≥ 1 - α (assuming the local positives
  are representative). It needs at least :func:`min_positives` positives.
* :func:`conformal_threshold`: split-conformal threshold. A new positive exchangeable with the
  calibration positives scores at or above it with probability ≥ ``target`` (marginal guarantee,
  averaged over calibration sets). It needs ``n ≥ 1/(1 - target) - 1`` positives.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.stats import beta


def clopper_pearson(k: int, n: int, alpha: float = 0.05, side: str = "two") -> Tuple[float, float]:
    """Exact binomial interval for k successes out of n (``side`` = "two", "lower" or "upper")."""
    a = alpha / 2 if side == "two" else alpha
    lo = beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def min_positives(target: float, misses: int = 0, alpha: float = 0.05) -> int:
    """Smallest number of positives for which ``misses`` misses still certify ``target`` (one-sided)."""
    n = misses + 1
    while clopper_pearson(n - misses, n, alpha, "lower")[0] < target:
        n += 1
    return n


def binomial_threshold(pos_scores: np.ndarray, target: float, alpha: float = 0.05) -> Optional[float]:
    """Highest threshold t with CP lower bound of #(score ≥ t)/n ≥ target; None if no threshold qualifies."""
    s = np.sort(np.asarray(pos_scores, float))[::-1]       # descending
    n = len(s)
    best = None
    for k in range(1, n + 1):                               # threshold at the k-th highest score keeps k
        if clopper_pearson(k, n, alpha, "lower")[0] >= target:
            best = float(s[k - 1])
            break
    return best


def conformal_threshold(pos_scores: np.ndarray, target: float) -> Optional[float]:
    """Split-conformal threshold: the ⌊(n + 1)(1 - target)⌋-th smallest positive score."""
    s = np.sort(np.asarray(pos_scores, float))
    n = len(s)
    k = math.floor((n + 1) * (1 - target))
    if k < 1:
        return None
    return float(s[k - 1])
