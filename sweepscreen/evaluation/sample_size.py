"""Exact binomial calculations for the deployment-pathway study design (Clopper–Pearson throughout)."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, binom


def _lower(k, n, a):
    return beta.ppf(a, k, n - k + 1) if k > 0 else 0.0


def _upper(k, n, a):
    return beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0


def power_lower_above(n: int, p: float, floor: float, alpha_one_sided: float = 0.025) -> float:
    """P(one-sided lower confidence bound on a proportion > ``floor``) when the true proportion is ``p``."""
    k = np.arange(n + 1)
    ok = np.array([_lower(int(x), n, alpha_one_sided) for x in k]) > floor
    return float(binom.pmf(k, n, p)[ok].sum())


def power_upper_below(n: int, p: float, cap: float, alpha_one_sided: float = 0.025) -> float:
    k = np.arange(n + 1)
    ok = np.array([_upper(int(x), n, alpha_one_sided) for x in k]) < cap
    return float(binom.pmf(k, n, p)[ok].sum())


def iut_power(n_cancers: int, n_negatives: int, added_sens: float, added_recall: float,
              margin: float = 0.05, cap: float = 0.05) -> float:
    """Power of the intersection-union test: added sensitivity > ``margin`` AND added recall < ``cap``."""
    return power_lower_above(n_cancers, added_sens, margin) * power_upper_below(n_negatives, added_recall, cap)


def max_certifiable_count(n: int, cap: float, alpha_one_sided: float = 0.05) -> int:
    """Largest count k of positives among n for which the one-sided upper bound on the rate is < ``cap``."""
    return max(k for k in range(n) if _upper(k, n, alpha_one_sided) < cap)


def cp_interval(k: int, n: int, alpha: float = 0.05):
    return float(_lower(k, n, alpha / 2)), float(_upper(k, n, alpha / 2))
