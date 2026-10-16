"""Resampling protocols, metrics and the corrected resampled t-test."""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

Split = Tuple[int, int, np.ndarray, np.ndarray]


def repeated_cv(y: np.ndarray, groups: Optional[np.ndarray] = None, folds: int = 5,
                repeats: int = 5) -> Iterator[Split]:
    """Protocol P1: stratified ``folds``-fold CV repeated ``repeats`` times (grouped when ``groups`` is given)."""
    index = np.arange(len(y))
    for rep in range(repeats):
        if groups is None:
            splitter = StratifiedKFold(folds, shuffle=True, random_state=rep).split(index, y)
        else:
            splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=rep).split(index, y, groups)
        for fold, (train, test) in enumerate(splitter):
            yield rep, fold, train, test


def half_splits(y: np.ndarray, groups: Optional[np.ndarray] = None, repeats: int = 5) -> Iterator[Split]:
    """Protocol P2: ``repeats`` stratified 50/50 train-test splits (the TabPFN benchmark protocol)."""
    index = np.arange(len(y))
    for rep in range(repeats):
        if groups is None:
            train, test = next(StratifiedKFold(2, shuffle=True, random_state=100 + rep).split(index, y))
        else:
            train, test = next(StratifiedGroupKFold(2, shuffle=True, random_state=100 + rep).split(index, y, groups))
        yield rep, 0, train, test


def split_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """AUROC, accuracy at threshold 0.5 and Brier score of probabilities ``p``."""
    return {"auroc": roc_auc_score(y, p), "accuracy": accuracy_score(y, p >= 0.5), "brier": brier_score_loss(y, p)}


def per_split_metrics(predictions: pd.DataFrame, models) -> pd.DataFrame:
    """Metrics of every model on every (rep, fold) split of a long prediction table."""
    rows = []
    for (rep, fold), g in predictions.groupby(["rep", "fold"]):
        for m in models:
            rows.append({"rep": rep, "fold": fold, "model": m, **split_metrics(g["y"].values, g[m].values)})
    return pd.DataFrame(rows)


def corrected_resampled_ttest(differences: np.ndarray, test_fraction: float, alpha: float = 0.05):
    """Nadeau & Bengio (2003) corrected resampled t interval for a mean paired difference.

    Returns ``(mean, lower, upper, p_value)``. The variance is inflated by ``1/k + n_test/n_train`` to
    account for the overlap between training sets of different resamples.
    """
    d = np.asarray(differences, dtype=float)
    k = len(d)
    if k < 2 or np.allclose(d, d[0]):
        return float(d.mean()), float(d.mean()), float(d.mean()), 1.0
    se = np.sqrt(d.var(ddof=1) * (1 / k + test_fraction / (1 - test_fraction)))
    t = stats.t.ppf(1 - alpha / 2, k - 1)
    mean = d.mean()
    p = 2 * stats.t.sf(abs(mean) / se, k - 1)
    return float(mean), float(mean - t * se), float(mean + t * se), float(p)
