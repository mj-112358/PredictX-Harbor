"""Score calibration and calibration metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class PlattCalibrator:
    """Logistic recalibration ``sigmoid(a * logit(p) + b)``; refit on a small local sample at a new site."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p)[:, None], y)
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(_logit(p)[:, None])[:, 1]


def calibration_slope_intercept(p: np.ndarray, y: np.ndarray):
    """Calibration slope (logistic regression of y on logit p) and calibration-in-the-large intercept
    (offset model with slope fixed to 1). Ideal: slope 1, intercept 0."""
    lp = _logit(p)
    slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y).coef_[0, 0])
    # intercept with offset: solve for b in logit(P(y)) = lp + b by Newton iterations
    b = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(lp + b)))
        grad = np.sum(y - q)
        hess = -np.sum(q * (1 - q))
        step = grad / hess
        b -= step
        if abs(step) < 1e-10:
            break
    return slope, float(b)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """ECE with equal-count (quantile) bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p)
    ece = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            ece += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return float(ece)


def reliability_curve(p: np.ndarray, y: np.ndarray, bins: int = 10):
    """(mean predicted, observed fraction, count) per quantile bin."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            rows.append((float(p[chunk].mean()), float(y[chunk].mean()), int(len(chunk))))
    return rows
