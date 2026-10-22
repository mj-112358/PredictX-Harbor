"""Malignancy head: suspicion score for a detected lesion.

Input per lesion frame: the CLS token, the mean of the patch tokens inside the lesion box (the
ground-truth box during training, the detected box at inference), and age with a missing-value
indicator (age is recorded for BrEaST only). Model: standardisation + L2 logistic regression with the
regularisation strength chosen by 5-fold grouped cross-validation on log loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Box = Tuple[int, int, int, int]


def roi_pool(patches: np.ndarray, box: Optional[Box], size: int = 224) -> np.ndarray:
    """Mean of the patch tokens whose cells intersect ``box`` (whole grid if ``box`` is None)."""
    g = patches.shape[0]
    if box is None:
        return patches.reshape(-1, patches.shape[-1]).mean(0)
    cell = size / g
    x0, y0, x1, y1 = box
    c0, r0 = int(max(0, x0 // cell)), int(max(0, y0 // cell))
    c1, r1 = int(min(g, np.ceil(x1 / cell))), int(min(g, np.ceil(y1 / cell)))
    if c1 <= c0 or r1 <= r0:
        return patches.reshape(-1, patches.shape[-1]).mean(0)
    return patches[r0:r1, c0:c1].reshape(-1, patches.shape[-1]).mean(0)


def lesion_features(cls: np.ndarray, patches: np.ndarray, boxes: Sequence[Optional[Box]],
                    ages: Sequence[Optional[float]], size: int = 224) -> np.ndarray:
    """(N, 2D + 2) feature matrix: CLS, ROI-pooled tokens, age (centred at 50, 0 if missing), age-missing flag."""
    age = np.array([a if a is not None and np.isfinite(a) else np.nan for a in ages], np.float32)
    missing = np.isnan(age).astype(np.float32)
    age_f = np.nan_to_num((age - 50.0) / 15.0)
    roi = np.stack([roi_pool(patches[i], boxes[i], size) for i in range(len(cls))])
    return np.column_stack([cls, roi, age_f, missing]).astype(np.float32)


@dataclass
class MalignancyHead:
    seed: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "MalignancyHead":
        n_splits = int(min(5, len(np.unique(groups))))
        cv = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
        self.model_ = make_pipeline(StandardScaler(), LogisticRegressionCV(
            Cs=np.logspace(-4, 1, 11), cv=cv, scoring="neg_log_loss", max_iter=5000))
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]
