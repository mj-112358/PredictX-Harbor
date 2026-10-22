"""The PredictX ensemble: an equal-weight average of TabPFN, CatBoost, logistic regression and HARBOR."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .cohorts import FeatureSpec
from .learners import ENSEMBLE_MEMBERS, make_learner


class PredictXEnsemble:
    """Equal-weight probability average of heterogeneous learners.

    For the Brier score and the log loss, the ensemble's score never exceeds the average score of its
    members (Jensen's inequality); for the Brier score the gap equals the variance of member forecasts.
    """

    def __init__(self, spec: FeatureSpec, seed: int = 0, members: Sequence[str] = ENSEMBLE_MEMBERS):
        self.spec, self.seed, self.members = spec, seed, tuple(members)

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_aux: Optional[pd.DataFrame] = None,
            y_aux: Optional[np.ndarray] = None) -> "PredictXEnsemble":
        """Fit every member; optional auxiliary rows with the same features are appended to the training set."""
        if X_aux is not None:
            X = pd.concat([X, X_aux], ignore_index=True)
            y = np.r_[y, y_aux]
        self.learners_: Dict[str, object] = {
            name: make_learner(name, self.spec, self.seed).fit(X, y) for name in self.members
        }
        return self

    def prepare_for_serving(self) -> "PredictXEnsemble":
        """Enable cached in-context inference where available and compile every member once."""
        for learner in self.learners_.values():
            if hasattr(learner, "enable_cache"):
                learner.enable_cache()
        self.predict_proba(pd.DataFrame([[np.nan] * len(self.spec)], columns=list(self.spec), dtype=float))
        return self

    def member_probabilities(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Positive-class probability of each member."""
        return {name: learner.predict_proba(X)[:, 1] for name, learner in self.learners_.items()}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.mean(list(self.member_probabilities(X).values()), axis=0)
        return np.column_stack([1 - p, p])
