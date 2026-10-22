"""Value-of-information test acquisition for the coronary endpoint.

A patient starts from the baseline work-up. Each test reveals a block of biomarkers at a cost. The
controller orders ``argmax_a E[U(p_{t+1}) | a] - U(p_t) - lambda * C(a)`` while that quantity is positive.
The predictive distribution of an unseen block is the empirical distribution of that block among the
``k`` training patients nearest to the current patient on the observed biomarkers. One predictor is fitted
per test-availability pattern, so every intermediate state is scored by a model trained on exactly the
biomarkers it observes.
"""
from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, Iterable, Set, Tuple

import numpy as np
import pandas as pd

from .cohorts import FeatureSpec
from .learners import CatBoostLearner

#: test name -> (biomarkers revealed, cost)
HEART_TESTS: Dict[str, Tuple[list, float]] = {
    "exercise_ecg": (["thalach", "exang", "oldpeak", "slope"], 1.0),
    "fluoroscopy": (["ca"], 2.0),
    "thallium": (["thal"], 2.0),
}
HEART_BASELINE = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"]


def utility(p: np.ndarray, kind: str) -> np.ndarray:
    """``"information"``: negative predictive entropy; ``"decision"``: negative Bayes 0-1 risk."""
    if kind == "decision":
        return -np.minimum(p, 1 - p)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p * np.log(p) + (1 - p) * np.log(1 - p)


def mask_tests(X: pd.DataFrame, available: Iterable[str], tests=HEART_TESTS) -> pd.DataFrame:
    """Copy of ``X`` with the biomarkers of every test not in ``available`` set to missing."""
    X = X.copy()
    for t, (cols, _) in tests.items():
        if t not in available:
            X[cols] = np.nan
    return X


class AcquisitionPolicy:
    """Myopic value-of-information controller over the tests in ``tests``.

    ``fit`` trains one CatBoost predictor per subset of tests and stores the training table for the
    nearest-neighbour predictive distribution; ``run`` orders tests for a single patient.
    """

    def __init__(self, spec: FeatureSpec, seed: int = 0, k: int = 25, tests=HEART_TESTS, baseline=HEART_BASELINE):
        self.spec, self.seed, self.k, self.tests, self.baseline = spec, seed, k, tests, baseline

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "AcquisitionPolicy":
        patterns = [frozenset(s) for r in range(len(self.tests) + 1) for s in itertools.combinations(self.tests, r)]
        self.models_: Dict[FrozenSet[str], CatBoostLearner] = {
            s: CatBoostLearner(self.spec, self.seed).fit(mask_tests(X, s, self.tests), y) for s in patterns
        }
        self.X_ = X.reset_index(drop=True)
        numeric = self.X_.apply(pd.to_numeric)
        self.mean_, self.std_ = numeric.mean(), numeric.std().replace(0, 1)
        return self

    def predict(self, X: pd.DataFrame, available: Set[str]) -> np.ndarray:
        """Disease probability from the predictor trained on the biomarkers of ``available`` tests."""
        return self.models_[frozenset(available)].predict_proba(X)[:, 1]

    def _neighbours(self, x: pd.Series, columns: list) -> np.ndarray:
        """Indices of the ``k`` training patients closest to ``x`` on the standardised ``columns``."""
        Z = ((self.X_[columns] - self.mean_[columns]) / self.std_[columns]).fillna(0).values
        z = ((x[columns] - self.mean_[columns]) / self.std_[columns]).fillna(0).values.astype(float)
        return np.argsort(((Z - z) ** 2).sum(1))[: self.k]

    def net_values(self, x: pd.DataFrame, performed: Set[str], lam: float,
                   kind: str = "information") -> Dict[str, float]:
        """Net value of information of every test not yet performed for the one-row patient ``x``.

        The value of test ``a`` is ``E[U(p_{t+1}) | a] - U(p_t) - lambda * C(a)``, where the expectation is taken
        over the values of ``a`` observed in the ``k`` nearest training patients.
        """
        p = self.predict(x, performed)[0]
        observed = self.baseline + [c for t in performed for c in self.tests[t][0]]
        neighbours = self._neighbours(x.iloc[0], observed)
        values = {}
        for t in (t for t in self.tests if t not in performed):
            cols, c = self.tests[t]
            candidates = pd.concat([x] * self.k, ignore_index=True)
            candidates[cols] = self.X_.loc[neighbours, cols].values
            values[t] = utility(self.predict(candidates, performed | {t}), kind).mean() - utility(p, kind) - lam * c
        return values

    def run(self, patient: pd.Series, lam: float, kind: str = "information"):
        """Sequentially acquire tests for one patient; returns (final probability, total cost, tests ordered)."""
        ordered: Set[str] = set()
        cost = 0.0
        while True:
            x = mask_tests(patient.to_frame().T, ordered, self.tests)
            best, best_gain = None, 0.0
            for t, gain in self.net_values(x, ordered, lam, kind).items():
                if gain > best_gain:
                    best, best_gain = t, gain
            if best is None:
                return self.predict(x, ordered)[0], cost, ordered
            ordered.add(best)
            cost += self.tests[best][1]
