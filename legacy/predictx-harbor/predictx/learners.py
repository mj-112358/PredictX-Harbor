"""Learners compared in the paper, behind one scikit-learn-style interface.

Every learner is created with :func:`make_learner` and exposes ``fit(X, y)`` and ``predict_proba(X)``
(an ``(n, 2)`` array), where ``X`` is a :class:`pandas.DataFrame` with ``NaN`` for unrecorded values.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from .cohorts import FeatureSpec
from .harbor import HarborClassifier
from .preprocessing import tabular_preprocessor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#: Model families used in published comparisons on these cohorts.
LITERATURE_STANDARD = ("LR", "SVM", "KNN", "NB", "RF", "MLP", "XGB")
#: Members of the PredictX ensemble.
ENSEMBLE_MEMBERS = ("PFN", "CAT", "LR", "HARBOR")


class PipelineLearner:
    """A scikit-learn estimator preceded by :func:`tabular_preprocessor`."""

    def __init__(self, estimator, spec: FeatureSpec, scale: bool = True):
        self.estimator, self.spec, self.scale = estimator, spec, scale

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.pipeline_ = make_pipeline(tabular_preprocessor(self.spec, self.scale), self.estimator)
        kwargs = {} if sample_weight is None else {f"{self.pipeline_.steps[-1][0]}__sample_weight": sample_weight}
        self.pipeline_.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X)


class TunedXGBoost:
    """XGBoost tuned with Optuna (TPE, ``n_trials`` trials, inner 3-fold CV AUROC). Native missing values."""

    def __init__(self, spec: FeatureSpec, seed: int = 0, n_trials: int = 25):
        self.spec, self.seed, self.n_trials = spec, seed, n_trials

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        import optuna
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        Xa = X.values.astype(float)
        folds = list(StratifiedKFold(3, shuffle=True, random_state=self.seed).split(Xa, y))

        def params_from(trial):
            return dict(n_estimators=trial.suggest_int("n", 50, 500, log=True),
                        max_depth=trial.suggest_int("d", 2, 6),
                        learning_rate=trial.suggest_float("lr", 0.01, 0.3, log=True),
                        subsample=trial.suggest_float("ss", 0.5, 1.0),
                        colsample_bytree=trial.suggest_float("cs", 0.5, 1.0),
                        min_child_weight=trial.suggest_float("mcw", 1, 10, log=True),
                        reg_lambda=trial.suggest_float("l2", 1e-3, 10, log=True))

        def objective(trial):
            prm = params_from(trial)
            scores = []
            for a, b in folds:
                m = xgb.XGBClassifier(**prm, n_jobs=1, random_state=self.seed, verbosity=0).fit(Xa[a], y[a])
                scores.append(roc_auc_score(y[b], m.predict_proba(Xa[b])[:, 1]))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.n_trials)
        b = study.best_params
        self.model_ = xgb.XGBClassifier(n_estimators=b["n"], max_depth=b["d"], learning_rate=b["lr"],
                                        subsample=b["ss"], colsample_bytree=b["cs"], min_child_weight=b["mcw"],
                                        reg_lambda=b["l2"], n_jobs=1, random_state=self.seed, verbosity=0)
        self.model_.fit(Xa, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X.values.astype(float))


class CatBoostLearner:
    """CatBoost with default hyperparameters (1000 iterations); categorical features declared natively."""

    def __init__(self, spec: FeatureSpec, seed: int = 0):
        self.spec, self.seed = spec, seed

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, t in self.spec.items():
            if t != "num":
                X[c] = X[c].fillna(-1).astype(int).astype(str)
        return X

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        from catboost import CatBoostClassifier

        categorical = [c for c, t in self.spec.items() if t != "num"]
        self.model_ = CatBoostClassifier(iterations=1000, random_seed=self.seed, verbose=0, thread_count=1,
                                         cat_features=categorical, allow_writing_files=False)
        self.model_.fit(self._frame(X), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._frame(X))


class TabPFNLearner:
    """TabPFN v2 (Hollmann et al., Nature 2025), in-context prior-fitted transformer, 8 estimators."""

    def __init__(self, spec: FeatureSpec, seed: int = 0, n_estimators: int = 8,
                 fit_mode: str = "fit_preprocessors"):
        self.spec, self.seed, self.n_estimators, self.fit_mode = spec, seed, n_estimators, fit_mode

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        from tabpfn import TabPFNClassifier

        categorical = [i for i, t in enumerate(self.spec.values()) if t != "num"]
        self.X_, self.y_ = X.values.astype(np.float32), np.asarray(y)
        self.model_ = TabPFNClassifier(device="cpu", n_estimators=self.n_estimators, random_state=self.seed,
                                       categorical_features_indices=categorical or None,
                                       ignore_pretraining_limits=True, fit_mode=self.fit_mode)
        self.model_.fit(self.X_, self.y_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if getattr(self, "model_", None) is None:  # restored from disk: re-condition on the stored context
            self.fit(pd.DataFrame(self.X_), self.y_)
        return self.model_.predict_proba(X.values.astype(np.float32))

    def enable_cache(self) -> None:
        """Cache the transformer's key-value states of the training context: identical predictions,
        sub-second inference for single patients, at the cost of a one-off conditioning pass."""
        self.fit_mode = "fit_with_cache"
        self.fit(pd.DataFrame(self.X_), self.y_)

    def __getstate__(self):
        # TabPFN is an in-context learner: its fitted state is the training table, so only that is persisted.
        state = self.__dict__.copy()
        state["model_"] = None
        return state


def make_learner(name: str, spec: FeatureSpec, seed: int = 0):
    """Instantiate learner ``name`` for a table described by ``spec``."""
    if name == "LR":
        return PipelineLearner(LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", max_iter=5000), spec)
    if name == "SVM":
        grid = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1]}
        return PipelineLearner(GridSearchCV(SVC(probability=True, random_state=seed), grid, cv=3,
                                            scoring="roc_auc"), spec)
    if name == "KNN":
        return PipelineLearner(GridSearchCV(KNeighborsClassifier(weights="distance"),
                                            {"n_neighbors": [5, 11, 21, 31]}, cv=3, scoring="roc_auc"), spec)
    if name == "NB":
        return PipelineLearner(GaussianNB(), spec)
    if name == "RF":
        return PipelineLearner(RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=1,
                                                      random_state=seed), spec, scale=False)
    if name == "MLP":
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=1000, random_state=seed)
        return PipelineLearner(GridSearchCV(mlp, {"alpha": [1e-4, 1e-2, 1]}, cv=3, scoring="roc_auc"), spec)
    if name == "XGB":
        return TunedXGBoost(spec, seed)
    if name == "CAT":
        return CatBoostLearner(spec, seed)
    if name == "PFN":
        return TabPFNLearner(spec, seed)
    if name == "HARBOR":
        return HarborClassifier(spec, seed)
    raise KeyError(f"unknown learner {name!r}")
