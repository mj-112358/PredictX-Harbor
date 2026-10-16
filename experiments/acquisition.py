"""Experiment P3: value-of-information test acquisition on the Cleveland cohort.

Usage:  python -m experiments.acquisition
Writes ``results/acquisition.csv`` (one row per split and policy).
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from predictx import cohorts
from predictx.acquisition import HEART_TESTS, AcquisitionPolicy, mask_tests
from predictx.evaluation import repeated_cv

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "acquisition.csv")
LAMBDAS = {"information": [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2], "decision": [0.0, 0.01, 0.02, 0.04, 0.1]}
SHORT = {"exercise_ecg": "EXT", "fluoroscopy": "FLU", "thallium": "THA"}


def main() -> None:
    cohort = cohorts.load("heart")
    X, y = cohort.X, cohort.y
    rows = []
    for rep, fold, train, test in repeated_cv(y):
        policy = AcquisitionPolicy(cohort.spec, seed=rep * 10 + fold).fit(X.iloc[train], y[train])
        X_test, y_test = X.iloc[test].reset_index(drop=True), y[test]
        for r in range(len(HEART_TESTS) + 1):
            for panel in itertools.combinations(HEART_TESTS, r):
                p = policy.predict(mask_tests(X_test, set(panel)), set(panel))
                rows.append(dict(rep=rep, fold=fold, policy="fixed:" + ("+".join(SHORT[t] for t in panel) or "none"),
                                 cost=sum(HEART_TESTS[t][1] for t in panel), n_tests=len(panel),
                                 auroc=roc_auc_score(y_test, p), acc=accuracy_score(y_test, p >= 0.5)))
        for kind, lams in LAMBDAS.items():
            for lam in lams:
                runs = [policy.run(X_test.iloc[i], lam, kind) for i in range(len(X_test))]
                p = np.array([r[0] for r in runs])
                rows.append(dict(rep=rep, fold=fold, policy=f"voi-{kind}:{lam}", cost=np.mean([r[1] for r in runs]),
                                 n_tests=np.mean([len(r[2]) for r in runs]), auroc=roc_auc_score(y_test, p),
                                 acc=accuracy_score(y_test, p >= 0.5)))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"rep {rep} fold {fold} done", flush=True)


if __name__ == "__main__":
    main()
