"""Out-of-sample predictions for every learner on every cohort (protocols P1 and P2).

Usage:
    python -m experiments.benchmark --cohort heart --features full --protocol cv --models LR CAT HARBOR

Writes ``results/predictions/<cohort>-<features>-<protocol>/<model>.csv`` with columns
``rep, fold, row, y, p``. A model suffixed ``+aux`` is trained with the auxiliary heart cohorts appended
to every training fold and evaluated on the Cleveland test fold only. Completed splits are skipped, so an
interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from predictx import cohorts
from predictx.evaluation import half_splits, repeated_cv
from predictx.learners import make_learner

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "predictions")
ALL_MODELS = ["LR", "SVM", "KNN", "NB", "RF", "MLP", "XGB", "CAT", "PFN", "HARBOR"]
AUX_MODELS = ["LR+aux", "CAT+aux", "PFN+aux", "HARBOR+aux"]


def run(cohort_name: str, features: str, protocol: str, models: list) -> None:
    cohort = cohorts.load(cohort_name, features)
    X, y = cohort.X, cohort.y
    splits = repeated_cv(y, cohort.groups) if protocol == "cv" else half_splits(y, cohort.groups)
    out_dir = os.path.join(RESULTS, f"{cohort_name}-{features}-{protocol}")
    os.makedirs(out_dir, exist_ok=True)
    done = {}
    for m in models:
        path = os.path.join(out_dir, f"{m}.csv")
        done[m] = set()
        if os.path.exists(path):
            done[m] = set(map(tuple, pd.read_csv(path)[["rep", "fold"]].drop_duplicates().values))

    for rep, fold, train, test in splits:
        pending = [m for m in models if (rep, fold) not in done[m]]
        if not pending:
            continue
        seed = rep * 10 + fold
        X_train, y_train = X.iloc[train].reset_index(drop=True), y[train]
        X_test = X.iloc[test].reset_index(drop=True)
        for m in pending:
            start = time.time()
            name, _, arm = m.partition("+")
            Xf, yf = X_train, y_train
            if arm == "aux":
                X_aux, y_aux = cohort.auxiliary
                Xf, yf = pd.concat([X_train, X_aux], ignore_index=True), np.r_[y_train, y_aux]
            p = make_learner(name, cohort.spec, seed).fit(Xf, yf).predict_proba(X_test)[:, 1]
            frame = pd.DataFrame({"rep": rep, "fold": fold, "row": test, "y": y[test], "p": p})
            path = os.path.join(out_dir, f"{m}.csv")
            frame.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"[{cohort_name}/{features}/{protocol}] {m} rep {rep} fold {fold}: {time.time() - start:.1f}s",
                  flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", required=True, choices=cohorts.ENDPOINTS)
    parser.add_argument("--features", default="full", choices=["full", "web"])
    parser.add_argument("--protocol", default="cv", choices=["cv", "half"])
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    models = args.models or (ALL_MODELS + (AUX_MODELS if args.cohort == "heart" else []))
    run(args.cohort, args.features, args.protocol, models)


if __name__ == "__main__":
    main()
