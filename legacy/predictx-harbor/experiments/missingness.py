"""Controlled-missingness ablation of the HARBOR availability channel.

Forty percent of every numeric feature of two complete cohorts (Wisconsin Diagnostic Breast Cancer and the
full ILPD table; ILPD age and sex are kept) is removed either completely at random (MCAR) or by
self-masking (MNAR: removal probability ``sigmoid(2 z + b)`` in the feature's own z-score, intercept ``b``
calibrated to 40%). HARBOR variants and reference learners are compared under protocol P1.

Usage:  python -m experiments.missingness --data WDBC --mechanism MNAR
Writes ``results/missingness/<data>-<mechanism>.csv`` (per-split AUROC); completed splits are skipped.
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from predictx import cohorts
from predictx.evaluation import repeated_cv
from predictx.harbor import HarborClassifier
from predictx.learners import make_learner

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "missingness")
RATE = 0.4


def inject_missing(X: pd.DataFrame, spec, mechanism: str, rng: np.random.Generator, protect=()) -> pd.DataFrame:
    X = X.copy()
    for c, t in spec.items():
        if t != "num" or c in protect:
            continue
        v = X[c].values
        observed = ~np.isnan(v)
        if mechanism == "MCAR":
            prob = np.full(len(v), RATE)
        else:
            z = np.nan_to_num((v - np.nanmean(v)) / (np.nanstd(v) + 1e-9))
            lo, hi = -20.0, 20.0
            for _ in range(60):  # bisection for the intercept that yields the target rate
                b = (lo + hi) / 2
                if (1 / (1 + np.exp(-(2.0 * z[observed] + b)))).mean() > RATE:
                    hi = b
                else:
                    lo = b
            prob = 1 / (1 + np.exp(-(2.0 * z + b)))
        X.loc[(rng.random(len(v)) < prob) & observed, c] = np.nan
    return X


def wdbc():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=[re.sub(r"\W+", "_", n) for n in data.feature_names])
    return X, (data.target == 0).astype(int), {c: "num" for c in X.columns}


def run(name: str, mechanism: str) -> None:
    if name == "WDBC":
        X0, y, spec = wdbc()
    else:
        liver = cohorts.load("liver")
        X0, y, spec = liver.X, liver.y, liver.spec
    X = inject_missing(X0, spec, mechanism, np.random.default_rng(123), protect=("age", "male"))
    mask = X.isna().values.astype(float)
    mask = mask[:, mask.std(0) > 0]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{name}-{mechanism}.csv")
    done = set(map(tuple, pd.read_csv(out)[["rep", "fold"]].values)) if os.path.exists(out) else set()
    for rep, fold, train, test in repeated_cv(y):
        if (rep, fold) in done:
            continue
        seed = rep * 10 + fold
        Xtr, Xte = X.iloc[train].reset_index(drop=True), X.iloc[test].reset_index(drop=True)
        scores = {"LR": make_learner("LR", spec, seed).fit(Xtr, y[train]).predict_proba(Xte)[:, 1]}
        for variant in ("value_only", "mask_concat", "two_channel"):
            model = HarborClassifier(spec, seed, variant=variant, n_members=1).fit(Xtr, y[train])
            scores[variant] = model.predict_proba(Xte)[:, 1]
        lr_mask = LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", max_iter=5000)
        scores["mask_only"] = lr_mask.fit(mask[train], y[train]).predict_proba(mask[test])[:, 1]
        rows = [dict(data=name, mechanism=mechanism, rep=rep, fold=fold, model=m, auroc=roc_auc_score(y[test], p))
                for m, p in scores.items()]
        pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(f"{name} {mechanism} rep {rep} fold {fold} done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, choices=["WDBC", "ILPD"])
    parser.add_argument("--mechanism", required=True, choices=["MCAR", "MNAR"])
    args = parser.parse_args()
    run(args.data, args.mechanism)


if __name__ == "__main__":
    main()
