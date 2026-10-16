"""Fit the deployed PredictX ensembles on the web feature sets and save them to ``artifacts/``.

Usage:  python -m service.train_models [--endpoints heart diabetes breast liver kidney]

Each endpoint's ensemble is trained on its complete cohort using exactly the fields collected by the web
form. The coronary ensemble additionally uses the Hungarian, Swiss and VA Long Beach cohorts.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import joblib

from predictx import cohorts
from predictx.ensemble import PredictXEnsemble

ARTIFACTS = os.environ.get("PREDICTX_ARTIFACTS", os.path.join(os.path.dirname(__file__), "..", "artifacts"))


def train(endpoint: str) -> dict:
    cohort = cohorts.load(endpoint, "web")
    model = PredictXEnsemble(cohort.spec, seed=0)
    X_aux, y_aux = cohort.auxiliary if cohort.auxiliary is not None else (None, None)
    start = time.time()
    model.fit(cohort.X, cohort.y, X_aux, y_aux)
    os.makedirs(ARTIFACTS, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACTS, f"{endpoint}.joblib"))
    n_train = len(cohort.y) + (0 if y_aux is None else len(y_aux))
    return {"endpoint": endpoint, "features": list(cohort.X.columns), "n_train": n_train,
            "fit_seconds": round(time.time() - start, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoints", nargs="+", default=list(cohorts.ENDPOINTS), choices=cohorts.ENDPOINTS)
    args = parser.parse_args()
    manifest = [train(e) for e in args.endpoints]
    with open(os.path.join(ARTIFACTS, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    for m in manifest:
        print(f"{m['endpoint']:9s} n={m['n_train']:4d}  features={len(m['features']):2d}  {m['fit_seconds']}s")


if __name__ == "__main__":
    main()
