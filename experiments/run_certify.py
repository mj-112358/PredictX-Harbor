"""Threshold certification at a new site (lesion-level proxy).

    python -m experiments.run_certify

Unit: a group (patient / video). Positive score = the group's maximum frame probability over its
lesion frames (videos: maximum persistent score, window 5). Negative score = the same over lesion-free
groups (where a site has them). For each held-out site, target sensitivity τ and local sample size n:
200 times, draw n positive groups for certification, set the threshold by each method, and evaluate on
the remaining positive groups.

Methods: ``naive`` (threshold reaching τ on the fold detector's own validation positives, no local
data), ``conformal`` (split-conformal on the n local positives), ``binomial`` (exact Clopper–Pearson
lower bound ≥ τ on the n local positives).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sweepscreen.aggregate import persistence_scores
from sweepscreen.safety import binomial_threshold, conformal_threshold

from .common import parser, read_predictions, write_json

TARGETS = (0.90, 0.95)
SIZES = (20, 30, 59, 100)


def group_scores(pred: pd.DataFrame):
    pos, neg = [], []
    for gid, g in pred.groupby("group_id"):
        if g.video_id.notna().any():
            s = np.nanmax(np.r_[persistence_scores(g.sort_values("frame_index").frame_prob.values, window=5), -1])
            (pos if g.lesion.any() else neg).append(s)
        else:
            les = g[g.lesion == 1]
            if len(les):
                pos.append(les.frame_prob.max())
            else:
                neg.append(g.frame_prob.max())
    return np.asarray(pos), np.asarray(neg)


def evaluate_site(pos, neg, val_pos, draws=200, seed=0, targets=TARGETS, sizes=SIZES):
    rng = np.random.default_rng(seed)
    rows = []
    for tau in targets:
        t_naive = float(np.quantile(val_pos, 1 - tau, method="lower")) if len(val_pos) else np.nan
        rows.append({"target": tau, "n": 0, "method": "naive", "draws": 1, "threshold_found": 1.0,
                     "mean_sensitivity": float((pos >= t_naive).mean()), "coverage": float((pos >= t_naive).mean() >= tau),
                     "mean_specificity": float((neg < t_naive).mean()) if len(neg) else np.nan})
        for n in sizes:
            if n >= len(pos):
                continue
            for method, fn in (("conformal", conformal_threshold), ("binomial", binomial_threshold)):
                sens, spec, found = [], [], 0
                for _ in range(draws):
                    idx = rng.permutation(len(pos))
                    cal, rest = pos[idx[:n]], pos[idx[n:]]
                    t = fn(cal, tau)
                    if t is None:
                        continue
                    found += 1
                    sens.append((rest >= t).mean())
                    if len(neg):
                        spec.append((neg < t).mean())
                rows.append({"target": tau, "n": n, "method": method, "draws": draws, "threshold_found": found / draws,
                             "mean_sensitivity": float(np.mean(sens)) if sens else np.nan,
                             "coverage": float(np.mean(np.asarray(sens) >= tau)) if sens else np.nan,
                             "mean_specificity": float(np.mean(spec)) if spec else np.nan})
    return rows


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    out = {}
    for path in sorted((args.results / "predictions").glob("loso_*.jsonl")):
        site = path.stem[len("loso_"):]
        pred = read_predictions(path)
        models = joblib.load(args.results / "models" / f"loso_{site}.joblib")
        pos, neg = group_scores(pred)
        out[site] = {"n_pos_groups": int(len(pos)), "n_neg_groups": int(len(neg)),
                     "rows": evaluate_site(pos, neg, np.asarray(models.info["val_pos_scores"]),
                                           draws=20 if args.fast else 200, sizes=(2, 3) if args.fast else SIZES)}
    write_json(out, args.results / "certify.json")


if __name__ == "__main__":
    main()
