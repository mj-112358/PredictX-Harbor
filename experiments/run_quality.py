"""Quality-gate performance per held-out site (LOSO models).

    python -m experiments.run_quality

Pass rate on held-out real frames, and rejection rate on held-out synthetic failures of each mode.
"""
from __future__ import annotations

import joblib
import numpy as np

from sweepscreen.quality import FAILURE_MODES

from .common import load, parser, write_json


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    table, store = load(args)
    qfeat = np.asarray(store.array("qfeat"))
    qfail = np.load(store.root / "qfail.npy")
    qrows = np.load(store.root / "qfail_rows.npy")
    out = {}
    for site in sorted(table.site.unique()):
        m = joblib.load(args.results / "models" / f"loso_{site}.joblib").quality
        rows = np.flatnonzero(table.site.values == site)
        res = {"n_frames": int(len(rows)), "pass_rate": float((m.score_features(qfeat[rows]) >= m.threshold_).mean()),
               "threshold": m.threshold_}
        sel = np.isin(qrows, rows)
        res["n_failure_sources"] = int(sel.sum())
        for k, mode in enumerate(FAILURE_MODES):
            if sel.any():
                res[f"reject_{mode}"] = float((m.score_features(qfail[sel, k]) < m.threshold_).mean())
        out[site] = res
    write_json(out, args.results / "quality.json")


if __name__ == "__main__":
    main()
