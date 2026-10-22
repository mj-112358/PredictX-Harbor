"""Leave-one-site-out and within-site (internal) evaluation of the detector, malignancy head and quality gate.

    python -m experiments.run_loso

For each site S: fit every component on all other sites (cross-site duplicate groups removed) and
predict S → ``results/predictions/loso_<S>.jsonl``; the fold models are saved to
``results/models/loso_<S>.joblib`` for the robustness experiment. Internal reference: 5-fold grouped
CV within S → ``results/predictions/internal_<S>.jsonl``. With ``--no-aug --results results/ablation_noaug``
the same is run without augmented training copies (augmentation ablation).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sweepscreen.data import splits
from sweepscreen.features import FeatureStore
from sweepscreen.train import fit, predict

from .common import config, load, parser, save_predictions, write_json


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--sites", nargs="*")
    p.add_argument("--no-internal", action="store_true")
    p.add_argument("--no-aug", action="store_true", help="ablation: train without augmented copies")
    args = p.parse_args(argv)
    table, store = load(args)
    cfg = config(args)
    aug_root = args.features.parent / "aug"
    aug = None if args.no_aug or not (aug_root / "index.csv").exists() else FeatureStore(aug_root)
    info = {}
    for site, tr, te in splits.leave_one_site_out(table, args.sites):
        splits.check_disjoint(table, tr, te)
        models = fit(store, table, tr, cfg, aug)
        save_predictions(predict(models, store, table, te), args.results / "predictions" / f"loso_{site}.jsonl")
        (args.results / "models").mkdir(parents=True, exist_ok=True)
        joblib.dump(models, args.results / "models" / f"loso_{site}.joblib", compress=3)
        info[site] = {"train_rows": int(len(tr)), "test_rows": int(len(te)), "frame_threshold": models.frame_threshold,
                      "quality_threshold": models.quality.threshold_, **{k: v for k, v in models.info.items()
                                                                          if not k.startswith("val_")}}
        print(f"[loso] {site}: train {len(tr)}, test {len(te)}", flush=True)
        if args.no_internal:
            continue
        rows = np.flatnonzero(table.site.values == site)
        sub = table.iloc[rows].reset_index(drop=True)
        n_groups = sub.group_id.nunique()
        if n_groups < 10 or sub.label.nunique() < 2:
            continue
        preds = []
        for k, itr, ite in splits.group_kfold(sub, n_splits=5, seed=0):
            m = fit(store, table, rows[itr], cfg, aug)
            preds.append(predict(m, store, table, rows[ite]).assign(fold=k))
        save_predictions(pd.concat(preds), args.results / "predictions" / f"internal_{site}.jsonl")
        print(f"[internal] {site}: done", flush=True)
    write_json(info, args.results / "loso_info.json")


if __name__ == "__main__":
    main()
