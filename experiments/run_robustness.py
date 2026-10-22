"""Robustness to handheld-probe degradations on held-out still-image sites.

    python -m experiments.run_robustness

For each still site S and each degradation at each graded level, the held-out images of S are
re-processed with the degradation, re-encoded, and scored by the LOSO model that never saw S.
Output ``results/robustness/<S>.jsonl``: frame probability, malignancy score (ground-truth box) and
quality score per image, degradation and level.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sweepscreen.data.loaders import STILL_SITES
from sweepscreen.data.loaders.common import read_raw
from sweepscreen.data.schema import from_frame
from sweepscreen.features import prepare_record, union_mask
from sweepscreen.models.detector import frame_stats
from sweepscreen.models.encoder import Encoder
from sweepscreen.models.malignancy import lesion_features
from sweepscreen.preprocess.augment import LEVELS, make
from sweepscreen.quality import frame_features

from .common import load, parser, save_predictions


def score(models, enc, records, degrade):
    prepared = []
    for r in records:
        raw = read_raw(r.image_path)
        prepared.append(prepare_record(raw, r.boxes, union_mask(r.mask_paths, raw.shape) if r.label != "normal" else None,
                                       degrade=degrade))
    cls, tok = enc(np.stack([p.image for p in prepared]))
    prob = models.detector.patch_probabilities(tok)
    fp = models.detector.calibrator_.predict_proba(frame_stats(prob))[:, 1]
    gtb = [max(p.boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if p.boxes else None for p in prepared]
    mal = models.malignancy.predict(lesion_features(cls, tok, gtb, [r.age for r in records])) \
        if models.malignancy is not None else np.full(len(records), np.nan)
    q = models.quality.score_features(np.stack([frame_features(p.cropped) for p in prepared]))
    return fp, mal, q


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--sites", nargs="*")
    args = p.parse_args(argv)
    table, _ = load(args)
    enc = Encoder(config=Encoder.tiny_config()) if args.fast else Encoder()
    sites = args.sites or [s for s in STILL_SITES if s in set(table.site)]
    levels = {k: v[:1] for k, v in LEVELS.items()} if args.fast else LEVELS
    for site in sites:
        models = joblib.load(args.results / "models" / f"loso_{site}.joblib")
        sub = table[(table.site == site) & ~table.discordant.astype(bool)]
        records = from_frame(sub)
        rows = []
        for name, lv in levels.items():
            for level in lv:
                for s in range(0, len(records), 128):
                    chunk = records[s:s + 128]
                    fp, mal, q = score(models, enc, chunk, make(name, level, seed=s))
                    for k, r in enumerate(chunk):
                        rows.append({"uid": r.uid, "group_id": sub.group_id.iloc[s + k], "label": r.label,
                                     "lesion": int(r.lesion_present), "degradation": name, "level": level,
                                     "frame_prob": float(fp[k]), "mal_gt": float(mal[k]), "quality": float(q[k]),
                                     "adequate": bool(q[k] >= models.quality.threshold_)})
                print(f"[robustness] {site} {name}@{level}", flush=True)
        save_predictions(pd.DataFrame(rows), args.results / "robustness" / f"{site}.jsonl")


if __name__ == "__main__":
    main()
