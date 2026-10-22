"""Temporal aggregation on BreastVid clips, with the detector trained without BreastVid (LOSO fold).

    python -m experiments.run_video

Frames the quality gate rejects break persistence runs. For each window w: clip detection rate (clip's maximum persistent score ≥ the fold's
operating threshold), false-positive rate on lesion-free frames inside clips (persistent score ≥
threshold), and frames from lesion onset to first detection. Video-level suspicion: mean malignancy
score over frames whose frame probability exceeds the threshold → benign-vs-malignant AUROC per clip.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sweepscreen.aggregate import persistence_scores
from sweepscreen.evaluation.metrics import auroc, bootstrap_ci

from .common import parser, read_predictions, write_json

WINDOWS = (1, 3, 5, 9)


def evaluate(pred: pd.DataFrame, threshold: float, quality_threshold: float, windows=WINDOWS) -> dict:
    out = {"threshold": threshold, "quality_threshold": quality_threshold, "n_videos": int(pred.video_id.nunique()), "windows": {}}
    for w in windows:
        det, fp_num, fp_den, delays = [], 0, 0, []
        for vid, g in pred.sort_values("frame_index").groupby("video_id"):
            s = persistence_scores(g.frame_prob.values, g.quality.values >= quality_threshold, window=w)
            hit = np.nan_to_num(s, nan=-1) >= threshold
            det.append(bool(hit.any()))
            lesion = g.lesion.values.astype(bool)
            free = ~lesion & ~np.isnan(s)
            fp_num += int((hit & free).sum())
            fp_den += int(free.sum())
            if hit.any() and lesion.any():
                delays.append(int(np.argmax(hit)) - int(np.argmax(lesion)))
        out["windows"][w] = {"clip_detection_rate": float(np.mean(det)), "clips_detected": int(np.sum(det)),
                             "lesion_free_frames": fp_den,
                             "lesion_free_fp_rate": float(fp_num / fp_den) if fp_den else float("nan"),
                             "median_frames_to_detection": float(np.median(delays)) if delays else float("nan")}
    rows = []
    for vid, g in pred.groupby("video_id"):
        above = g[g.frame_prob >= threshold]
        src = above if len(above) else g
        rows.append({"video": vid, "malignant": int((g.label == "malignant").any()), "suspicion": float(src.mal_det.mean()),
                     "suspicion_gt": float(g[g.lesion == 1].mal_gt.mean()) if (g.lesion == 1).any() else np.nan})
    v = pd.DataFrame(rows).dropna()
    out["video_malignancy_auroc_detected_box"] = bootstrap_ci(auroc, v.malignant.values, v.suspicion.values)
    out["video_malignancy_auroc_gt_box"] = bootstrap_ci(auroc, v.malignant.values, v.suspicion_gt.values)
    return out


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    pred = read_predictions(args.results / "predictions" / "loso_BreastVid.jsonl")
    models = joblib.load(args.results / "models" / "loso_BreastVid.joblib")
    write_json(evaluate(pred, models.frame_threshold, models.quality.threshold_), args.results / "video.json")


if __name__ == "__main__":
    main()
