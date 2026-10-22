"""CPU runtime of each pipeline stage, measured on real frames with the registered model.

    python -m experiments.run_runtime

Reports median milliseconds per frame for preprocessing, quality features + gate, encoder (batch 1
and batch 16), detector head + boxes, malignancy head, and end-to-end sweep throughput (frames per
second), plus model sizes and peak resident memory.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from sweepscreen.data.loaders.common import read_raw
from sweepscreen.features import prepare_record
from sweepscreen.models.detector import frame_stats
from sweepscreen.models.encoder import DEFAULT_WEIGHTS, Encoder
from sweepscreen.models.malignancy import lesion_features
from sweepscreen.pipeline import SweepScreen
from sweepscreen.quality import frame_features

from .common import ROOT, load, parser, write_json


def _timeit(fn, repeats):
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000)


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    table, _ = load(args)
    reg = json.loads((args.models_dir / "registry.json").read_text())
    entry = [e for e in reg if e["current"]][-1]
    enc = Encoder(config=Encoder.tiny_config(), threads=args.threads) if args.fast else Encoder(threads=args.threads)
    app = SweepScreen.load(args.models_dir / entry["file"], enc)
    rng = np.random.default_rng(0)
    paths = table.image_path.values[rng.choice(len(table), size=min(32, len(table)), replace=False)]
    raws = [read_raw(p) for p in paths]
    reps = 3 if args.fast else 10

    prep = [prepare_record(r, []) for r in raws]
    out = {"cpu": platform.processor() or platform.machine(), "threads": args.threads, "torch": torch.__version__,
           "frames_timed": len(raws)}
    out["preprocess_ms"] = _timeit(lambda: prepare_record(raws[0], []), reps)
    out["quality_ms"] = _timeit(lambda: app.models.quality.score_features(frame_features(prep[0].cropped)[None]), reps)
    one = np.stack([prep[0].image])
    batch = np.stack([p.image for p in prep[:16]])
    out["encoder_batch1_ms"] = _timeit(lambda: enc(one), reps)
    out["encoder_batch16_ms_per_frame"] = _timeit(lambda: enc(batch), max(2, reps // 3)) / len(batch)
    cls, tok = enc(one)
    det = app.models.detector

    def head():
        pr = det.patch_probabilities(tok)
        det.calibrator_.predict_proba(frame_stats(pr))
        det.boxes(pr[0])
    out["detector_head_ms"] = _timeit(head, reps)
    if app.models.malignancy is not None:
        X = lesion_features(cls, tok, [(50, 50, 120, 120)], [None])
        out["malignancy_ms"] = _timeit(lambda: app.models.malignancy.predict(X), reps)
    frames = raws[:16]
    t = time.perf_counter()
    app.analyze_frames(frames)
    out["end_to_end_frames_per_second"] = float(len(frames) / (time.perf_counter() - t))
    bundle = args.models_dir / entry["file"]
    out["bundle_mb"] = bundle.stat().st_size / 1e6
    w = DEFAULT_WEIGHTS / "model.safetensors"
    out["encoder_weights_mb"] = w.stat().st_size / 1e6 if w.exists() else None
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    out["cpu_count"] = os.cpu_count()
    write_json(out, args.results / "runtime.json")


if __name__ == "__main__":
    main()
