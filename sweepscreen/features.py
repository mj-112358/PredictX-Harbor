"""Shared preprocessing → encoding stage.

``prepare_record`` turns a record into the model input (224 x 224 float image), lesion boxes and patch
coverage targets in prepared coordinates, and the cropped uint8 frame the quality gate reads.
``extract`` runs it over a record table and writes a feature store:

    data/features/<variant>/index.csv     uid order, prepared boxes, crop and letterbox parameters
    data/features/<variant>/cls.npy       (N, D) float32 CLS tokens
    data/features/<variant>/tokens.npy    (N, G, G, D) float16 patch tokens (memory-mapped)
    data/features/<variant>/targets.npy   (N, G, G) float16 lesion coverage per patch
    data/features/<variant>/qfeat.npy     (N, F) float32 quality features
    data/features/<variant>/qfail.npy     (M, K, F) quality features of K synthetic failures for M sampled rows
    data/features/<variant>/qfail_rows.npy (M,) row indices of those samples
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

from .data.loaders.common import read_mask, read_raw
from .data.schema import from_frame
from .preprocess import crop_to_content, letterbox, normalize_intensity, remove_annotations
from .quality import FAILURE_MODES, frame_features, synthesize_failure

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features"


@dataclass
class PreparedRecord:
    image: np.ndarray            # (size, size) float32
    cropped: np.ndarray          # cropped (and degraded) uint8 frame, for the quality gate
    boxes: list                  # prepared coordinates
    targets: np.ndarray          # (grid, grid) coverage
    crop: tuple
    scale: float
    pad: tuple


def prepare_record(raw: np.ndarray, boxes, mask: Optional[np.ndarray] = None, size: int = 224, grid: int = 16,
                   degrade=None) -> PreparedRecord:
    gray = remove_annotations(raw)
    crop = crop_to_content(gray)
    g = crop.apply(gray)
    b = crop.boxes(list(boxes))
    m = crop.apply(mask) if mask is not None else None
    if degrade is not None:
        h0 = g.shape[0]
        g, b = degrade(g, b)
        if m is not None and g.shape[0] != h0:     # geometric degradation (depth crop): crop the mask too
            m = m[:g.shape[0]]
    img, lb = letterbox(normalize_intensity(g), size)
    pb = lb.boxes(b)
    if m is not None and m.any():
        mm, _ = letterbox(m.astype(np.float32), size)
    else:
        mm = np.zeros((size, size), np.float32)
        for x0, y0, x1, y1 in pb:
            mm[max(0, y0):y1, max(0, x0):x1] = 1
    targets = cv2.resize(mm, (grid, grid), interpolation=cv2.INTER_AREA)
    return PreparedRecord(img, g, pb, targets.astype(np.float32), (crop.x0, crop.y0, crop.x1, crop.y1), lb.scale,
                          (lb.pad_x, lb.pad_y))


def union_mask(paths: List[str], shape) -> Optional[np.ndarray]:
    if not paths:
        return None
    m = np.zeros(shape[:2], np.uint8)
    for p in paths:
        mk = read_mask(p)
        if mk.shape != m.shape:
            mk = cv2.resize(mk, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        m |= mk
    return m


def extract(table: pd.DataFrame, encoder, variant: str = "clean", degrade=None, qfail_every: int = 5,
            out_root: Path = FEATURES, chunk: int = 256, seed: int = 0) -> Path:
    """Write the feature store for ``table`` rows (in order). Resumable: finished chunks are skipped."""
    out = out_root / variant
    out.mkdir(parents=True, exist_ok=True)
    records = from_frame(table)
    n, G, D = len(records), encoder.grid, encoder.dim
    tokens = np.lib.format.open_memmap(out / "tokens.npy", mode="r+" if (out / "tokens.npy").exists() else "w+",
                                       dtype=np.float16, shape=(n, G, G, D))
    arrays = {}
    for name, shape, dt in (("cls", (n, D), np.float32), ("targets", (n, G, G), np.float16),
                            ("qfeat", (n, len(frame_features(np.zeros((8, 8), np.uint8)))), np.float32)):
        path = out / f"{name}.npy"
        arrays[name] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=dt, shape=shape)
    done_path = out / "done_chunks.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    rng = np.random.default_rng(seed)
    qfail_rows = list(range(0, n, qfail_every)) if variant == "clean" else []
    qfail = {}
    index_rows = [None] * n
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        if start in done and (out / "index_part" / f"{start}.json").exists():
            continue
        imgs, part = [], []
        for i in range(start, stop):
            r = records[i]
            raw = read_raw(r.image_path)
            pr = prepare_record(raw, r.boxes, union_mask(r.mask_paths, raw.shape) if r.label != "normal" else None,
                                degrade=degrade)
            imgs.append(pr.image)
            arrays["targets"][i] = pr.targets
            arrays["qfeat"][i] = frame_features(pr.cropped)
            if i in qfail_rows:
                qfail[i] = np.stack([frame_features(synthesize_failure(pr.cropped, m, rng)) for m in FAILURE_MODES])
            part.append({"row": i, "uid": r.uid, "boxes": pr.boxes, "crop": pr.crop, "scale": pr.scale, "pad": pr.pad})
        cls, patches = encoder(np.stack(imgs))
        arrays["cls"][start:stop] = cls
        tokens[start:stop] = patches.astype(np.float16)
        (out / "index_part").mkdir(exist_ok=True)
        (out / "index_part" / f"{start}.json").write_text(json.dumps(part))
        if qfail:
            np.save(out / "index_part" / f"qfail_{start}.npy", np.stack([qfail[i] for i in sorted(qfail)]))
            np.save(out / "index_part" / f"qfailrows_{start}.npy", np.array(sorted(qfail)))
            qfail = {}
        done.add(start)
        done_path.write_text(json.dumps(sorted(done)))
        print(f"[{variant}] {stop}/{n}", flush=True)
    tokens.flush()
    for a in arrays.values():
        a.flush()
    # assemble index and quality-failure arrays
    for p in sorted((out / "index_part").glob("*.json"), key=lambda p: int(p.stem)):
        for row in json.loads(p.read_text()):
            index_rows[row["row"]] = row
    idx = pd.DataFrame(index_rows)
    idx["boxes"] = idx.boxes.map(json.dumps)
    idx.to_csv(out / "index.csv", index=False)
    parts = sorted((out / "index_part").glob("qfail_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    if parts:
        np.save(out / "qfail.npy", np.concatenate([np.load(p) for p in parts]))
        np.save(out / "qfail_rows.npy", np.concatenate([np.load(str(p).replace("qfail_", "qfailrows_")) for p in parts]))
    (out / "meta.json").write_text(json.dumps({"encoder": encoder.name, "variant": variant, "n": n, "grid": G,
                                               "dim": D, "failure_modes": list(FAILURE_MODES)}))
    return out


@dataclass
class FeatureStore:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.index = pd.read_csv(self.root / "index.csv")
        self.meta = json.loads((self.root / "meta.json").read_text())

    def array(self, name: str, mmap: bool = True):
        return np.load(self.root / f"{name}.npy", mmap_mode="r" if mmap else None)

    def boxes(self, rows) -> list:
        return [[tuple(b) for b in json.loads(self.index.boxes.iloc[i])] for i in rows]
