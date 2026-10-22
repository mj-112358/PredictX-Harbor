"""Unified record schema shared by every dataset loader.

One :class:`Record` is one image or one video frame. Boxes are ``(x0, y0, x1, y1)`` in pixel
coordinates of the original image, half-open on the right/bottom. ``label`` is the image-level
class: ``"normal"`` (no lesion), ``"benign"`` or ``"malignant"``. A frame of a lesion video that does
not show the lesion keeps the video's label but has ``boxes == []`` and ``lesion_present == False``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LABELS = ("normal", "benign", "malignant")
Box = Tuple[int, int, int, int]


@dataclass
class Record:
    uid: str                                  # "<site>:<local id>", unique across all sites
    site: str                                 # dataset / acquisition site name
    image_path: str                           # absolute path of the image or frame
    label: str                                # one of LABELS
    patient_id: str                           # "<site>:<patient>" (falls back to the image id)
    boxes: List[Box] = field(default_factory=list)
    mask_paths: List[str] = field(default_factory=list)
    lesion_id: Optional[str] = None
    video_id: Optional[str] = None
    frame_index: Optional[int] = None
    device: Optional[str] = None
    age: Optional[float] = None
    birads: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if self.label not in LABELS:
            raise ValueError(f"{self.uid}: label {self.label!r} not in {LABELS}")
        self.boxes = [tuple(int(v) for v in b) for b in self.boxes]
        for x0, y0, x1, y1 in self.boxes:
            if not (x1 > x0 and y1 > y0):
                raise ValueError(f"{self.uid}: degenerate box {(x0, y0, x1, y1)}")

    @property
    def lesion_present(self) -> bool:
        return bool(self.boxes) or (self.label != "normal" and bool(self.mask_paths))

    @property
    def malignant(self) -> bool:
        return self.label == "malignant"


def to_frame(records: Sequence[Record]) -> pd.DataFrame:
    """Records as a DataFrame; list and dict fields are JSON-encoded so the table can be saved as CSV."""
    rows = []
    for r in records:
        d = asdict(r)
        d["boxes"] = json.dumps(d["boxes"])
        d["mask_paths"] = json.dumps(d["mask_paths"])
        d["meta"] = json.dumps(d["meta"], default=str)
        d["lesion_present"] = r.lesion_present
        rows.append(d)
    return pd.DataFrame(rows)


def from_frame(frame: pd.DataFrame) -> List[Record]:
    """Inverse of :func:`to_frame`."""
    out = []
    for d in frame.to_dict("records"):
        d = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}
        d.pop("lesion_present", None)
        for k in ("group_id", "dup_cluster"):
            d.pop(k, None)
        d["boxes"] = [tuple(b) for b in json.loads(d["boxes"])]
        d["mask_paths"] = json.loads(d["mask_paths"])
        d["meta"] = json.loads(d["meta"])
        if d.get("frame_index") is not None:
            d["frame_index"] = int(d["frame_index"])
        out.append(Record(**d))
    return out


def boxes_from_mask(mask: np.ndarray, min_area: int = 16) -> List[Box]:
    """One box per 8-connected foreground component of a binary mask (components < ``min_area`` px dropped)."""
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary.max(axis=2)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes
