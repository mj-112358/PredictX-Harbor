"""BUSI (Al-Dhabyani et al., 2020; Baheya Hospital, Cairo), curated release (Aumente-Maestro et al., 2025).

Curated layout (used): ``Curated BUSI/images/<class>_id_<n>.png`` with masks
``Curated BUSI/masks/<class>_id_<n>_mask.png`` (``_mask_1`` for a second lesion); 222 benign, 164
malignant, 64 normal images after the curators removed duplicates, axillary images and label
inconsistencies. The original layout (``Dataset_BUSI_with_GT/<class>/<class> (N).png``) is also
accepted. BUSI has no patient identifiers, so each image is its own patient; near-duplicate
clustering in :mod:`sweepscreen.data.dedup` groups repeated views.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from ..schema import Record, boxes_from_mask
from .common import LayoutError, images_under, prepare_root, read_mask

SITE = "BUSI"
DEVICE = "GE LOGIQ E9 / E9 Agile (dataset-level)"
_MASK = re.compile(r"_mask(_\d+)?$", re.IGNORECASE)
_CLASS = re.compile(r"^(benign|malignant|normal)", re.IGNORECASE)


def _label(path: Path) -> str:
    for part in [path.stem, *reversed(path.parts[:-1])]:
        m = _CLASS.match(part)
        if m:
            return m.group(1).lower()
    raise LayoutError(f"BUSI: cannot tell the class of {path}")


def load(raw_dir: str, extracted_dir: str) -> List[Record]:
    root = prepare_root(raw_dir, extracted_dir)
    files = images_under(root)
    masks: Dict[str, list] = {}
    for p in files:
        if _MASK.search(p.stem):
            masks.setdefault(_MASK.sub("", p.stem), []).append(p)
    records: List[Record] = []
    for p in files:
        if _MASK.search(p.stem):
            continue
        label = _label(p)
        mask_paths = sorted(masks.get(p.stem, []))
        boxes = []
        if label != "normal":
            for m in mask_paths:
                boxes += boxes_from_mask(read_mask(m))
        local = re.sub(r"\W+", "_", p.stem).strip("_")
        records.append(Record(uid=f"{SITE}:{local}", site=SITE, image_path=str(p), label=label,
                              patient_id=f"{SITE}:{local}", boxes=boxes,
                              mask_paths=[str(m) for m in mask_paths] if label != "normal" else [], device=DEVICE))
    if not records:
        raise LayoutError(f"BUSI: no images under {root}")
    return records
