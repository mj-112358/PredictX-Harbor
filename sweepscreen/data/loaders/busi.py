"""BUSI (Al-Dhabyani et al., 2020; Baheya Hospital, Cairo).

Layout: ``Dataset_BUSI_with_GT/{benign,malignant,normal}/<class> (N).png`` with masks
``<class> (N)_mask.png`` and, for a second lesion, ``<class> (N)_mask_1.png``. The dataset has no
patient identifiers, so each image is its own patient; near-duplicate clustering in
:mod:`sweepscreen.data.dedup` groups repeated views of the same lesion.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from ..schema import Record, boxes_from_mask
from .common import LayoutError, images_under, prepare_root, read_mask

SITE = "BUSI"
DEVICE = "GE LOGIQ E9 / E9 Agile (dataset-level)"
_MASK = re.compile(r"_mask(_\d+)?$", re.IGNORECASE)


def load(raw_dir: str, extracted_dir: str) -> List[Record]:
    root = prepare_root(raw_dir, extracted_dir)
    records: List[Record] = []
    for label in ("benign", "malignant", "normal"):
        dirs = [d for d in root.rglob(label) if d.is_dir()]
        if not dirs:
            raise LayoutError(f"BUSI: no '{label}' folder under {root}")
        files = images_under(dirs[0])
        masks = {}
        for p in files:
            if _MASK.search(p.stem):
                masks.setdefault(_MASK.sub("", p.stem), []).append(p)
        for p in files:
            if _MASK.search(p.stem):
                continue
            mask_paths = sorted(masks.get(p.stem, []))
            boxes = []
            if label != "normal":
                for m in mask_paths:
                    boxes += boxes_from_mask(read_mask(m))
            local = re.sub(r"\W+", "_", p.stem).strip("_")
            records.append(Record(uid=f"{SITE}:{local}", site=SITE, image_path=str(p), label=label,
                                  patient_id=f"{SITE}:{local}", boxes=boxes,
                                  mask_paths=[str(m) for m in mask_paths], device=DEVICE))
    if not records:
        raise LayoutError(f"BUSI: no images under {root}")
    return records
