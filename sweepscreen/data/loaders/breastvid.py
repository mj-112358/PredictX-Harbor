"""BreastVid, the CVA-Net "miccai_buv" video dataset (Lin et al., MICCAI 2022).

Layout: ``rawframes/{benign,malignant}/<video_id>/<frame>.png`` and COCO-style ``train.json`` /
``val.json`` with per-frame lesion boxes ``[x, y, w, h]``. Frames listed without an annotation are
lesion-free frames of a lesion video. The corrections published in the CVA-Net README (2023-03-25)
are applied here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from ..schema import Record
from .common import LayoutError, images_under, prepare_root

SITE = "BreastVid"
DROP = {"benign/x66ef02e7f1b9a0ef"}
RELABEL_MALIGNANT = {"benign/x63c9ba1377f35bf6", "benign/x5a1c46ec6377e946"}
SAME_PATIENT = {"benign/x63c9ba1377f35bf6", "benign/x5a1c46ec6377e946", "malignant/2390fbea047347b",
                "malignant/7a39ab5d4970bf89"}


def _video_key(path: str) -> str:
    """'<class>/<video_id>' from any path containing '.../<class>/<video_id>/<frame>'."""
    parts = Path(path.replace("\\", "/")).parts
    for i, part in enumerate(parts[:-2]):
        if part.lower() in ("benign", "malignant"):
            return f"{part.lower()}/{parts[i + 1]}"
    raise LayoutError(f"BreastVid: cannot find class/video in {path!r}")


def _frame_index(name: str) -> int:
    digits = re.findall(r"\d+", Path(name).stem)
    return int(digits[-1]) if digits else 0


def load(raw_dir: str, extracted_dir: str) -> List[Record]:
    root = prepare_root(raw_dir, extracted_dir)
    jsons = [p for p in root.rglob("*.json") if p.stem in ("train", "val", "test")]
    if not jsons:
        raise LayoutError(f"BreastVid: no train.json/val.json under {root}")
    frames = {str(p.relative_to(root)).replace("\\", "/"): p for p in images_under(root)}
    by_suffix: Dict[str, Path] = {}
    for rel, p in frames.items():
        key = _video_key(rel) + "/" + p.name
        by_suffix[key] = p

    boxes: Dict[str, list] = {}
    split_of: Dict[str, str] = {}
    for js in jsons:
        coco = json.loads(js.read_text())
        id_to_key = {}
        for im in coco["images"]:
            key = _video_key(im["file_name"]) + "/" + Path(im["file_name"]).name
            id_to_key[im["id"]] = key
            boxes.setdefault(key, [])
            split_of[key] = js.stem
        for an in coco.get("annotations", []):
            x, y, w, h = an["bbox"]
            if w > 0 and h > 0:
                boxes[id_to_key[an["image_id"]]].append((round(x), round(y), round(x + w), round(y + h)))

    records: List[Record] = []
    for key, bxs in sorted(boxes.items()):
        video = key.rsplit("/", 1)[0]
        if video in DROP:
            continue
        path = by_suffix.get(key)
        if path is None:
            raise LayoutError(f"BreastVid: annotated frame {key} not found on disk")
        label = "malignant" if video in RELABEL_MALIGNANT else video.split("/")[0]
        patient = "corrected_group_1" if video in SAME_PATIENT else video.replace("/", "_")
        records.append(Record(uid=f"{SITE}:{key}", site=SITE, image_path=str(path), label=label,
                              patient_id=f"{SITE}:{patient}", boxes=bxs, lesion_id=f"{SITE}:{video}",
                              video_id=f"{SITE}:{video}", frame_index=_frame_index(path.name),
                              meta={"official_split": split_of[key]}))
    return records
