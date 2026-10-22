"""UDIAT Dataset B (Yap et al., 2018; UDIAT Diagnostic Centre, Sabadell).

Layout: ``original/*.png`` images, ``GT/*.png`` masks with the same stem, and a spreadsheet mapping
image name to lesion type (benign / malignant). Collected on a Siemens ACUSON Sequoia C512 with a
17L5 HD linear probe (dataset-level device).
"""
from __future__ import annotations

from typing import List

from ..schema import Record, boxes_from_mask
from .common import LayoutError, find_column, find_table, images_under, label_from_text, norm, prepare_root, read_mask

SITE = "UDIAT"
DEVICE = "Siemens ACUSON Sequoia C512, 17L5 HD (dataset-level)"


def _key(name) -> str:
    """Matching key of an image name: normalised stem without extension; purely numeric names lose
    leading zeros, because spreadsheets store ``000001`` as the number 1."""
    s = str(name)
    if isinstance(name, float) and name.is_integer():
        s = str(int(name))
    stem = s.rsplit(".", 1)[0] if "." in s and not s.replace(".", "").isdigit() else s
    k = norm(stem)
    return (k.lstrip("0") or "0") if k.isdigit() else k


def load(raw_dir: str, extracted_dir: str) -> List[Record]:
    root = prepare_root(raw_dir, extracted_dir)
    table = find_table(root, ["Type"])
    c_type = find_column(table, "Type")
    c_name = find_column(table, "Image", "Name", "File", "ID")
    originals, gts = {}, {}
    for p in images_under(root):
        parts = {norm(x) for x in p.parts}
        if "gt" in parts or "groundtruth" in parts or "mask" in norm(p.stem):
            gts[_key(norm(p.stem).replace("mask", ""))] = p
        else:
            originals[_key(p.stem)] = p
    records: List[Record] = []
    for row in table.to_dict("records"):
        key = _key(row[c_name])
        img = originals.get(key)
        if img is None:
            raise LayoutError(f"UDIAT: image {row[c_name]!r} not found among {len(originals)} originals")
        label = label_from_text(row[c_type])
        if label is None:
            raise LayoutError(f"UDIAT: unknown type {row[c_type]!r}")
        mask = gts.get(key)
        boxes = boxes_from_mask(read_mask(mask)) if mask else []
        records.append(Record(uid=f"{SITE}:{img.stem}", site=SITE, image_path=str(img), label=label,
                              patient_id=f"{SITE}:{img.stem}", boxes=boxes,
                              mask_paths=[str(mask)] if mask else [], device=DEVICE))
    return records
