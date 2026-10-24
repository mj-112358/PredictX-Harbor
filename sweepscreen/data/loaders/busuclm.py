"""BUS-UCLM (Vallez et al., Sci Data 2025; Hospital General de Ciudad Real, Spain).

Layout: ``BUS-UCLM/images/<PATIENT>_<n>.png``, RGB masks ``BUS-UCLM/masks/<same name>`` (green =
benign lesion, red = malignant lesion, black = background), and ``INFO.csv`` with columns Image,
Resolution, Label (Normal/Benign/Malignant), Doppler, Marks and Combined. The four-letter prefix of
the file name is the patient code (38 patients). Doppler and combined-mode images are excluded
because the pipeline reads B-mode only. Acquired on a Siemens ACUSON S2000 (dataset-level).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from ..schema import Record, boxes_from_mask
from .common import LayoutError, find_column, find_table, images_under, label_from_text, prepare_root, read_mask

SITE = "BUS-UCLM"
DEVICE = "Siemens ACUSON S2000 (dataset-level)"


def _yes(v) -> bool:
    return str(v).strip().lower() in ("yes", "y", "true", "1")


def load(raw_dir: str, extracted_dir: str) -> List[Record]:
    root = prepare_root(raw_dir, extracted_dir)
    table = find_table([root, Path(raw_dir)], ["Image", "Label"])
    c_img, c_lab = find_column(table, "Image"), find_column(table, "Label")
    c_dop = find_column(table, "Doppler", required=False)
    c_comb = find_column(table, "Combined", required=False)
    c_marks = find_column(table, "Marks", required=False)
    images, masks = {}, {}
    for p in images_under(root):
        (masks if any(x.lower().startswith("mask") for x in p.parts) else images)[p.name] = p
    records: List[Record] = []
    for row in table.to_dict("records"):
        name = str(row[c_img]).strip()
        if (c_dop and _yes(row[c_dop])) or (c_comb and _yes(row[c_comb])):
            continue
        img = images.get(name)
        if img is None:
            raise LayoutError(f"BUS-UCLM: image {name!r} listed in INFO.csv not found")
        label = label_from_text(row[c_lab])
        if label is None:
            raise LayoutError(f"BUS-UCLM: unknown label {row[c_lab]!r}")
        mask = masks.get(name)
        boxes = boxes_from_mask(read_mask(mask)) if (mask is not None and label != "normal") else []
        patient = name.split("_")[0]
        records.append(Record(uid=f"{SITE}:{img.stem}", site=SITE, image_path=str(img), label=label,
                              patient_id=f"{SITE}:{patient}", boxes=boxes,
                              mask_paths=[str(mask)] if (mask is not None and label != "normal") else [],
                              device=DEVICE, meta={"marks": bool(c_marks and _yes(row[c_marks]))}))
    if not records:
        raise LayoutError(f"BUS-UCLM: no usable images under {root}")
    return records
