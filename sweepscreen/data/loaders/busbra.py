"""BUS-BRA (Gómez-Flores et al., 2024; National Cancer Institute, Rio de Janeiro).

Layout: a case table (CSV) with one row per image (columns include an image ID, a case/patient ID,
pathology, histology, BI-RADS, device and side), an image folder and a mask folder. Images are
matched to rows by file stem; masks by a stem that contains the image's ID and the token "mask".
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from ..schema import Record, boxes_from_mask
from .common import LayoutError, find_column, find_table, images_under, label_from_text, norm, prepare_root, read_mask

SITE = "BUS-BRA"


def _key(stem: str) -> str:
    """Matching key of an image or mask stem: the alphanumeric ID with any 'mask'/'bus' prefix removed."""
    s = norm(stem)
    return re.sub(r"^(mask|bus)+", "", s)


def load(raw_dir: str, extracted_dir: str) -> List[Record]:
    root = prepare_root(raw_dir, extracted_dir)
    table = find_table([root, Path(raw_dir)], ["ID", "Case", "Pathology", "BIRADS"])
    c_id, c_path = find_column(table, "ID"), find_column(table, "Pathology")
    c_case = find_column(table, "Case", "Patient", required=False)
    c_birads = find_column(table, "BIRADS", "BI-RADS", required=False)
    c_dev = find_column(table, "Device", required=False)
    c_hist = find_column(table, "Histology", required=False)
    c_side = find_column(table, "Side", required=False)

    images: Dict[str, Path] = {}
    masks: Dict[str, Path] = {}
    for p in images_under(root):
        (masks if "mask" in p.stem.lower() else images)[_key(p.stem)] = p
    records: List[Record] = []
    missing = []
    for row in table.to_dict("records"):
        key = _key(str(row[c_id]))
        if key not in images:
            missing.append(row[c_id])
            continue
        label = label_from_text(row[c_path])
        if label is None:
            raise LayoutError(f"BUS-BRA: unknown pathology {row[c_path]!r}")
        mask = masks.get(key)
        boxes = boxes_from_mask(read_mask(mask)) if mask else []
        case = str(row[c_case]) if c_case else str(row[c_id])
        records.append(Record(
            uid=f"{SITE}:{row[c_id]}", site=SITE, image_path=str(images[key]), label=label,
            patient_id=f"{SITE}:{case}", boxes=boxes, mask_paths=[str(mask)] if mask else [],
            lesion_id=f"{SITE}:{case}:{row[c_side]}" if c_side else None,
            device=str(row[c_dev]) if c_dev else None,
            birads=str(row[c_birads]) if c_birads else None,
            meta={k: row[c] for k, c in (("histology", c_hist), ("side", c_side)) if c}))
    if missing and len(missing) > 0.05 * len(table):
        raise LayoutError(f"BUS-BRA: {len(missing)} of {len(table)} table rows have no image file (e.g. {missing[:3]})")
    return records
