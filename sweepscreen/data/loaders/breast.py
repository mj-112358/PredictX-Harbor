"""BrEaST / Breast-Lesions-USG (Pawłowska et al., 2024; Poland).

Layout: PNG images and tumour masks, and a clinical spreadsheet with one row per image naming the
image file and mask file(s), with age, tissue composition, signs, symptoms, BI-RADS descriptors, the
BI-RADS category, verification method, diagnosis and classification (benign / malignant / normal).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from ..schema import Record, boxes_from_mask
from .common import LayoutError, find_column, find_table, images_under, label_from_text, norm, prepare_root, read_mask

SITE = "BrEaST"
_META = ("Tissue_composition", "Signs", "Symptoms", "Shape", "Margin", "Echogenicity", "Posterior_features",
         "Halo", "Calcifications", "Skin_thickening", "Verification", "Diagnosis", "Pixel_size")


def load(raw_dir: str, extracted_dir: str) -> List[Record]:
    root = prepare_root(raw_dir, extracted_dir)
    table = find_table([root, Path(raw_dir)], ["Image_filename", "Classification"])
    c_img, c_cls = find_column(table, "Image_filename"), find_column(table, "Classification")
    c_mask = find_column(table, "Mask_tumor_filename", "Mask_filename", required=False)
    c_case = find_column(table, "CaseID", "Case", required=False)
    c_age = find_column(table, "Age", required=False)
    c_birads = find_column(table, "BIRADS", required=False)
    meta_cols = {m: find_column(table, m, required=False) for m in _META}
    files = {norm(p.name): p for p in images_under(root)}

    records: List[Record] = []
    for row in table.to_dict("records"):
        img = files.get(norm(row[c_img]))
        if img is None:
            raise LayoutError(f"BrEaST: image {row[c_img]!r} listed in the table but not found")
        label = label_from_text(row[c_cls])
        if label is None:
            raise LayoutError(f"BrEaST: unknown classification {row[c_cls]!r}")
        mask_paths, boxes = [], []
        if c_mask and isinstance(row[c_mask], str):
            for name in row[c_mask].replace(";", ",").split(","):
                m = files.get(norm(name.strip()))
                if m is not None:
                    mask_paths.append(str(m))
                    boxes += boxes_from_mask(read_mask(m))
        case = str(row[c_case]) if c_case else img.stem
        age = row[c_age] if c_age else None
        records.append(Record(
            uid=f"{SITE}:{img.stem}", site=SITE, image_path=str(img), label=label, patient_id=f"{SITE}:{case}",
            boxes=boxes if label != "normal" else [], mask_paths=mask_paths,
            age=float(age) if age is not None and str(age).replace(".", "", 1).isdigit() else None,
            birads=str(row[c_birads]) if c_birads else None,
            meta={m: row[c] for m, c in meta_cols.items() if c is not None}))
    return records
