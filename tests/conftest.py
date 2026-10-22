"""Test fixtures: tiny synthetic files laid out like each public dataset.

These exist only to exercise loaders and pipeline code paths. They are generated procedurally
(speckled background, a dark elliptical "lesion", burned-in UI text) and are never used for any
reported result.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


def synthetic_frame(rng, h=200, w=260, lesion=True, ui=True, colour=False, center=None, radius=(30, 18)):
    """A speckled B-mode-like frame inside a dark border, with an optional hypoechoic ellipse and UI text."""
    img = np.zeros((h, w), np.uint8)
    fov = (rng.rayleigh(1.0, (h - 40, w - 40)) * 60 + 40).clip(0, 255).astype(np.uint8)
    img[20:h - 20, 20:w - 20] = cv2.GaussianBlur(fov, (3, 3), 0)
    mask = np.zeros((h, w), np.uint8)
    if lesion:
        cx, cy = center or (int(rng.integers(70, w - 70)), int(rng.integers(60, h - 60)))
        cv2.ellipse(mask, (cx, cy), radius, 0, 0, 360, 1, -1)
        img[mask > 0] = (img[mask > 0] * 0.25).astype(np.uint8)
    if ui:
        cv2.putText(img, "GE 12L", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
    if colour:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.line(img, (60, 60), (90, 60), (0, 255, 255), 1)
    return img, mask


def _write(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def busi_tree(tmp_path, rng):
    root = tmp_path / "busi" / "Dataset_BUSI_with_GT"
    for label, n in (("benign", 3), ("malignant", 2), ("normal", 2)):
        for i in range(1, n + 1):
            img, mask = synthetic_frame(rng, lesion=label != "normal")
            _write(root / label / f"{label} ({i}).png", img)
            _write(root / label / f"{label} ({i})_mask.png", mask * 255)
    # second-lesion mask on one benign image
    _, m2 = synthetic_frame(rng, center=(200, 150), radius=(12, 10))
    _write(root / "benign" / "benign (1)_mask_1.png", m2 * 255)
    return tmp_path / "busi"


@pytest.fixture
def busbra_tree(tmp_path, rng):
    root = tmp_path / "busbra" / "BUSBRA"
    rows = []
    for i in range(1, 7):
        side = "l" if i % 2 else "r"
        img, mask = synthetic_frame(rng)
        _write(root / "Images" / f"bus_{i:04d}-{side}.png", img)
        _write(root / "Masks" / f"mask_{i:04d}-{side}.png", mask * 255)
        rows.append({"ID": f"bus_{i:04d}-{side}", "Case": (i + 1) // 2, "Histology": "fibroadenoma",
                     "Pathology": "malignant" if i > 4 else "benign", "BIRADS": 3 + (i > 4), "Device": "GE Logiq 7",
                     "Side": side})
    pd.DataFrame(rows).to_csv(root / "bus_data.csv", index=False)
    return tmp_path / "busbra"


@pytest.fixture
def breast_tree(tmp_path, rng):
    root = tmp_path / "breast"
    rows = []
    for i in range(1, 5):
        cls = ["benign", "malignant", "benign", "normal"][i - 1]
        img, mask = synthetic_frame(rng, lesion=cls != "normal")
        _write(root / "images" / f"case{i:03d}.png", img)
        mname = f"case{i:03d}_tumor.png" if cls != "normal" else ""
        if mname:
            _write(root / "images" / mname, mask * 255)
        rows.append({"CaseID": i, "Image_filename": f"case{i:03d}.png", "Mask_tumor_filename": mname,
                     "Age": 40 + i, "Tissue_composition": "heterogeneous", "BIRADS": "4a",
                     "Classification": cls, "Verification": "biopsy"})
    pd.DataFrame(rows).to_excel(root / "BrEaST-Lesions-USG-clinical-data.xlsx", index=False)
    return root


@pytest.fixture
def udiat_tree(tmp_path, rng):
    root = tmp_path / "udiat" / "BUS_Dataset_B"
    rows = []
    for i in range(1, 5):
        img, mask = synthetic_frame(rng)
        _write(root / "original" / f"{i:06d}.png", img)
        _write(root / "GT" / f"{i:06d}.png", mask * 255)
        rows.append({"Image": f"{i:06d}", "Type": "Malignant" if i == 4 else "Benign"})
    pd.DataFrame(rows).to_excel(root / "DatasetB.xlsx", index=False)
    return tmp_path / "udiat"


@pytest.fixture
def breastvid_tree(tmp_path, rng):
    root = tmp_path / "breastvid" / "miccai_buv"
    images, anns, k = [], [], 0
    videos = [("benign", "vidA", 8), ("malignant", "vidB", 8), ("benign", "x66ef02e7f1b9a0ef", 3)]
    for cls, vid, n in videos:
        for f in range(n):
            visible = 1 <= f <= n - 2
            cx = 100 + 3 * f
            img, mask = synthetic_frame(rng, lesion=visible, center=(cx, 100))
            rel = f"rawframes/{cls}/{vid}/{f:05d}.png"
            _write(root / rel, img)
            k += 1
            images.append({"id": k, "file_name": rel, "width": img.shape[1], "height": img.shape[0]})
            if visible:
                ys, xs = np.nonzero(mask)
                anns.append({"id": k, "image_id": k, "category_id": 1,
                             "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]})
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "lesion"}]}
    (root / "train.json").write_text(json.dumps(coco))
    (root / "val.json").write_text(json.dumps({"images": [], "annotations": [], "categories": coco["categories"]}))
    return tmp_path / "breastvid"
