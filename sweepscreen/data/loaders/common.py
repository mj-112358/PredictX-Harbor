"""Helpers shared by the dataset loaders."""
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class LayoutError(RuntimeError):
    """The files on disk do not match the layout a loader expects."""


def prepare_root(raw_dir: str | os.PathLike, extracted_dir: str | os.PathLike) -> Path:
    """Return a directory with the dataset unpacked.

    Every ``*.zip`` under ``raw_dir`` is extracted once into ``extracted_dir`` (skipped when a
    ``.extracted`` marker exists). If ``raw_dir`` holds no zips it is returned unchanged.
    """
    raw_dir, extracted_dir = Path(raw_dir), Path(extracted_dir)
    if not raw_dir.exists():
        raise LayoutError(f"{raw_dir} does not exist; see docs/data_request.md")
    zips = sorted(raw_dir.rglob("*.zip"))
    if not zips:
        return raw_dir
    marker = extracted_dir / ".extracted"
    if not marker.exists():
        extracted_dir.mkdir(parents=True, exist_ok=True)
        for z in zips:
            with zipfile.ZipFile(z) as fh:
                for member in fh.namelist():  # refuse path traversal
                    target = (extracted_dir / member).resolve()
                    if not str(target).startswith(str(extracted_dir.resolve())):
                        raise LayoutError(f"unsafe path {member!r} in {z}")
                fh.extractall(extracted_dir)
        marker.write_text("\n".join(str(z) for z in zips))
    return extracted_dir


def images_under(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())


def norm(name: str) -> str:
    """Normalised column or file token: lower case, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(frame: pd.DataFrame, *candidates: str, required: bool = True) -> Optional[str]:
    """First column whose normalised name equals (or, failing that, contains) a candidate."""
    cols = {norm(c): c for c in frame.columns}
    for cand in candidates:
        if norm(cand) in cols:
            return cols[norm(cand)]
    for cand in candidates:
        for k, c in cols.items():
            if norm(cand) in k:
                return c
    if required:
        raise LayoutError(f"none of {candidates} found in columns {list(frame.columns)}")
    return None


def find_table(root, must_have: Iterable[str]) -> pd.DataFrame:
    """The first CSV/XLSX under ``root`` (a path or a list of paths) whose columns contain every name in ``must_have``."""
    must_have = list(must_have)
    roots = [Path(r) for r in (root if isinstance(root, (list, tuple)) else [root])]
    tried = []
    candidates = []
    for r in roots:
        candidates += sorted(list(r.rglob("*.csv")) + list(r.rglob("*.xlsx")) + list(r.rglob("*.xls")))
    for p in candidates:
        try:
            frame = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig") if p.suffix == ".csv" else pd.read_excel(p)
        except Exception as exc:  # pragma: no cover - unreadable side files
            tried.append(f"{p.name}: {exc}")
            continue
        if all(find_column(frame, m, required=False) for m in must_have):
            frame.attrs["path"] = str(p)
            return frame
        tried.append(f"{p.name}: columns {list(frame.columns)[:12]}")
    raise LayoutError(f"no table under {root} has columns {must_have}; tried: {tried}")


def read_gray(path: str | os.PathLike) -> np.ndarray:
    """8-bit greyscale image (handles non-ASCII paths)."""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise LayoutError(f"cannot decode image {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def read_raw(path: str | os.PathLike) -> np.ndarray:
    """Decoded 8-bit image, colour (BGR) kept when present so coloured annotations can be detected."""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise LayoutError(f"cannot decode image {path}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    if np.array_equal(b, g) and np.array_equal(g, r):
        return b.copy()
    return img


def read_mask(path: str | os.PathLike) -> np.ndarray:
    """Binary mask (any non-zero pixel, in any channel, is foreground)."""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise LayoutError(f"cannot decode mask {path}")
    if m.ndim == 3:
        m = m.max(axis=2)
    return (m > 0).astype(np.uint8)


def label_from_text(value: object) -> Optional[str]:
    """Map free-text class labels to the schema labels."""
    s = norm(value)
    if not s or s == "nan":
        return None
    if s.startswith("malig") or s in {"m", "1", "cancer"}:
        return "malignant"
    if s.startswith("benig") or s in {"b", "0"}:
        return "benign"
    if s.startswith("normal") or s in {"n", "none"}:
        return "normal"
    return None


def stem_index(paths: Iterable[Path]) -> Dict[str, Path]:
    """Map normalised file stem -> path."""
    return {norm(p.stem): p for p in paths}
