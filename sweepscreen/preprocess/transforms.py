"""Intensity normalisation, aspect-preserving resizing and the full preprocessing chain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .crop import Crop, crop_to_content, remove_annotations

Box = Tuple[int, int, int, int]


def normalize_intensity(gray: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile window [low, high] mapped to [0, 1] (float32). Robust to gain differences between scanners."""
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, [low, high])
    if hi - lo < 1e-3:
        return np.zeros_like(g)
    return np.clip((g - np.float32(lo)) / np.float32(hi - lo), 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int
    size: int

    def boxes(self, boxes: List[Box]) -> List[Box]:
        out = []
        for x0, y0, x1, y1 in boxes:
            b = (int(round(x0 * self.scale)) + self.pad_x, int(round(y0 * self.scale)) + self.pad_y,
                 int(round(x1 * self.scale)) + self.pad_x, int(round(y1 * self.scale)) + self.pad_y)
            if b[2] > b[0] and b[3] > b[1]:
                out.append(b)
        return out

    def invert_box(self, box: Box) -> Box:
        x0, y0, x1, y1 = box
        s = self.scale
        return (int((x0 - self.pad_x) / s), int((y0 - self.pad_y) / s), int((x1 - self.pad_x) / s),
                int((y1 - self.pad_y) / s))


def letterbox(img: np.ndarray, size: int = 224, pad_value: float = 0.0) -> Tuple[np.ndarray, Letterbox]:
    """Resize the longer side to ``size`` (area interpolation) and pad to a ``size`` x ``size`` square."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((size, size) + img.shape[2:], pad_value, dtype=resized.dtype)
    py, px = (size - nh) // 2, (size - nw) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, Letterbox(s, px, py, size)


@dataclass
class Prepared:
    image: np.ndarray        # (size, size) float32 in [0, 1]
    boxes: List[Box]         # in prepared coordinates
    crop: Crop
    letterbox: Letterbox


def prepare(raw: np.ndarray, boxes: List[Box] = (), size: int = 224, degrade=None) -> Prepared:
    """Full chain: annotation removal → field-of-view crop → optional degradation → normalisation → letterbox.

    ``raw`` is the decoded image (greyscale or BGR, uint8). ``degrade`` is an optional callable applied
    to the cropped uint8 greyscale image (used for robustness tests and augmentation).
    """
    gray = remove_annotations(raw)
    crop = crop_to_content(gray)
    g = crop.apply(gray)
    b = crop.boxes(list(boxes))
    if degrade is not None:
        g, b = degrade(g, b)
    img, lb = letterbox(normalize_intensity(g), size)
    return Prepared(img, lb.boxes(b), crop, lb)
