"""Removal of scanner user-interface regions and burned-in annotations.

Exported ultrasound images usually contain the B-mode field of view surrounded by a dark frame with
text (patient fields, depth scale, logos) and sometimes calipers or markers drawn over the image.

* :func:`crop_to_content` keeps the largest bright connected region (the field of view), found
  after suppressing thin text by morphological opening, and returns its bounding box.
* :func:`annotation_mask` marks pixels that are coloured (any annotation drawn in colour) or that
  belong to small, fully saturated components (white text, caliper crosses) inside the field of view.
* :func:`remove_annotations` inpaints those pixels from their surroundings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Crop:
    x0: int
    y0: int
    x1: int
    y1: int

    def apply(self, img: np.ndarray) -> np.ndarray:
        return img[self.y0:self.y1, self.x0:self.x1]

    def boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Shift and clip boxes into the cropped frame; boxes falling outside are dropped."""
        out = []
        w, h = self.x1 - self.x0, self.y1 - self.y0
        for x0, y0, x1, y1 in boxes:
            nx0, ny0 = max(0, x0 - self.x0), max(0, y0 - self.y0)
            nx1, ny1 = min(w, x1 - self.x0), min(h, y1 - self.y0)
            if nx1 > nx0 and ny1 > ny0:
                out.append((nx0, ny0, nx1, ny1))
        return out


def crop_to_content(gray: np.ndarray, threshold: int = 12, min_fraction: float = 0.2) -> Crop:
    """Bounding box of the ultrasound field of view.

    Pixels brighter than ``threshold`` are opened with a 15-px kernel (removes text strokes and thin
    scale bars), then closed to fill dark anechoic areas, and the largest component is kept. If that
    component covers less than ``min_fraction`` of the image, the full image is returned.
    """
    h, w = gray.shape[:2]
    binary = (gray > threshold).astype(np.uint8)
    k = max(3, int(round(min(h, w) * 0.03)) | 1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3 * k, 3 * k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return Crop(0, 0, w, h)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[i]
    if area < min_fraction * h * w:
        return Crop(0, 0, w, h)
    return Crop(int(x), int(y), int(x + bw), int(y + bh))


def annotation_mask(image: np.ndarray, chroma_threshold: int = 40, max_component_fraction: float = 0.002) -> np.ndarray:
    """Binary mask of burned-in annotations.

    ``image`` may be greyscale or BGR. Coloured pixels (max channel difference > ``chroma_threshold``)
    are annotations. Saturated pixels (≥ 250) are annotations when they form components smaller than
    ``max_component_fraction`` of the image (text, crosses, dotted calipers); large saturated areas are
    left alone as possible tissue.
    """
    if image.ndim == 3:
        bgr = image[..., :3].astype(np.int16)
        chroma = bgr.max(axis=2) - bgr.min(axis=2)
        mask = (chroma > chroma_threshold).astype(np.uint8)
        gray = cv2.cvtColor(image[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        mask = np.zeros(image.shape, np.uint8)
        gray = image
    sat = (gray >= 250).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sat, connectivity=8)
    limit = max_component_fraction * gray.size
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= limit
    mask |= small[labels].astype(np.uint8)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def remove_annotations(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Greyscale image with annotation pixels inpainted (Telea)."""
    if mask is None:
        mask = annotation_mask(image)
    gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if not mask.any():
        return gray.copy()
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
