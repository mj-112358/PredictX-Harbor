"""Hand-crafted adequacy features of a cropped greyscale ultrasound frame (uint8)."""
from __future__ import annotations

import cv2
import numpy as np

FEATURE_NAMES = (
    "mean", "std", "p05", "p95", "entropy", "dark_fraction", "saturated_fraction", "laplacian_var_log",
    "contact_left", "contact_right", "contact_max", "signal_depth", "row_profile_std", "gradient_energy_log",
)


def _dark_run(profile: np.ndarray, threshold: float) -> float:
    """Length (as a fraction) of the leading run of entries below ``threshold``."""
    below = profile < threshold
    run = np.argmin(below) if not below.all() else len(below)
    return run / len(profile)


def frame_features(gray: np.ndarray) -> np.ndarray:
    """Feature vector in the order of :data:`FEATURE_NAMES`.

    * intensity statistics and histogram entropy (blank, under- or over-gained frames);
    * Laplacian variance and gradient energy (blur, lack of speckle texture);
    * ``contact_left/right``: width of dark bands at the lateral edges, the signature of a probe face
      only partly in contact with the skin;
    * ``signal_depth``: fraction of rows (from the top) before the row-mean intensity falls to the
      noise floor; very small values mean no penetration.
    """
    g = gray.astype(np.float32)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1)
    entropy = -(p[p > 0] * np.log2(p[p > 0])).sum()
    lap = cv2.Laplacian(gray, cv2.CV_32F).var()
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    col = g.mean(axis=0)
    row = g.mean(axis=1)
    floor = max(8.0, 0.25 * float(np.percentile(col, 90)))   # relative to the well-coupled columns
    left, right = _dark_run(col, floor), _dark_run(col[::-1], floor)
    above = np.flatnonzero(row >= floor)
    depth = (above.max() + 1) / len(row) if len(above) else 0.0
    return np.array([
        g.mean(), g.std(), np.percentile(g, 5), np.percentile(g, 95), entropy,
        (gray < 10).mean(), (gray > 245).mean(), np.log1p(lap),
        left, right, max(left, right), depth, row.std(), np.log1p(float(np.mean(gx ** 2 + gy ** 2))),
    ], dtype=np.float32)
