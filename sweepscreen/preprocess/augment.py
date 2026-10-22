"""Degradations that mimic low-cost handheld probes and imperfect scanning.

Each degradation is ``f(gray_uint8, boxes, level, rng) -> (gray_uint8, boxes)``. ``LEVELS`` fixes the
graded severities used in training augmentation and in the robustness evaluation.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def speckle(g, boxes, sigma, rng):
    """Multiplicative Rayleigh-distributed speckle with scale ``sigma`` (mean-preserving)."""
    noise = rng.rayleigh(scale=1.0, size=g.shape) / np.sqrt(np.pi / 2)
    out = g.astype(np.float32) * (1 + sigma * (noise - 1))
    return np.clip(out, 0, 255).astype(np.uint8), boxes


def blur(g, boxes, sigma, rng):
    return cv2.GaussianBlur(g, (0, 0), sigmaX=float(sigma)), boxes


def downsample(g, boxes, factor, rng):
    """Down-sample by ``factor`` and back (loss of lateral and axial resolution)."""
    h, w = g.shape
    small = cv2.resize(g, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), boxes


def jpeg(g, boxes, quality, rng):
    ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE), boxes


def gain(g, boxes, factor, rng):
    """Overall gain change (``factor`` > 1 brighter, < 1 darker) with clipping."""
    return np.clip(g.astype(np.float32) * factor, 0, 255).astype(np.uint8), boxes


def depth_crop(g, boxes, fraction, rng):
    """Remove the deepest ``fraction`` of the image (shallow depth setting); boxes are clipped."""
    h = g.shape[0]
    keep = max(1, int(round(h * (1 - fraction))))
    out = []
    for x0, y0, x1, y1 in boxes:
        y1c = min(y1, keep)
        if y1c > y0:
            out.append((x0, y0, x1, y1c))
    return g[:keep].copy(), out


def contact_loss(g, boxes, fraction, rng):
    """Partial probe contact: a dark wedge covering ``fraction`` of the width from one side."""
    h, w = g.shape
    width = int(round(w * fraction))
    out = g.copy()
    if width > 0:
        side = rng.integers(2)
        ramp = np.linspace(0.05, 1.0, width, dtype=np.float32)[None, :]
        if side == 0:
            out[:, :width] = (out[:, :width] * ramp).astype(np.uint8)
        else:
            out[:, w - width:] = (out[:, w - width:] * ramp[:, ::-1]).astype(np.uint8)
    return out, boxes


DEGRADATIONS: Dict[str, Callable] = {
    "speckle": speckle, "blur": blur, "downsample": downsample, "jpeg": jpeg, "gain": gain,
    "depth_crop": depth_crop, "contact_loss": contact_loss,
}

#: Graded severities (mild → severe) used for augmentation and for the robustness evaluation.
LEVELS: Dict[str, List[float]] = {
    "speckle": [0.1, 0.2, 0.3],
    "blur": [1.0, 2.0, 3.0],
    "downsample": [0.5, 0.33, 0.25],
    "jpeg": [50, 30, 15],
    "gain": [0.6, 1.4, 1.8],
    "depth_crop": [0.15, 0.25, 0.4],
    "contact_loss": [0.15, 0.3, 0.5],
}


def make(name: str, level: float, seed: int = 0):
    """Degradation callable with fixed parameters, for :func:`sweepscreen.preprocess.prepare`."""
    fn = DEGRADATIONS[name]

    def apply(g, boxes):
        return fn(g, list(boxes), level, np.random.default_rng(seed))
    apply.__name__ = f"{name}@{level}"
    return apply


def random_augment(g, boxes, rng: np.random.Generator, p_each: float = 0.15):
    """Training augmentation: each degradation applied independently with probability ``p_each`` at a
    random graded level, then a horizontal flip with probability 0.5."""
    boxes = list(boxes)
    for name, fn in DEGRADATIONS.items():
        if rng.random() < p_each:
            g, boxes = fn(g, boxes, float(rng.choice(LEVELS[name])), rng)
    if rng.random() < 0.5:
        w = g.shape[1]
        g = g[:, ::-1].copy()
        boxes = [(w - x1, y0, w - x0, y1) for x0, y0, x1, y1 in boxes]
    return g, boxes


class RandomAugment:
    """:func:`random_augment` with a generator seeded from ``seed`` and the image content, so each image
    receives the same degradation regardless of processing order (and on resumed runs)."""

    def __init__(self, seed: int = 0, p_each: float = 0.3):
        self.seed, self.p_each = seed, p_each

    def __call__(self, g, boxes):
        import zlib

        rng = np.random.default_rng([self.seed, zlib.crc32(np.ascontiguousarray(g).tobytes())])
        return random_augment(g, boxes, rng, self.p_each)
