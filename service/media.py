"""Decoding uploads and rendering result images."""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    return b.copy() if np.array_equal(b, g) and np.array_equal(g, r) else img


def decode_video(data: bytes, suffix: str, max_fps: float = 15.0, max_frames: int = 900) -> List[np.ndarray]:
    """Frames of a video upload, temporally subsampled to at most ``max_fps`` frames per second."""
    with tempfile.NamedTemporaryFile(suffix=suffix) as fh:
        fh.write(data)
        fh.flush()
        cap = cv2.VideoCapture(fh.name)
        if not cap.isOpened():
            raise ValueError("could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or max_fps
        step = max(1, int(round(fps / max_fps)))
        frames, k = [], 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if k % step == 0:
                frames.append(frame)
            k += 1
        cap.release()
    if not frames:
        raise ValueError("video contains no frames")
    return frames


def decode_upload(name: str, data: bytes) -> List[np.ndarray]:
    suffix = Path(name or "").suffix.lower()
    if suffix in VIDEO_TYPES:
        return decode_video(data, suffix)
    if suffix in IMAGE_TYPES or not suffix:
        return [decode_image(data)]
    raise ValueError(f"unsupported file type {suffix!r}")


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def draw_boxes(frame: np.ndarray, boxes: Sequence[Sequence[int]], scores: Sequence[float]) -> np.ndarray:
    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    for (x0, y0, x1, y1), s in zip(boxes, scores):
        cv2.rectangle(out, (x0, y0), (x1, y1), (52, 104, 235), 2)
        cv2.putText(out, f"{s:.2f}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 104, 235), 1)
    return out


def heatmap_overlay(prepared: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Prepared 224 x 224 frame (float [0, 1]) with the lesion heatmap blended in."""
    base = cv2.cvtColor((prepared * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    colour = cv2.applyColorMap((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = np.clip(heatmap, 0, 1)[..., None] * 0.6
    return (base * (1 - alpha) + colour * alpha).astype(np.uint8)
