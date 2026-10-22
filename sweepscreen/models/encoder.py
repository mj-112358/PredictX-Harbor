"""Frozen DINOv2 encoder.

A preprocessed 224 x 224 greyscale frame is replicated to three channels, normalised with the
ImageNet statistics DINOv2 was trained with, and passed through ViT-S/14. The encoder returns the
CLS token (global descriptor, D = 384) and the 16 x 16 grid of patch tokens (one per 14 x 14-pixel
patch), which the detector reads. Weights are loaded from a local folder only; no network access.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "pretrained" / "dinov2-small"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Encoder:
    """``Encoder()(images)`` -> ``(cls, patches)`` with shapes ``(N, D)`` and ``(N, G, G, D)``."""

    def __init__(self, weights: Optional[str] = None, config=None, image_size: int = 224, threads: int = 4,
                 batch_size: int = 32):
        from transformers import Dinov2Config, Dinov2Model

        torch.set_num_threads(threads)
        if config is not None:            # randomly initialised model (tests only)
            torch.manual_seed(0)
            self.model = Dinov2Model(config)
            self.name = "dinov2-random-test"
        else:
            path = Path(weights or DEFAULT_WEIGHTS)
            if not (path / "config.json").exists():
                raise FileNotFoundError(f"DINOv2 weights not found in {path}; see docs/data_request.md")
            self.model = Dinov2Model.from_pretrained(str(path), local_files_only=True)
            self.name = path.name
        self.model.eval()
        self.image_size, self.batch_size = image_size, batch_size
        self.patch = self.model.config.patch_size
        self.grid = image_size // self.patch
        self.dim = self.model.config.hidden_size

    @staticmethod
    def tiny_config():
        from transformers import Dinov2Config

        return Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                            patch_size=14, image_size=224)

    def _tensor(self, images: np.ndarray) -> torch.Tensor:
        x = np.repeat(np.asarray(images, np.float32)[:, None], 3, axis=1)
        x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
        return torch.from_numpy(x)

    @torch.inference_mode()
    def __call__(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        images = np.asarray(images)
        if images.ndim == 2:
            images = images[None]
        cls_out, patch_out = [], []
        for s in range(0, len(images), self.batch_size):
            h = self.model(pixel_values=self._tensor(images[s:s + self.batch_size])).last_hidden_state
            cls_out.append(h[:, 0].numpy())
            patch_out.append(h[:, 1:].reshape(len(h), self.grid, self.grid, self.dim).numpy())
        return np.concatenate(cls_out), np.concatenate(patch_out)
