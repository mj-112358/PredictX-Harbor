import numpy as np
import pytest

from sweepscreen.models.detector import PatchDetector, coverage_targets, frame_stats
from sweepscreen.models.encoder import Encoder
from sweepscreen.models.malignancy import MalignancyHead, lesion_features, roi_pool


def test_encoder_shapes_with_tiny_random_model():
    enc = Encoder(config=Encoder.tiny_config(), threads=1)
    cls, patches = enc(np.random.default_rng(0).random((3, 224, 224), dtype=np.float32))
    assert cls.shape == (3, 32) and patches.shape == (3, 16, 16, 32)


def test_encoder_requires_local_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        Encoder(weights=str(tmp_path))


def test_coverage_targets_box():
    t = coverage_targets([[(0, 0, 28, 14)]], None, size=224, grid=16)
    assert t.shape == (1, 16, 16)
    assert t[0, 0, 0] == 1 and t[0, 0, 1] == 1 and t[0, 1, 0] == 0


def test_frame_stats():
    p = np.zeros((1, 4, 4))
    p[0, 0, :4] = [0.9, 0.8, 0.7, 0.6]
    s = frame_stats(p)[0]
    assert np.allclose(s, [0.9, 0.75, 4 / 16])


def _toy_tokens(n, rng, grid=16, dim=8):
    """Tokens where lesion patches carry a shifted feature; returns tokens, targets, frame labels, groups."""
    tokens = rng.normal(0, 1, (n, grid, grid, dim)).astype(np.float32)
    targets = np.zeros((n, grid, grid), np.float32)
    labels = np.zeros(n, int)
    for i in range(n):
        if i % 2 == 0:
            r, c = rng.integers(2, grid - 5, 2)
            targets[i, r:r + 3, c:c + 3] = 1
            tokens[i, r:r + 3, c:c + 3, 0] += 3.0
            labels[i] = 1
    return tokens, targets, labels, np.arange(n) // 2


def test_patch_detector_learns_and_localises():
    rng = np.random.default_rng(0)
    tok, tgt, lab, grp = _toy_tokens(120, rng)
    det = PatchDetector(dim=8, epochs=40, hidden=16, seed=0).fit(tok, tgt, lab, grp)
    tok2, tgt2, lab2, _ = _toy_tokens(40, np.random.default_rng(1))
    p = det.frame_probability(tok2)
    assert p[lab2 == 1].mean() > p[lab2 == 0].mean() + 0.4
    d = det.detect(tok2[:1])[0]
    assert d.heatmap.shape == (224, 224) and d.boxes
    ys, xs = np.nonzero(tgt2[0])
    cx, cy = (xs.mean() + 0.5) * 14, (ys.mean() + 0.5) * 14
    x0, y0, x1, y1 = d.boxes[0]
    assert x0 <= cx <= x1 and y0 <= cy <= y1


def test_roi_pool_selects_cells():
    patches = np.zeros((16, 16, 2))
    patches[0, 0] = [1, 1]
    assert np.allclose(roi_pool(patches, (0, 0, 14, 14)), [1, 1])
    assert np.allclose(roi_pool(patches, None), [1 / 256, 1 / 256])


def test_malignancy_head_uses_signal():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    cls = rng.normal(0, 1, (n, 6)) + y[:, None] * 1.5
    patches = rng.normal(0, 1, (n, 16, 16, 6))
    X = lesion_features(cls, patches, [None] * n, [None if i % 3 else 50.0 for i in range(n)])
    assert X.shape == (n, 14)
    head = MalignancyHead().fit(X[:150], y[:150], np.arange(150))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[150:], head.predict(X[150:])) > 0.85
