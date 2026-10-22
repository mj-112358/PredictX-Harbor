import numpy as np

from sweepscreen.quality import FAILURE_MODES, FEATURE_NAMES, QualityGate, coverage_status, frame_features, synthesize_failure
from tests.conftest import synthetic_frame


def _frames(rng, n):
    return [synthetic_frame(rng, h=160, w=200, ui=False)[0][20:140, 20:180].copy() for _ in range(n)]


def test_feature_vector_shape(rng):
    f = frame_features(_frames(rng, 1)[0])
    assert f.shape == (len(FEATURE_NAMES),) and np.isfinite(f).all()


def test_contact_feature_detects_dark_side(rng):
    g = _frames(rng, 1)[0]
    bad = synthesize_failure(g, "contact", np.random.default_rng(1))
    i = FEATURE_NAMES.index("contact_max")
    assert frame_features(bad)[i] > 0.4 > frame_features(g)[i]


def test_gate_separates_failures(rng):
    train, test = _frames(rng, 60), _frames(rng, 20)
    gate = QualityGate(target_pass=0.95).fit(train)
    assert gate.adequate(test).mean() >= 0.8
    r = np.random.default_rng(5)
    for mode in FAILURE_MODES:
        bad = [synthesize_failure(g, mode, r) for g in test]
        assert gate.adequate(bad).mean() <= 0.2, mode


def test_coverage_rule():
    assert coverage_status([True] * 40 + [False] * 10)["readable"]
    assert not coverage_status([True] * 20)["readable"]              # too few frames
    assert not coverage_status([True] * 40 + [False] * 60)["readable"]  # too low a fraction
