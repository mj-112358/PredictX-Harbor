import numpy as np

from sweepscreen.aggregate import aggregate_breast, decide_woman, persistence_scores
from sweepscreen.safety import (PlattCalibrator, binomial_threshold, calibration_slope_intercept, clopper_pearson,
                                conformal_threshold, expected_calibration_error, min_positives)


def test_persistence_needs_consecutive_frames():
    p = [0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    s = persistence_scores(p, window=3)
    assert np.isnan(s[:2]).all()
    assert np.nanmax(s) == 1.0 and s[8] == 1.0     # three consecutive positives
    assert s[3] < 1


def test_persistence_breaks_at_inadequate_frames():
    p = [1, 1, 1, 1]
    s = persistence_scores(p, adequate=[True, True, False, True], window=2)
    assert s[1] == 1 and np.isnan(s[2]) and np.isnan(s[3])


def test_woman_decision_rules():
    lesion = aggregate_breast([0.1] * 10 + [0.9] * 6 + [0.1] * 30, window=5)
    clean = aggregate_breast([0.1] * 50, window=5)
    short = aggregate_breast([0.1] * 10, window=5)               # too few frames -> unreadable
    assert decide_woman({"left": lesion, "right": clean}, 0.5).decision == "RECALL"
    assert decide_woman({"left": clean, "right": clean}, 0.5).decision == "NORMAL"
    assert decide_woman({"left": clean, "right": short}, 0.5).decision == "UNREADABLE"
    assert decide_woman({"left": lesion, "right": short}, 0.5).decision == "RECALL"


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 59, 0.05, "lower")
    assert abs(lo - 0.0) < 1e-12
    assert abs(clopper_pearson(59, 59, 0.05, "lower")[0] - 0.95) < 0.001


def test_min_positives_matches_protocol_table():
    assert min_positives(0.95, 0) == 59 and min_positives(0.95, 1) == 93 and min_positives(0.90, 0) == 29


def test_binomial_threshold_guarantee_by_simulation():
    rng = np.random.default_rng(0)
    target, n, covered = 0.80, 60, 0
    found = 0
    for _ in range(400):
        pos = rng.normal(1, 1, n)
        t = binomial_threshold(pos, target)
        if t is None:
            continue
        found += 1
        from scipy.stats import norm
        covered += (1 - norm.cdf(t, loc=1, scale=1)) >= target   # true sensitivity at t
    assert found > 300 and covered / found >= 0.95


def test_conformal_threshold_marginal_coverage():
    rng = np.random.default_rng(1)
    hits = []
    for _ in range(2000):
        cal = rng.normal(size=30)
        t = conformal_threshold(cal, 0.9)
        hits.append(rng.normal() >= t)
    assert np.mean(hits) >= 0.88
    assert conformal_threshold(np.arange(5.0), 0.95) is None      # too few positives


def test_calibration_metrics():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 20000)
    y = rng.random(20000) < p
    slope, intercept = calibration_slope_intercept(p, y)
    assert abs(slope - 1) < 0.08 and abs(intercept) < 0.05
    assert expected_calibration_error(p, y) < 0.02
    p_over = np.clip(p * 0.5, 1e-3, 1)
    cal = PlattCalibrator().fit(p_over, y)
    assert expected_calibration_error(cal(p_over), y) < expected_calibration_error(p_over, y)
