import numpy as np

from sweepscreen.evaluation.detection import average_precision, froc, iou, match
from sweepscreen.evaluation.metrics import auroc, bootstrap_ci, sens_at_spec, spec_at_sens


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0


def test_match_greedy_by_score():
    m = match([((0, 0, 10, 10), 0.2), ((0, 0, 10, 9), 0.9)], [(0, 0, 10, 10)], 0.5)
    assert m == [(0.9, True), (0.2, False)]


def test_perfect_detector_ap_and_froc():
    gts = [[(0, 0, 10, 10)], [], [(5, 5, 20, 20)]]
    preds = [[((0, 0, 10, 10), 0.9)], [((1, 1, 3, 3), 0.1)], [((5, 5, 20, 20), 0.8)]]
    assert average_precision(preds, gts) == 1.0
    f = froc(preds, gts)
    assert f["at"][0.5] == 1.0 and f["n_lesions"] == 2


def test_ap_half():
    gts = [[(0, 0, 10, 10)], [(0, 0, 10, 10)]]
    preds = [[((0, 0, 10, 10), 0.9)], [((50, 50, 60, 60), 0.8)]]
    assert abs(average_precision(preds, gts) - 0.5) < 1e-9


def test_classification_metrics():
    y = np.array([0] * 100 + [1] * 100)
    p = np.r_[np.linspace(0, 0.6, 100), np.linspace(0.4, 1, 100)]
    assert 0.9 < auroc(y, p) < 1
    assert sens_at_spec(y, p, 0.95) < 1 and spec_at_sens(y, p, 0.95) < 1
    est, lo, hi = bootstrap_ci(auroc, y, p, n_boot=200)
    assert lo <= est <= hi
