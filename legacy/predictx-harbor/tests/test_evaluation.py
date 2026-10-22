import numpy as np
import pytest

from predictx.evaluation import corrected_resampled_ttest, half_splits, repeated_cv


def test_repeated_cv_partitions_every_repeat():
    y = np.array([0, 1] * 50)
    splits = list(repeated_cv(y))
    assert len(splits) == 25
    for rep in range(5):
        tests = np.concatenate([te for r, _, _, te in splits if r == rep])
        assert sorted(tests.tolist()) == list(range(100))
    for _, _, train, test in splits:
        assert not set(train) & set(test)


def test_grouped_splits_keep_groups_together():
    y = np.array([0, 1] * 50)
    groups = np.repeat(np.arange(50), 2)
    for _, _, train, test in repeated_cv(y, groups):
        assert not set(groups[train]) & set(groups[test])
    for _, _, train, test in half_splits(y, groups):
        assert not set(groups[train]) & set(groups[test])


def test_corrected_interval_matches_closed_form():
    d = np.array([0.01, 0.03, -0.01, 0.02, 0.00, 0.04, 0.01, 0.02, 0.03, -0.02])
    mean, lo, hi, p = corrected_resampled_ttest(d, test_fraction=0.2)
    from scipy import stats
    se = np.sqrt(d.var(ddof=1) * (1 / 10 + 0.25))
    assert mean == pytest.approx(d.mean())
    assert hi - mean == pytest.approx(stats.t.ppf(0.975, 9) * se)
    assert lo < mean < hi and 0 < p < 1


def test_correction_widens_the_naive_interval():
    d = np.random.default_rng(0).normal(0.01, 0.02, 25)
    _, lo, hi, _ = corrected_resampled_ttest(d, 0.2)
    naive = 2 * 2.064 * d.std(ddof=1) / np.sqrt(25)
    assert hi - lo > naive
