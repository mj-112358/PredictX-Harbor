import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def test_ensemble_never_scores_worse_than_its_average_member():
    """Proposition 4 (Jensen): proper scores of the averaged forecast are at most the members' mean score,
    and for the Brier score the gap equals the variance of member forecasts."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        y = rng.integers(0, 2, 30)
        P = rng.uniform(0.01, 0.99, (4, 30))
        avg = P.mean(0)
        assert brier_score_loss(y, avg) <= np.mean([brier_score_loss(y, p) for p in P]) + 1e-12
        assert log_loss(y, avg, labels=[0, 1]) <= np.mean([log_loss(y, p, labels=[0, 1]) for p in P]) + 1e-12
        gap = np.mean([brier_score_loss(y, p) for p in P]) - brier_score_loss(y, avg)
        assert np.isclose(gap, ((P - avg) ** 2).mean())
