import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from predictx.harbor import HarborClassifier, feature_layout, forward, init_params

SPEC = {"a": "num", "b": "num", "c": ("cat", 3)}


def _batch(n=16, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, 3)).astype(np.float32)
    V[:, 2] = 0
    C = np.zeros((n, 3), np.int32)
    C[:, 2] = rng.integers(0, 3, n)
    M = (rng.random((n, 3)) > 0.3).astype(np.float32)
    return jnp.asarray(V), jnp.asarray(C), jnp.asarray(M)


def test_l1_norm_of_softmax_attention_has_zero_gradient():
    """Proposition 2: the L1 norm of a row-softmax matrix is constant, so its gradient vanishes."""
    logits = jax.random.normal(jax.random.PRNGKey(0), (6, 6))

    def l1(z):
        return jnp.abs(jax.nn.softmax(z, axis=-1)).sum()

    assert float(l1(logits)) == pytest.approx(6.0, abs=1e-5)
    assert float(jnp.abs(jax.grad(l1)(logits)).max()) < 1e-6


def test_entropy_penalty_has_nonzero_gradient_and_sparsifies():
    """Proposition 3: row entropy has a usable gradient; descending it concentrates attention."""
    def entropy(z):
        a = jax.nn.softmax(z, axis=-1)
        return (-(a * jnp.log(a + 1e-12)).sum(-1)).mean()

    z = jax.random.normal(jax.random.PRNGKey(0), (6, 6))
    assert float(jnp.abs(jax.grad(entropy)(z)).max()) > 1e-2
    before = float(jax.nn.softmax(z, -1).max(-1).min())
    for _ in range(300):
        z = z - 0.5 * jax.grad(entropy)(z)
    assert float(jax.nn.softmax(z, -1).max(-1).min()) > max(before, 0.95)


@pytest.mark.parametrize("variant", ["two_channel", "mask_concat", "value_only", "no_gate", "no_graph"])
def test_unobserved_values_cannot_influence_the_output(variant):
    """Mask integrity: the stored value of an unobserved biomarker never reaches the prediction."""
    is_numeric, offsets, n_cat = feature_layout(SPEC)
    params = init_params(jax.random.PRNGKey(1), 3, n_cat)
    V, C, M = _batch()
    perturbed = jnp.where(M > 0, V, V + 100.0)
    z1, _ = forward(params, V, C, M, jnp.asarray(is_numeric), jnp.asarray(offsets), variant)
    z2, _ = forward(params, perturbed, C, M, jnp.asarray(is_numeric), jnp.asarray(offsets), variant)
    np.testing.assert_allclose(np.asarray(z1), np.asarray(z2), atol=1e-6)


def test_classifier_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 3, n).astype(float)})
    y = (X["a"] + 0.5 * X["c"] > 0.5).astype(int).values
    X.loc[rng.random(n) < 0.2, "b"] = np.nan
    model = HarborClassifier(SPEC, seed=0, n_members=2, max_epochs=150).fit(X.iloc[:150], y[:150])
    p = model.predict_proba(X.iloc[150:])
    assert p.shape == (50, 2) and np.allclose(p.sum(1), 1)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[150:], p[:, 1]) > 0.85
