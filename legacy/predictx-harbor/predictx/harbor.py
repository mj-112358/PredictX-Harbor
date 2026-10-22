"""HARBOR: reliability-conditioned biomarker graph encoder (JAX implementation).

A patient is a set of ``J`` biomarker tokens. Observed tokens combine a concept embedding with a value
embedding; unobserved tokens keep their concept embedding and receive a learned missingness-state
embedding. Each layer mixes two attention graphs with an availability-dependent gate ``gamma``:

* the value graph ``A^v`` attends only among observed biomarkers over gated value tokens
  ``H^v = m * r * H``;
* the missingness-state graph ``A^m`` attends among all biomarkers over availability tokens
  ``H^m = E_c + m * e_obs + (1 - m) * e_miss`` that carry no measured value.

Sparsity of ``A^v`` is encouraged with a row-entropy penalty (an L1 penalty is constant under softmax).

Variants used in the ablations:
    ``two_channel``  the model above (default)
    ``mask_concat``  one mask-aware token set, two attention heads over the same tokens
    ``value_only``   unobserved tokens are zeroed and excluded (no missingness information)
    ``no_gate``      two_channel with the reliability gate fixed to 1
    ``no_graph``     two_channel with both attention matrices replaced by the identity
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from sklearn.model_selection import train_test_split

from .cohorts import FeatureSpec
from .preprocessing import TokenEncoder

VARIANTS = ("two_channel", "mask_concat", "value_only", "no_gate", "no_graph")
_NEG_INF = -1e9


# ------------------------------------------------------------------------------------------ parameters
def init_params(key: jax.Array, n_features: int, n_category_rows: int, width: int = 32, depth: int = 2) -> dict:
    """Initial parameters of a HARBOR encoder with ``depth`` layers of width ``width``.

    Embeddings: ``E`` (concepts), ``w_val``/``b_val`` (numeric values), ``cat`` (categorical levels),
    ``e_obs``/``e_miss`` (availability states), ``w_r``/``b_r`` (reliability gate). Per layer: ``Q1``/``K1``/``W1``
    (value graph), ``Q2``/``K2``/``W2`` (missingness-state graph), ``w_g``/``b_g`` (channel gate), a two-layer
    feed-forward block ``F1``/``F2`` and two layer norms. ``head1``/``head2`` form the output MLP.
    """
    keys = iter(jax.random.split(key, 64))

    def normal(*shape, scale=None):
        return jax.random.normal(next(keys), shape) * (scale if scale is not None else 1 / np.sqrt(shape[0]))

    d = width
    params = {
        "E": normal(n_features, d, scale=0.1),            # concept embeddings E_c
        "w_val": normal(n_features, d, scale=0.1),        # numeric value embedding v * w + b
        "b_val": normal(n_features, d, scale=0.1),
        "cat": normal(max(n_category_rows, 1), d, scale=0.1),
        "e_miss": normal(d, scale=0.1),                   # missingness-state embedding
        "e_obs": normal(d, scale=0.1),                    # observed-state embedding
        "w_r": normal(d + 1, scale=0.1), "b_r": jnp.array(2.0),   # gate starts open: sigmoid(2) = 0.88
        "head1": normal(d, d), "head_b1": jnp.zeros(d), "head2": normal(d, scale=0.1), "head_b2": jnp.zeros(()),
        "layers": [],
    }
    for _ in range(depth):
        params["layers"].append({
            "Q1": normal(d, d), "K1": normal(d, d), "W1": normal(d, d, scale=0.1 / np.sqrt(d)),
            "Q2": normal(d, d), "K2": normal(d, d), "W2": normal(d, d, scale=0.1 / np.sqrt(d)),
            "w_g": normal(d, scale=0.1), "b_g": jnp.zeros(()),
            "ln1_s": jnp.ones(d), "ln1_b": jnp.zeros(d),
            "F1": normal(d, 2 * d), "Fb1": jnp.zeros(2 * d),
            "F2": normal(2 * d, d, scale=0.1 / np.sqrt(2 * d)), "Fb2": jnp.zeros(d),
            "ln2_s": jnp.ones(d), "ln2_b": jnp.zeros(d),
        })
    return params


# ------------------------------------------------------------------------------------------ forward pass
def _layer_norm(x, scale, bias):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * scale + bias


def attention(H, Q, K, key_mask):
    """Row-softmax attention weights over keys allowed by ``key_mask`` (shape ``(n, J)``)."""
    d = Q.shape[0]
    logits = jnp.einsum("njd,nkd->njk", H @ Q, H @ K) / np.sqrt(d)
    logits = jnp.where(key_mask[:, None, :] > 0, logits, _NEG_INF)
    return jax.nn.softmax(logits, axis=-1)


def row_entropy(A, row_mask):
    """Mean Shannon entropy of the attention rows selected by ``row_mask``."""
    entropy = -(A * jnp.log(A + 1e-12)).sum(-1)
    return (entropy * row_mask).sum() / (row_mask.sum() + 1e-9)


def forward(params, V, C, M, is_numeric, category_offset, variant: str = "two_channel"):
    """Return logits ``(n,)`` and the mean value-graph entropy for a batch of patients."""
    n, J = M.shape
    numeric_embedding = V[..., None] * params["w_val"] + params["b_val"]
    category_embedding = params["cat"][C + category_offset[None, :]]
    value_embedding = jnp.where(is_numeric[None, :, None] > 0, numeric_embedding, category_embedding)
    E = jnp.broadcast_to(params["E"], (n, J, params["E"].shape[1]))
    h0 = E + value_embedding
    m = M[..., None]
    if variant == "no_gate":
        r = jnp.ones((n, J))
    else:
        r = jax.nn.sigmoid(jnp.concatenate([h0, M[..., None]], -1) @ params["w_r"] + params["b_r"])
    gated_mask = (M * r)[..., None]
    missing_token = E + params["e_miss"]
    availability = E + m * params["e_obs"] + (1 - m) * params["e_miss"]
    all_keys = jnp.ones_like(M)
    entropy = 0.0

    H = gated_mask * h0 if variant == "value_only" else m * h0 + (1 - m) * missing_token

    for lp in params["layers"]:
        if variant == "value_only":
            Hv = gated_mask * H
            Av = attention(Hv, lp["Q1"], lp["K1"], M)
            message = Av @ (Hv @ lp["W1"])
            entropy += row_entropy(Av, M)
            H = _layer_norm(H + message, lp["ln1_s"], lp["ln1_b"]) * m
        elif variant == "mask_concat":
            Hs = m * (r[..., None] * H) + (1 - m) * H
            A1 = attention(Hs, lp["Q1"], lp["K1"], all_keys)
            A2 = attention(Hs, lp["Q2"], lp["K2"], all_keys)
            gamma = jax.nn.sigmoid(Hs.mean(1) @ lp["w_g"] + lp["b_g"])[:, None, None]
            message = gamma * (A1 @ (Hs @ lp["W1"])) + (1 - gamma) * (A2 @ (Hs @ lp["W2"]))
            entropy += row_entropy(A1, all_keys)
            H = _layer_norm(H + message, lp["ln1_s"], lp["ln1_b"])
        else:
            Hv = gated_mask * H
            gamma = jax.nn.sigmoid(availability.mean(1) @ lp["w_g"] + lp["b_g"])[:, None, None]
            if variant == "no_graph":
                message = gamma * (Hv @ lp["W1"]) + (1 - gamma) * (availability @ lp["W2"])
            else:
                Av = attention(Hv, lp["Q1"], lp["K1"], M)
                Am = attention(availability, lp["Q2"], lp["K2"], all_keys)
                message = gamma * (Av @ (Hv @ lp["W1"])) + (1 - gamma) * (Am @ (availability @ lp["W2"]))
                entropy += row_entropy(Av, M)
            H = _layer_norm(H + message, lp["ln1_s"], lp["ln1_b"])
        F = jax.nn.gelu(H @ lp["F1"] + lp["Fb1"]) @ lp["F2"] + lp["Fb2"]
        H = _layer_norm(H + F, lp["ln2_s"], lp["ln2_b"])
        if variant == "value_only":
            H = H * m

    pooled = H.sum(1) / (M.sum(1, keepdims=True) + 1e-9) if variant == "value_only" else H.mean(1)
    logits = jax.nn.gelu(pooled @ params["head1"] + params["head_b1"]) @ params["head2"] + params["head_b2"]
    return logits, entropy / len(params["layers"])


# ------------------------------------------------------------------------------------------ training
_COMPILED: Dict[tuple, tuple] = {}


def _compiled_functions(variant, is_numeric, category_offset, lam_graph, learning_rate, weight_decay):
    """JIT-compiled training step, validation loss and prediction function, cached per configuration."""
    key = (variant, tuple(np.asarray(is_numeric).tolist()), tuple(np.asarray(category_offset).tolist()),
           lam_graph, learning_rate, weight_decay)
    if key in _COMPILED:
        return _COMPILED[key]
    is_numeric = jnp.asarray(is_numeric, dtype=jnp.float32)
    category_offset = jnp.asarray(category_offset, dtype=jnp.int32)
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)

    def loss_fn(p, V, C, M, y, w):
        z, entropy = forward(p, V, C, M, is_numeric, category_offset, variant)
        bce = optax.sigmoid_binary_cross_entropy(z, y)
        return (bce * w).sum() / w.sum() + lam_graph * entropy

    @jax.jit
    def step(p, state, V, C, M, y, w):
        loss, grads = jax.value_and_grad(loss_fn)(p, V, C, M, y, w)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss

    @jax.jit
    def validation_loss(p, V, C, M, y, w):
        z, _ = forward(p, V, C, M, is_numeric, category_offset, variant)
        return (optax.sigmoid_binary_cross_entropy(z, y) * w).sum() / w.sum()

    @jax.jit
    def probabilities(p, V, C, M):
        z, _ = forward(p, V, C, M, is_numeric, category_offset, variant)
        return jax.nn.sigmoid(z)

    _COMPILED[key] = (optimizer, step, validation_loss, probabilities)
    return _COMPILED[key]


def _pad(a: np.ndarray, rows: int) -> np.ndarray:
    """Zero-pad ``a`` to ``rows`` rows so every batch has one static shape (one compilation per model)."""
    out = np.zeros((rows,) + a.shape[1:], dtype=a.dtype)
    out[: len(a)] = a
    return out


def feature_layout(spec: FeatureSpec) -> Tuple[np.ndarray, np.ndarray, int]:
    """Numeric indicator, offset into the shared category table, and total number of category rows."""
    is_numeric, offsets, total = [], [], 0
    for t in spec.values():
        if t == "num":
            is_numeric.append(1)
            offsets.append(0)
        else:
            is_numeric.append(0)
            offsets.append(total)
            total += t[1]
    return np.array(is_numeric), np.array(offsets), total


class HarborClassifier:
    """Deep ensemble of HARBOR encoders, each early-stopped on its own stratified validation split.

    Parameters
    ----------
    spec : feature specification of the input table.
    seed : base random seed.
    variant : one of :data:`VARIANTS`.
    n_members : ensemble size.
    width, depth : token dimension and number of layers.
    learning_rate, weight_decay : AdamW settings.
    lam_graph : weight of the attention-entropy penalty.
    max_epochs, patience : early-stopping budget (full-batch epochs).
    validation_fraction : share of the training rows held out for early stopping.
    """

    def __init__(self, spec: FeatureSpec, seed: int = 0, variant: str = "two_channel", n_members: int = 5,
                 width: int = 32, depth: int = 2, learning_rate: float = 3e-3, weight_decay: float = 1e-4,
                 lam_graph: float = 1e-3, max_epochs: int = 400, patience: int = 40, validation_fraction: float = 0.2):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.spec, self.seed, self.variant, self.n_members = spec, seed, variant, n_members
        self.width, self.depth = width, depth
        self.learning_rate, self.weight_decay, self.lam_graph = learning_rate, weight_decay, lam_graph
        self.max_epochs, self.patience, self.validation_fraction = max_epochs, patience, validation_fraction
        self.members_: List[Tuple[TokenEncoder, dict]] = []

    def _functions(self):
        is_numeric, offsets, _ = feature_layout(self.spec)
        return _compiled_functions(self.variant, is_numeric, offsets, self.lam_graph, self.learning_rate,
                                   self.weight_decay)

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "HarborClassifier":
        """Train ``n_members`` encoders with AdamW; each keeps the parameters of its lowest validation loss."""
        y = np.asarray(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        _, _, n_category_rows = feature_layout(self.spec)
        optimizer, step, validation_loss, _ = self._functions()
        self.pad_rows_ = len(y) + 8
        self.members_ = []
        for member in range(self.n_members):
            train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=self.validation_fraction, stratify=y,
                                                  random_state=self.seed * 31 + member)
            encoder = TokenEncoder(self.spec).fit(X.iloc[train_idx])
            tr, va = encoder.transform(X.iloc[train_idx]), encoder.transform(X.iloc[val_idx])
            N = self.pad_rows_
            tr_arrays = [jnp.asarray(_pad(tr[k], N)) for k in ("V", "C", "M")]
            va_arrays = [jnp.asarray(_pad(va[k], N)) for k in ("V", "C", "M")]
            y_tr = jnp.asarray(_pad(y[train_idx].astype(np.float32), N))
            w_tr = jnp.asarray(_pad(w[train_idx].astype(np.float32), N))
            y_va = jnp.asarray(_pad(y[val_idx].astype(np.float32), N))
            w_va = jnp.asarray(_pad(np.ones(len(val_idx), np.float32), N))

            params = init_params(jax.random.PRNGKey(self.seed * 101 + member), len(self.spec), n_category_rows,
                                 self.width, self.depth)
            state = optimizer.init(params)
            best_loss, best_params, stall = np.inf, params, 0
            for _ in range(self.max_epochs):
                params, state, _ = step(params, state, *tr_arrays, y_tr, w_tr)
                loss = float(validation_loss(params, *va_arrays, y_va, w_va))
                if loss < best_loss - 1e-5:
                    best_loss, best_params, stall = loss, params, 0
                else:
                    stall += 1
                    if stall >= self.patience:
                        break
            self.members_.append((encoder, jax.tree_util.tree_map(np.asarray, best_params)))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Average of the member probabilities, returned as an ``(n, 2)`` array."""
        _, _, _, probabilities = self._functions()
        N = self.pad_rows_
        member_predictions = []
        for encoder, params in self.members_:
            arrays = encoder.transform(X)
            out = []
            for start in range(0, len(X), N):
                chunk = [jnp.asarray(_pad(arrays[k][start:start + N], N)) for k in ("V", "C", "M")]
                out.append(np.asarray(probabilities(params, *chunk))[: min(N, len(X) - start)])
            member_predictions.append(np.concatenate(out))
        p = np.mean(member_predictions, axis=0)
        return np.column_stack([1 - p, p])
