"""Fold-safe preprocessing. Every transformer here is fitted on training rows only."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from .cohorts import FeatureSpec


def tabular_preprocessor(spec: FeatureSpec, scale: bool = True) -> ColumnTransformer:
    """Median imputation with missing-value indicators (numeric), optional standardisation, and one-hot
    encoding with an explicit "missing" level (categorical)."""
    numeric = [c for c, t in spec.items() if t == "num"]
    categorical = [c for c, t in spec.items() if t != "num"]
    steps = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        steps.append(StandardScaler())
    transformers = [("num", make_pipeline(*steps), numeric)]
    if categorical:
        transformers.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value=-1),
                                                  OneHotEncoder(handle_unknown="ignore")), categorical))
    return ColumnTransformer(transformers)


class TokenEncoder:
    """Converts a feature table into the three arrays consumed by HARBOR.

    ``V`` holds quantile-normalised numeric values (0 where unobserved or categorical), ``C`` integer
    category codes (0 where unobserved or numeric) and ``M`` the availability mask (1 = observed).
    """

    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self.quantiles_: Dict[str, QuantileTransformer] = {}

    def fit(self, X: pd.DataFrame) -> "TokenEncoder":
        for c, t in self.spec.items():
            if t == "num":
                v = X[c].dropna().values.reshape(-1, 1)
                self.quantiles_[c] = QuantileTransformer(n_quantiles=min(200, len(v)),
                                                         output_distribution="normal",
                                                         subsample=10**9).fit(v)
        return self

    def transform(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        n, J = X.shape
        V = np.zeros((n, J), np.float32)
        C = np.zeros((n, J), np.int32)
        M = (~X.isna().values).astype(np.float32)
        for j, (c, t) in enumerate(self.spec.items()):
            observed = M[:, j] > 0
            if not observed.any():
                continue
            if t == "num":
                z = self.quantiles_[c].transform(X.loc[observed, c].values.reshape(-1, 1)).ravel()
                V[observed, j] = np.clip(z, -5, 5)
            else:
                C[observed, j] = X.loc[observed, c].values.astype(np.int32)
        return {"V": V, "C": C, "M": M}
