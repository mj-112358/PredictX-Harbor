"""Patient-level and leave-one-site-out splits.

Splits operate on **groups** (patient + near-duplicate + video components from
:mod:`sweepscreen.data.dedup`), so no patient, video or near-duplicate image appears on both sides.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def group_kfold(table: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Stratified (by label) group k-fold over the rows of ``table`` (needs ``label`` and ``group_id``)."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(splitter.split(np.zeros(len(table)), table.label.values, table.group_id.values)):
        yield k, tr, te


def leave_one_site_out(table: pd.DataFrame, sites: List[str] | None = None) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """For each held-out site: (site, train row indices, test row indices).

    Training rows exclude every group that also has records at the held-out site (cross-site
    near-duplicates), so the held-out site is never seen in training.
    """
    sites = sites or sorted(table.site.unique())
    for site in sites:
        test = np.flatnonzero(table.site.values == site)
        leaked = set(table.group_id.values[test])
        train = np.flatnonzero((table.site.values != site) & ~table.group_id.isin(leaked).values)
        yield site, train, test


def check_disjoint(table: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> None:
    """Raise if any group or patient appears on both sides."""
    for col in ("group_id", "patient_id"):
        if col in table:
            both = set(table[col].values[train]) & set(table[col].values[test])
            if both:
                raise AssertionError(f"{len(both)} {col} values on both sides, e.g. {sorted(both)[:3]}")
