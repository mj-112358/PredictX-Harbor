"""Cohort loaders for the five PredictX endpoints.

Every loader returns a :class:`Cohort`: a feature table with ``NaN`` for unrecorded values, binary labels
(1 = disease), a feature specification and optional grouping/auxiliary data. Two feature sets are provided:

* ``"full"`` – every predictor recorded in the source cohort (used for benchmarking).
* ``"web"``  – exactly the fields collected by the PredictX web forms (used for the deployed models).

For the coronary, diabetes and breast endpoints the two sets coincide.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("PREDICTX_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))

#: A feature is either numeric (``"num"``) or categorical with ``n`` integer-coded levels (``("cat", n)``).
FeatureType = Union[str, Tuple[str, int]]
FeatureSpec = Dict[str, FeatureType]

ENDPOINTS = ("heart", "diabetes", "breast", "liver", "kidney")


@dataclass
class Cohort:
    """A disease cohort: features ``X``, labels ``y``, feature specification, and optional patient groups
    (repeated samples of one patient), auxiliary cohorts and the categorical level names."""

    name: str
    X: pd.DataFrame
    y: np.ndarray
    spec: FeatureSpec
    groups: Optional[np.ndarray] = None
    auxiliary: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
    categories: Dict[str, list] = field(default_factory=dict)

    @property
    def categorical_indices(self) -> list:
        return [i for i, t in enumerate(self.spec.values()) if t != "num"]


def _encode(series: pd.Series, levels: list) -> pd.Series:
    """Map categorical values to integer codes 0..K-1 (``NaN`` stays ``NaN``)."""
    mapping = {v: i for i, v in enumerate(levels)}
    return series.astype("object").map(mapping).astype(float)


# --------------------------------------------------------------------------------------------- coronary
HEART_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
HEART_CATEGORICAL = {
    "sex": ["Female", "Male"],
    "cp": ["asymptomatic", "atypical angina", "non-anginal pain", "typical angina"],
    "fbs": [0.0, 1.0],
    "restecg": ["ST-T wave abnormality", "normal", "probable/definite hypertrophy"],
    "exang": ["No", "Yes"],
    "slope": ["downsloping", "flat", "upsloping"],
    "thal": ["fixed defect", "normal", "reversable defect"],
}


def _heart_site(site: str) -> pd.DataFrame:
    import pyreadr

    frame = list(pyreadr.read_r(os.path.join(DATA_DIR, f"heart_disease_{site}.rda")).values())[0].copy()
    # Zero cholesterol and zero resting blood pressure denote unrecorded values in the UCI files.
    frame.loc[frame["chol"] == 0, "chol"] = np.nan
    frame.loc[frame["trestbps"] == 0, "trestbps"] = np.nan
    return frame


def _heart_table(frame: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame(index=frame.index)
    for c in HEART_NUMERIC:
        X[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
    for c, levels in HEART_CATEGORICAL.items():
        values = frame[c].astype(float) if c == "fbs" else frame[c]
        X[c] = _encode(values, levels)
    y = (frame["num"].astype(float) > 0).astype(int).values
    return X.reset_index(drop=True), y


def load_heart(features: str = "full") -> Cohort:
    """Cleveland cohort; the Hungarian, Swiss and VA Long Beach cohorts are returned as auxiliary data."""
    X, y = _heart_table(_heart_site("cl"))
    aux = [_heart_table(_heart_site(s)) for s in ("hu", "ch", "va")]
    X_aux = pd.concat([a[0] for a in aux], ignore_index=True)
    y_aux = np.concatenate([a[1] for a in aux])
    spec: FeatureSpec = {c: "num" for c in HEART_NUMERIC}
    spec.update({c: ("cat", len(v)) for c, v in HEART_CATEGORICAL.items()})
    return Cohort("heart", X, y, spec, auxiliary=(X_aux, y_aux), categories=dict(HEART_CATEGORICAL))


# --------------------------------------------------------------------------------------------- kidney
KIDNEY_WEB_NUMERIC = ["bp", "sg", "al", "su"]
KIDNEY_WEB_CATEGORICAL = {"rbc": ["normal", "abnormal"], "pc": ["normal", "abnormal"],
                          "pcc": ["notpresent", "present"]}
KIDNEY_EXTRA_NUMERIC = ["age", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
KIDNEY_EXTRA_BINARY = ["ba", "htn", "dm", "cad", "appet", "pe", "ane"]


def _read_ckd() -> pd.DataFrame:
    names, rows, in_data = [], [], False
    with open(os.path.join(DATA_DIR, "ckd_full.arff"), encoding="latin-1") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if s.lower().startswith("@attribute"):
                names.append(s.split()[1].strip("'"))
            elif s.lower().startswith("@data"):
                in_data = True
            elif in_data:
                rows.append([t.strip() for t in s.split(",") if t.strip() != ""])
    frame = pd.DataFrame(rows, columns=names).replace("?", np.nan)
    for c in frame.columns:
        if frame[c].dtype == object:
            frame[c] = frame[c].str.strip()
    return frame


def load_kidney(features: str = "full") -> Cohort:
    frame = _read_ckd()
    y = (frame["class"] == "ckd").astype(int).values
    X = pd.DataFrame(index=frame.index)
    spec: FeatureSpec = {}
    for c in KIDNEY_WEB_NUMERIC:
        X[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
        spec[c] = "num"
    for c, levels in KIDNEY_WEB_CATEGORICAL.items():
        X[c] = _encode(frame[c], levels)
        spec[c] = ("cat", 2)
    if features == "full":
        for c in KIDNEY_EXTRA_NUMERIC:
            X[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
            spec[c] = "num"
        for c in KIDNEY_EXTRA_BINARY:
            X[c] = _encode(frame[c], sorted(frame[c].dropna().unique()))
            spec[c] = ("cat", 2)
    return Cohort("kidney", X, y, spec, categories=dict(KIDNEY_WEB_CATEGORICAL))


# --------------------------------------------------------------------------------------------- liver
LIVER_WEB = ["tb", "db", "alkphos", "sgpt", "tp", "alb", "ag_ratio"]


def load_liver(features: str = "full") -> Cohort:
    cols = ["age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp", "alb", "ag_ratio", "label"]
    frame = pd.read_csv(os.path.join(DATA_DIR, "ilpd.csv"), header=None, names=cols)
    y = (frame["label"] == 1).astype(int).values
    X = frame[LIVER_WEB].astype(float).copy()
    spec: FeatureSpec = {c: "num" for c in LIVER_WEB}
    if features == "full":
        X.insert(0, "age", frame["age"].astype(float))
        X.insert(1, "male", (frame["gender"] == "Male").astype(float))
        X["sgot"] = frame["sgot"].astype(float)
        spec = {c: "num" for c in X.columns}
        spec["male"] = ("cat", 2)
    return Cohort("liver", X, y, spec)


# --------------------------------------------------------------------------------------------- diabetes
DIABETES_FIELDS = ["pregnancies", "glucose", "dbp", "skin", "insulin", "bmi", "dpf", "age"]


def load_diabetes(features: str = "full") -> Cohort:
    frame = pd.read_csv(os.path.join(DATA_DIR, "pima.csv"))
    frame.columns = DIABETES_FIELDS + ["y"]
    for c in ["glucose", "dbp", "skin", "insulin", "bmi"]:  # zero denotes an unrecorded value
        frame.loc[frame[c] == 0, c] = np.nan
    return Cohort("diabetes", frame[DIABETES_FIELDS].astype(float), frame["y"].astype(int).values,
                  {c: "num" for c in DIABETES_FIELDS})


# --------------------------------------------------------------------------------------------- breast
BREAST_FIELDS = ["clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape", "marginal_adhesion",
                 "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses"]


def load_breast(features: str = "full") -> Cohort:
    import pyreadr

    frame = list(pyreadr.read_r(os.path.join(DATA_DIR, "bcw_original.rda")).values())[0].copy()
    X = frame[BREAST_FIELDS].apply(pd.to_numeric, errors="coerce").astype(float)
    y = frame["class"].astype(str).str.lower().isin(["4", "malignant"]).astype(int).values
    return Cohort("breast", X, y, {c: "num" for c in BREAST_FIELDS}, groups=frame["sample_code_number"].values)


LOADERS = {"heart": load_heart, "diabetes": load_diabetes, "breast": load_breast, "liver": load_liver,
           "kidney": load_kidney}


def load(name: str, features: str = "full") -> Cohort:
    """Load endpoint ``name`` with feature set ``"full"`` or ``"web"``."""
    if features not in ("full", "web"):
        raise ValueError("features must be 'full' or 'web'")
    return LOADERS[name](features)


#: Column order of the web feature set of each endpoint (identical to ``load(name, "web").X.columns``).
WEB_COLUMNS = {
    "heart": HEART_NUMERIC + list(HEART_CATEGORICAL),
    "diabetes": DIABETES_FIELDS,
    "breast": BREAST_FIELDS,
    "liver": LIVER_WEB,
    "kidney": KIDNEY_WEB_NUMERIC + list(KIDNEY_WEB_CATEGORICAL),
}


def load_columns(name: str) -> list:
    """Model input columns of the web feature set of endpoint ``name``."""
    return list(WEB_COLUMNS[name])
