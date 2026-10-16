"""Input schemas of the five PredictX web forms and their conversion to model features.

Each endpoint has a schema of named fields. :func:`to_model_frame` validates a payload (a mapping from field
name to value) and returns a one-row :class:`pandas.DataFrame` in the column order and coding the trained
model expects. Numeric fields may be omitted or ``None``; the models treat them as unrecorded.

The coronary form on the website encodes categorical answers as one-hot fields (for example ``Sex_1`` or ``CP_2``);
:func:`heart_from_onehot` converts that layout to the named schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from . import cohorts


@dataclass(frozen=True)
class Field:
    feature: str                                   # model feature name
    label: str                                     # label shown on the form
    unit: str = ""
    range: Optional[Tuple[float, float]] = None    # admissible numeric range
    options: Optional[Dict[str, Any]] = None       # categorical answer -> value stored in the cohort
    zero_is_missing: bool = False                  # a recorded 0 means "not measured"


SCHEMAS: Dict[str, Dict[str, Field]] = {
    "heart": {
        "age": Field("age", "Age", "years", (1, 120)),
        "resting_bp": Field("trestbps", "Resting systolic blood pressure", "mmHg", (50, 260), zero_is_missing=True),
        "cholesterol": Field("chol", "Total cholesterol", "mg/dL", (50, 700), zero_is_missing=True),
        "max_heart_rate": Field("thalach", "Maximum exercise heart rate", "beats/min", (40, 230)),
        "st_depression": Field("oldpeak", "Exercise-induced ST depression", "mm", (-3, 10)),
        "major_vessels": Field("ca", "Major vessels coloured by fluoroscopy", "count", (0, 3)),
        "sex": Field("sex", "Sex", options={"female": "Female", "male": "Male"}),
        "chest_pain": Field("cp", "Chest-pain type", options={
            "typical_angina": "typical angina", "atypical_angina": "atypical angina",
            "non_anginal_pain": "non-anginal pain", "asymptomatic": "asymptomatic"}),
        "fasting_glucose_over_120": Field("fbs", "Fasting blood sugar > 120 mg/dL", options={False: 0.0, True: 1.0}),
        "resting_ecg": Field("restecg", "Resting ECG", options={
            "normal": "normal", "st_t_abnormality": "ST-T wave abnormality",
            "lv_hypertrophy": "probable/definite hypertrophy"}),
        "exercise_angina": Field("exang", "Exercise-induced angina", options={False: "No", True: "Yes"}),
        "st_slope": Field("slope", "Peak exercise ST segment", options={
            "upsloping": "upsloping", "flat": "flat", "downsloping": "downsloping"}),
        "thallium": Field("thal", "Thallium perfusion test", options={
            "normal": "normal", "fixed_defect": "fixed defect", "reversible_defect": "reversable defect"}),
    },
    "diabetes": {
        "pregnancies": Field("pregnancies", "Number of pregnancies", "count", (0, 25)),
        "glucose": Field("glucose", "Plasma glucose", "mg/dL", (20, 600), zero_is_missing=True),
        "diastolic_bp": Field("dbp", "Diastolic blood pressure", "mmHg", (20, 160), zero_is_missing=True),
        "skin_thickness": Field("skin", "Triceps skin-fold thickness", "mm", (3, 100), zero_is_missing=True),
        "insulin": Field("insulin", "2-hour serum insulin", "µU/mL", (5, 1000), zero_is_missing=True),
        "bmi": Field("bmi", "Body-mass index", "kg/m²", (10, 80), zero_is_missing=True),
        "diabetes_pedigree": Field("dpf", "Diabetes pedigree function", "", (0, 3)),
        "age": Field("age", "Age", "years", (1, 120)),
    },
    "breast": {
        "clump_thickness": Field("clump_thickness", "Clump thickness", "score 1-10", (1, 10)),
        "cell_size_uniformity": Field("uniformity_of_cell_size", "Uniformity of cell size", "score 1-10", (1, 10)),
        "cell_shape_uniformity": Field("uniformity_of_cell_shape", "Uniformity of cell shape", "score 1-10", (1, 10)),
        "marginal_adhesion": Field("marginal_adhesion", "Marginal adhesion", "score 1-10", (1, 10)),
        "epithelial_cell_size": Field("single_epithelial_cell_size", "Single epithelial cell size", "score 1-10",
                                      (1, 10)),
        "bare_nuclei": Field("bare_nuclei", "Bare nuclei", "score 1-10", (1, 10)),
        "bland_chromatin": Field("bland_chromatin", "Bland chromatin", "score 1-10", (1, 10)),
        "normal_nucleoli": Field("normal_nucleoli", "Normal nucleoli", "score 1-10", (1, 10)),
        "mitoses": Field("mitoses", "Mitoses", "score 1-10", (1, 10)),
    },
    "liver": {
        "total_bilirubin": Field("tb", "Total bilirubin", "mg/dL", (0, 80)),
        "direct_bilirubin": Field("db", "Direct bilirubin", "mg/dL", (0, 40)),
        "alkaline_phosphatase": Field("alkphos", "Alkaline phosphatase (ALP)", "U/L", (10, 3000)),
        "alt": Field("sgpt", "Alanine aminotransferase (ALT)", "U/L", (1, 3000)),
        "total_protein": Field("tp", "Total protein", "g/dL", (1, 12)),
        "albumin": Field("alb", "Albumin", "g/dL", (0.5, 7)),
        "ag_ratio": Field("ag_ratio", "Albumin/globulin ratio", "", (0.1, 4)),
    },
    "kidney": {
        "blood_pressure": Field("bp", "Blood pressure", "mmHg", (40, 200)),
        "specific_gravity": Field("sg", "Urine specific gravity", "", (1.0, 1.04)),
        "albumin_grade": Field("al", "Urine albumin dipstick grade", "0-5", (0, 5)),
        "sugar_grade": Field("su", "Urine sugar dipstick grade", "0-5", (0, 5)),
        "red_blood_cells": Field("rbc", "Urine red blood cells", options={"normal": "normal", "abnormal": "abnormal"}),
        "pus_cells": Field("pc", "Urine pus cells", options={"normal": "normal", "abnormal": "abnormal"}),
        "pus_cell_clumps": Field("pcc", "Pus cell clumps", options={"not_present": "notpresent", "present": "present"}),
    },
}


def _category_levels(endpoint: str) -> Dict[str, list]:
    if endpoint == "heart":
        return cohorts.HEART_CATEGORICAL
    if endpoint == "kidney":
        return cohorts.KIDNEY_WEB_CATEGORICAL
    return {}


def to_model_frame(endpoint: str, payload: Mapping[str, Any]) -> pd.DataFrame:
    """Validate ``payload`` against the endpoint schema and return the model input row."""
    if endpoint not in SCHEMAS:
        raise ValueError(f"unknown endpoint {endpoint!r}")
    schema = SCHEMAS[endpoint]
    unknown = set(payload) - set(schema)
    if unknown:
        raise ValueError(f"unknown fields for {endpoint}: {sorted(unknown)}")
    if endpoint == "liver" and payload.get("ag_ratio") is None:
        tp, alb = payload.get("total_protein"), payload.get("albumin")
        if tp is not None and alb is not None and tp > alb:
            payload = {**payload, "ag_ratio": alb / (tp - alb)}
    levels = _category_levels(endpoint)
    row: Dict[str, float] = {}
    for name, field in schema.items():
        value = payload.get(name)
        if value is None:
            row[field.feature] = np.nan
        elif field.options is not None:
            if value not in field.options:
                raise ValueError(f"{name}: expected one of {list(field.options)}, got {value!r}")
            row[field.feature] = float(levels[field.feature].index(field.options[value]))
        else:
            value = float(value)
            if field.zero_is_missing and value == 0:
                row[field.feature] = np.nan
                continue
            lo, hi = field.range
            if not lo <= value <= hi:
                raise ValueError(f"{name}={value} outside the admissible range [{lo}, {hi}] {field.unit}")
            row[field.feature] = value
    columns = list(cohorts.load_columns(endpoint))
    return pd.DataFrame([row])[columns]


def heart_from_onehot(form: Mapping[str, float]) -> Dict[str, Any]:
    """Convert the website's one-hot coronary form to the named schema.

    Numeric fields keep their website labels; categorical groups are read from their one-hot columns:
    ``Sex_0``/``Sex_1`` (female/male), ``CP_0..CP_3`` (typical angina, atypical angina, non-anginal pain,
    asymptomatic), ``FBS_0`` (1 when fasting blood sugar <= 120 mg/dL), resting-ECG, exercise-angina,
    ST-slope and thallium groups, and ``CA_0..CA_2`` (number of vessels; none set means three).
    """
    def pick(options: Dict[str, Any]):
        chosen = [value for key, value in options.items() if float(form.get(key, 0)) == 1]
        if len(chosen) != 1:
            raise ValueError(f"exactly one of {list(options)} must be 1")
        return chosen[0]

    vessels = [k for k in range(3) if float(form.get(f"CA_{k}", 0)) == 1]
    return {
        "age": form.get("Age in Years"),
        "resting_bp": form.get("Resting systolic blood pressure (mmHg)"),
        "cholesterol": form.get("Total cholesterol (mg/dL)"),
        "max_heart_rate": form.get("Maximum exercise heart rate (beats/min)"),
        "st_depression": form.get("Exercise-induced ST depression (mm)"),
        "sex": pick({"Sex_0": "female", "Sex_1": "male"}),
        "chest_pain": pick({"CP_0": "typical_angina", "CP_1": "atypical_angina", "CP_2": "non_anginal_pain",
                            "CP_3": "asymptomatic"}),
        "fasting_glucose_over_120": float(form.get("FBS_0", 1)) != 1,
        "resting_ecg": pick({"Normal resting ECG": "normal", "ST-T abnormality on resting ECG": "st_t_abnormality",
                             "LVH on resting ECG": "lv_hypertrophy"}),
        "exercise_angina": pick({"No exercise-induced angina": False, "Exercise-induced angina": True}),
        "st_slope": pick({"Upsloping ST segment": "upsloping", "Horizontal ST segment": "flat",
                          "Downsloping ST segment": "downsloping"}),
        "major_vessels": vessels[0] if vessels else 3,
        "thallium": pick({"Normal perfusion test": "normal", "Fixed perfusion defect": "fixed_defect",
                          "Reversible perfusion defect": "reversible_defect"}),
    }
