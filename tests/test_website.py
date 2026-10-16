import numpy as np
import pytest

from predictx import cohorts
from predictx.website import SCHEMAS, heart_from_onehot, to_model_frame

EXAMPLES = {
    "heart": dict(age=58, resting_bp=140, cholesterol=260, max_heart_rate=120, st_depression=2.0, major_vessels=2,
                  sex="male", chest_pain="asymptomatic", fasting_glucose_over_120=False, resting_ecg="lv_hypertrophy",
                  exercise_angina=True, st_slope="flat", thallium="reversible_defect"),
    "diabetes": dict(pregnancies=2, glucose=150, diastolic_bp=70, skin_thickness=30, insulin=120, bmi=31.0,
                     diabetes_pedigree=0.5, age=45),
    "breast": dict(clump_thickness=8, cell_size_uniformity=7, cell_shape_uniformity=7, marginal_adhesion=5,
                   epithelial_cell_size=6, bare_nuclei=9, bland_chromatin=7, normal_nucleoli=8, mitoses=2),
    "liver": dict(total_bilirubin=3.2, direct_bilirubin=1.6, alkaline_phosphatase=450, alt=95, total_protein=6.2,
                  albumin=2.8, ag_ratio=0.82),
    "kidney": dict(blood_pressure=90, specific_gravity=1.010, albumin_grade=3, sugar_grade=0,
                   red_blood_cells="abnormal", pus_cells="abnormal", pus_cell_clumps="present"),
}

HEART_FORM = {
    "Age in Years": 58, "Resting systolic blood pressure (mmHg)": 140, "Total cholesterol (mg/dL)": 260,
    "Maximum exercise heart rate (beats/min)": 120, "Exercise-induced ST depression (mm)": 2.0,
    "Sex_0": 0, "Sex_1": 1, "CP_0": 0, "CP_1": 0, "CP_2": 0, "CP_3": 1, "FBS_0": 1,
    "Normal resting ECG": 0, "ST-T abnormality on resting ECG": 0, "LVH on resting ECG": 1,
    "No exercise-induced angina": 0, "Exercise-induced angina": 1,
    "Upsloping ST segment": 0, "Horizontal ST segment": 1, "Downsloping ST segment": 0,
    "CA_0": 0, "CA_1": 0, "CA_2": 1,
    "Fixed perfusion defect": 0, "Normal perfusion test": 0, "Reversible perfusion defect": 1,
}


@pytest.mark.parametrize("endpoint", cohorts.ENDPOINTS)
def test_example_rows_match_training_columns(endpoint):
    row = to_model_frame(endpoint, EXAMPLES[endpoint])
    assert list(row.columns) == cohorts.load_columns(endpoint)
    assert row.shape == (1, len(SCHEMAS[endpoint]))
    assert not row.isna().any().any()


@pytest.mark.parametrize("endpoint", cohorts.ENDPOINTS)
def test_codes_agree_with_cohort_coding(endpoint):
    """Every categorical answer maps to a code that occurs in the training cohort."""
    cohort = cohorts.load(endpoint, "web")
    row = to_model_frame(endpoint, EXAMPLES[endpoint])
    for c, t in cohort.spec.items():
        if t != "num":
            assert row[c].iloc[0] in set(cohort.X[c].dropna().unique())


def test_onehot_heart_form_equals_named_payload():
    a = to_model_frame("heart", heart_from_onehot(HEART_FORM))
    b = to_model_frame("heart", EXAMPLES["heart"])
    assert a.equals(b)


def test_zero_marks_unmeasured_values_like_the_training_data():
    row = to_model_frame("diabetes", {**EXAMPLES["diabetes"], "insulin": 0, "skin_thickness": 0})
    assert np.isnan(row["insulin"].iloc[0]) and np.isnan(row["skin"].iloc[0])


def test_ag_ratio_is_derived_when_omitted():
    payload = {k: v for k, v in EXAMPLES["liver"].items() if k != "ag_ratio"}
    row = to_model_frame("liver", payload)
    assert row["ag_ratio"].iloc[0] == pytest.approx(2.8 / (6.2 - 2.8))


@pytest.mark.parametrize("bad", [{"sex": "other"}, {"age": -4}, {"unknown_field": 1}])
def test_invalid_input_is_rejected(bad):
    with pytest.raises(ValueError):
        to_model_frame("heart", {**EXAMPLES["heart"], **bad})


def test_onehot_group_must_have_exactly_one_choice():
    with pytest.raises(ValueError):
        heart_from_onehot({**HEART_FORM, "CP_0": 1})
