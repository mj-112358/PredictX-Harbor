# PredictX-HARBOR

**Built with PriorLabs-TabPFN.**

Multi-disease risk prediction from routine clinical biomarkers: coronary disease, diabetes, breast
malignancy, liver disease and chronic kidney disease. This repository contains the library, every
experiment reported in the paper, the trained web-form models and the prediction service.

## Installation

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.13. TabPFN v2 downloads its public weights from Hugging Face on first use.

## Repository layout

| Path | Contents |
|---|---|
| `predictx/cohorts.py` | Loaders for the five cohorts; `full` and `web` feature sets |
| `predictx/preprocessing.py` | Fold-safe tabular preprocessing and the HARBOR token encoder |
| `predictx/harbor.py` | HARBOR encoder (JAX), its ablation variants and the deep-ensemble classifier |
| `predictx/learners.py` | Literature-standard comparators, CatBoost, tuned XGBoost, TabPFN v2 |
| `predictx/ensemble.py` | PredictX ensemble (equal-weight average of TabPFN, CatBoost, LR, HARBOR) |
| `predictx/acquisition.py` | Value-of-information test-acquisition controller |
| `predictx/evaluation.py` | Resampling protocols and the Nadeau and Bengio corrected resampled t-test |
| `predictx/website.py` | Schemas of the five web forms and conversion to model inputs |
| `experiments/` | Scripts that produce every result, table and figure |
| `service/` | Training of the deployed models and the HTTP service |
| `tests/` | Unit and end-to-end tests |
| `results/` | Per-split out-of-sample predictions of every model |
| `paper/` | LaTeX source of the manuscript |
| `data/` | The public cohorts |
| `licenses/` | Third-party model licence (TabPFN v2) |

## Reproducing the paper

```bash
scripts/reproduce.sh
```

The script runs every experiment (`experiments/benchmark.py` for protocols P1 and P2 and the web feature sets,
`experiments/acquisition.py` for P3, `experiments/missingness.py` for P4), regenerates all tables, figures and
quoted numbers with `experiments/report.py`, trains the deployed models, runs the tests and builds
`paper/main.pdf`. Prediction files are written per model
(`results/predictions/<cohort>-<features>-<protocol>/<model>.csv`) and completed splits are skipped, so runs can
be interrupted and resumed. Library versions are pinned in `requirements.txt`; the grouped cross-validation
splits of the breast cohort depend on the scikit-learn version.

## Prediction service

```bash
python -m service.train_models          # fits the five deployed ensembles and the coronary test controller -> artifacts/
uvicorn service.app:app --port 8000
```

```bash
curl -X POST localhost:8000/predict/diabetes -H 'Content-Type: application/json' \
     -d '{"pregnancies": 2, "glucose": 150, "diastolic_bp": 70, "skin_thickness": 30,
          "insulin": 120, "bmi": 31.0, "diabetes_pedigree": 0.5, "age": 45}'
# {"endpoint": "diabetes", "risk_score": 0.649, "risk_percent": 64.9, "model_version": "720cc3c78f1d",
#  "attribution": "Built with PriorLabs-TabPFN"}
```

| Route | Description |
|---|---|
| `GET /health` | Status and loaded endpoints |
| `GET /schema/{endpoint}` | Field names, labels, units, ranges and options of a form |
| `POST /predict/{endpoint}` | Risk score for `heart`, `diabetes`, `breast`, `liver` or `kidney` |
| `POST /predict/heart/onehot` | Risk score from the one-hot coronary form used on the website |
| `POST /acquire/heart?lam=0` | Risk score and net value of information of every coronary test not yet performed |

Every prediction is appended to a JSON-lines inference record (`PREDICTX_RECORDS`, default
`records/inference.jsonl`) with the request, the model version (SHA-256 prefix of the artifact) and the score.

## Tests

```bash
python -m pytest
```

The suite checks the propositions of the paper (zero gradient of the L1 attention penalty, entropy descent,
proper-score inequality of the ensemble), mask integrity of every HARBOR variant, the resampling protocols,
form-to-cohort coding for all five endpoints, and end-to-end service requests (risk scores, monotonic response to
a dominant marker, inference records and test recommendations).

## Licences

The PredictX ensemble includes TabPFN v2, whose weights are distributed under the Prior Labs License 1.1
(`licenses/TabPFN-LICENSE.txt`). Section 10 of that licence requires any website or user interface that serves
these models to display **"Built with PriorLabs-TabPFN"**; the service returns this string in every response
(`attribution`).
