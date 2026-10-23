# SweepScreen

AI reading of **handheld whole-breast ultrasound sweeps** for community breast screening. A health
worker performs clinical breast examination and sweeps both breasts with a handheld probe; SweepScreen
returns one result per woman: **NORMAL**, **RECALL** or **UNREADABLE** (rescan), with the key frame,
lesion box, heatmap and a malignancy suspicion score. Everything runs on a CPU.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Data layer | `sweepscreen/data` | Loaders for BUSI, BUS-BRA, BrEaST, UDIAT Dataset B and BreastVid into one schema; perceptual-hash deduplication; patient/video/duplicate groups; grouped and leave-one-site-out splits; SHA-256 manifest |
| Preprocessing | `sweepscreen/preprocess` | Overlay and caliper inpainting, field-of-view crop, intensity normalisation, letterboxing; seven handheld-probe degradations at three levels |
| Q, quality gate | `sweepscreen/quality` | Per-frame adequacy classifier (contact, gain, blank, blur, reverberation) and per-breast coverage rule |
| Encoder | `sweepscreen/models/encoder.py` | Frozen DINOv2 ViT-S/14: CLS token and 16 x 16 patch tokens |
| D, lesion detector | `sweepscreen/models/detector.py` | Convolutional patch head → frame lesion probability, boxes, heatmap |
| M, malignancy head | `sweepscreen/models/malignancy.py` | CLS + ROI tokens (+ age) → suspicion score |
| A, aggregation | `sweepscreen/aggregate` | Persistence over consecutive adequate frames → breast → woman decision |
| Safety layer | `sweepscreen/safety` | Platt calibration; exact-binomial and conformal threshold certification at a new site |
| Evaluation | `sweepscreen/evaluation`, `experiments/` | LOSO and internal CV, FROC, AP, AUROC, calibration, robustness, quality gate, video aggregation, certification, runtime |
| Service | `service/` | FastAPI (`/analyze/image`, `/analyze/sweep`), health-worker web page, model registry, JSONL inference log, `Dockerfile` |

## Data and weights

`docs/data_request.md` lists the exact files, versions and licences. Place them under `data/raw/<dataset>/`
and `models/pretrained/dinov2-small/`.

## Run

```bash
python -m pip install -r requirements.txt
scripts/reproduce.sh            # every experiment, the deployed model, all tables/figures/numbers, tests, paper PDF
uvicorn service.app:app --port 8000     # then open http://localhost:8000
```

Individual steps: `python -m experiments.run_features`, `run_loso`, `run_video`, `run_quality`,
`run_certify`, `run_robustness`, `train_final`, `run_runtime`, `report` (all under `experiments`).

## Paper

`paper/` holds the systems paper. Every number in it is a macro written by `experiments/report.py`
into `paper/generated/macros.tex`; `scripts/check_paper_numbers.py` fails if the text uses a number
that the code did not produce.

## Tests

```bash
python -m pytest
```

The tests cover each module on small synthetic files laid out like each dataset, plus the service
routes and a smoke run of every experiment script.

## Repository history

`legacy/predictx-harbor/` is the earlier five-disease tabular project, kept for provenance.
`docs/phase0_audit.md` records its audit. `docs/niche_decision.md` and `docs/protocol.md` are the
earlier planning documents for the screening use case.
