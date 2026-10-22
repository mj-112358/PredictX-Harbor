# SweepScreen

*Working name, pending a trademark check.*

AI reading of **handheld whole-breast ultrasound sweeps** for **first-step breast cancer screening
of asymptomatic women** in settings without mammography. A health worker performs clinical breast
examination (CBE) and a handheld sweep of both breasts; the software returns one result per woman:
**NORMAL**, **RECALL** or **UNREADABLE** (rescan).

**Status: protocol stage. Not a medical device, not clinically validated, not for clinical use.**

## Where things stand

| Document | Contents |
|---|---|
| `docs/protocol.md` | Pre-registered protocol (draft until tagged `protocol-v1.0`): pipeline, endpoints, the confirmatory test, the field-study design and sample sizes |
| `docs/niche_decision.md` | Why this use case, positioning against screening tools available in India, data reality, risks |
| `docs/phase0_audit.md` | Audit of the earlier PredictX-HARBOR project that this work replaces |
| `docs/deviations.md` | Protocol deviations after freezing |
| `docs/CHANGELOG.md` | What changed and why |
| `scripts/sample_size_field_study.py`, `scripts/sample_size_iut.py` | Exact binomial calculations behind every sample size in the protocol (outputs in `docs/sample_size_*output.txt`) |
| `legacy/predictx-harbor/` | The archived five-disease tabular project (not part of SweepScreen) |

## What the evidence will and will not show

Public breast-ultrasound datasets are curated still images or lesion-centred clips from women
already referred for imaging. They are used here **for development only**, and cannot measure
screening sensitivity, recall rate or false recalls. Those come only from the field study described
in the protocol.

## Setup

```bash
python -m pip install -r requirements.txt
python scripts/sample_size_iut.py
```
