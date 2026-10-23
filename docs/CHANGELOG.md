# Changelog

Running log of what changed in the project and why. Newest first.

## 2026-10-23: SweepScreen pipeline built end to end; systems paper drafted

* Author redirected the work: stop the field-study and protocol work, build the complete pipeline,
  and write a technical systems paper. `docs/protocol.md` and `docs/niche_decision.md` are kept as
  planning history; the field-study design moved to a short appendix of the paper ("deployment
  pathway").
* Added `docs/data_request.md`: the exact datasets (BUSI, BUS-BRA, BrEaST, UDIAT B, BreastVid; BUS-UCLM
  recommended) and weights (DINOv2 ViT-S/14). TabPFN is no longer used: the malignancy head is a
  logistic regression.
* Built `sweepscreen/`:
  * data layer: five loaders, schema, pHash dedup and groups, LOSO and grouped splits, manifest;
  * preprocessing: overlay inpainting, FOV crop, normalisation, letterbox, seven handheld
    degradations;
  * quality gate, frozen encoder, patch-level detector, malignancy head;
  * temporal aggregation;
  * calibration and exact-binomial / conformal threshold certification;
  * evaluation metrics (FROC, AP, bootstrap);
  * a shared training path used by every fold and by the deployed model, and the inference pipeline.
* Built `experiments/`: features (clean + augmented), LOSO/internal, augmentation ablation, video
  aggregation, robustness, quality gate, certification, runtime, final training with registry,
  and a report generator that writes every table, figure and number macro.
* Built `service/`: FastAPI, health-worker web page, JSONL inference log, `Dockerfile`.
* Added `scripts/reproduce.sh` and `scripts/check_paper_numbers.py`.
* Tests: 60+ covering every module, the service, and a smoke run of every experiment script on
  synthetic fixture files laid out like each dataset (fixtures are never used for results).
* Paper (`paper/`) rewritten as a systems paper. Results are macros only; it compiles against
  fixture outputs in a scratch directory. Real numbers are produced when the data are present.
* Not yet done, and blocked: no dataset or encoder weight could be downloaded in this environment,
  so no real result exists yet. The Docker image could not be built here (no Docker daemon).

## 2026-10-22: Author decisions D1–D6 applied (protocol v0.11, memo v3); project renamed SweepScreen

* **Renamed to SweepScreen** (working name; trademark check pending). The five-disease PredictX-HARBOR
  project (library, experiments, results, service, tests, paper, trained artifacts, data) was moved
  unchanged with `git mv` to `legacy/predictx-harbor/`, keeping history. Its 30 non-TabPFN tests still
  pass from there. New top-level `README.md` and `pyproject.toml` describe SweepScreen only.
* Protocol v0.11:
  * D1: sensitivity > 0.80 with 150 cancers is labelled a pilot claim; > 0.85 is deferred to a pivotal study.
  * D2: primary operating point is AI-added recall ≤ 5 points; ≤ 10 points is secondary.
  * D3: confirmatory test replaced by an intersection-union test (added sensitivity > 5 points AND
    added recall < 5 points), because CBE + AI ⊇ CBE makes a McNemar test trivially positive. Power
    re-computed with the new `scripts/sample_size_iut.py`: ~0.86 at +12 points / 3.5% added recall,
    ~0.65 at +10 points.
  * D4: enriched arm needs ≥ 40% non-palpable or ≤ 2 cm imaging-detected cancers, reported as a
    separate stratum; CBE-negative cancers are the key subgroup.
  * D5: certification tables now show 0 / 1 / 2 misses.
  * D6: HARBOR and the VoI controller removed entirely, including the exploratory HARBOR test.
  * The earlier "pilot" was renamed **run-in** and enlarged to ≥ 500 women, so that it can certify the
    5-point cap with headroom (certified only if observed ≤ 3.2%).
* Phase 0 audit: path note added (the audited files now live under `legacy/predictx-harbor/`).

## 2026-10-22: Scope change to first-step screening (protocol v0.10, memo v2)

* Author changed the use case from lesion triage to **screening asymptomatic women**: a health worker
  does a handheld whole-breast sweep with or after CBE, and the AI outputs NORMAL / RECALL / UNREADABLE
  per woman.
* `docs/protocol.md` rewritten as v0.10: lesion detection (frame detector plus persistence-based
  woman-level aggregation) is the core, with a quality/coverage reject gate. Primary endpoints are
  at screening prevalence: sensitivity at recall ≤ 5% and ≤ 10%, false recalls per 1,000, and
  unreadable rate. The confirmatory comparison is CBE + AI vs CBE alone (paired McNemar). C1 is
  adapted to a woman-level recall ceiling and sensitivity floor, plus spectrum weighting (new) and
  hierarchical borrowing (H1).
* Added an enriched field-study design (~3,000 camp women + ~150 cancers + ~150 benign, preceded by
  a ~300 + 60 pilot), with exact binomial sample sizes from the new
  `scripts/sample_size_field_study.py` (output in `docs/sample_size_output.txt`).
* Literature search: no public whole-breast sweep or screening ultrasound dataset exists. Public
  still images are declared development-only.
* `docs/niche_decision.md` v2: India positioning (CBE/NP-NCD, iBreastExam, Thermalytix,
  mammography, AI ultrasound), data reality, and risks, including **PC-PNDT Act restrictions on
  portable ultrasound** (high severity).
* HARBOR moved out of the primary pipeline (one exploratory field-data test retained); the VoI
  controller is not used.

## 2026-10-22: Track A chosen; Phase 2 protocol drafted

* Author chose Track A and asked for a novel contribution in a design that can support a screening
  device, not "just another triage classifier".
* Added `docs/protocol.md` v0.9 (draft): two-stage device pipeline (S1 lesion detection, then S2
  malignancy referral). Main contribution C1: certified-sensitivity referral under site shift
  (exact local certification, source-weighted conformal, and a hierarchical borrowing-strength
  hypothesis that would reduce the number of local cancers needed). Leave-one-site-out across
  countries is the main evidence. One confirmatory comparison is prespecified.
* Dropped the value-of-information controller from the protocol, because Phase 0 showed it did not
  beat fixed-panel mixtures, and these data have no sequential test decision for it to optimise.
* Added `docs/deviations.md` (empty until the protocol is frozen after the label-blind data audit).
* Corrected a draft sample-size figure before commit (an estimated 180 cancers replaced by the exact
  binomial 100 / 240).

## 2026-10-22: Phase 0 (audit and reproduction) and Phase 1 (niche decision memo)

* Added `docs/phase0_audit.md`: environment check, reproduction of the stored results, and a
  verdict on each weakness raised in the review, checked against code and `results/`.
* Added `docs/niche_decision.md`: comparison of tracks A–D and a recommendation. Awaiting the
  author's decision before any build (Phase 2 onward).
* No library, experiment, result or paper file was changed. All re-runs were written to a scratch
  directory outside the repository so the stored results stay untouched.
