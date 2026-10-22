# SweepScreen pre-registered protocol: AI-read handheld whole-breast ultrasound for first-step screening of asymptomatic women

**Version 0.11 (DRAFT, not frozen; for author review).** Incorporates the author's decisions of
2026-10-22 (Section 0). It becomes v1.0, frozen and tagged `protocol-v1.0`, only after author
approval and the label-blind data audit (Section 5). Nothing is evaluated on a held-out site
before that tag exists. Every later change goes in `docs/deviations.md`.

Reporting: TRIPOD+AI (2024), CLAIM (2024 update), STARD 2015; for field studies also ICMR's 2023
ethical guidelines for AI in health.

Sample-size numbers come from `scripts/sample_size_field_study.py` and `scripts/sample_size_iut.py`
(outputs in `docs/sample_size_output.txt` and `docs/sample_size_iut_output.txt`).

---

## 0. Decisions fixed by the author (2026-10-22)

| # | Decision |
|---|---|
| D1 | Sensitivity target **> 0.80 with 150 cancers**, labelled a **pilot** accuracy study. A claim of > 0.85 is deferred to a **pivotal** study. |
| D2 | Primary operating point: **AI-added recall ≤ 5 percentage points** (OP-A5). Secondary: ≤ 10 points (OP-A10). |
| D3 | Confirmatory comparison: **CBE + AI vs CBE alone at capped added recall**, with a test that accounts for the recall cost, because CBE + AI ⊇ CBE (Section 7.2). |
| D4 | Enriched cancer arm: **≥ 40% non-palpable or ≤ 2 cm cancers, recruited from imaging-detected cases**, reported as a separate stratum. **CBE-negative cancers are the key subgroup.** |
| D5 | Certification figures state the number of misses they assume, with bounds for one or two misses (Section 4.3). |
| D6 | HARBOR and the value-of-information controller are removed. Project renamed **SweepScreen**; the five-disease framing is dropped (the old project is archived in `legacy/predictx-harbor/`). |

## 1. Intended use

**Setting.** Community screening of asymptomatic women aged 30–65 (the NP-NCD target group) at
screening camps, sub-centres or Ayushman Arogya Mandirs, with no mammography and no radiologist on
site.

**Workflow.** A trained health worker performs CBE, then a handheld ultrasound sweep of both breasts
following a fixed protocol (quadrant or radial sweeps plus the axillary tail; finalised in the
run-in, Section 8.2). The AI returns one result per woman:

| Output | Meaning | Action |
|---|---|---|
| **NORMAL** | No suspicious lesion detected in an adequate scan | Routine re-screening per programme interval |
| **RECALL** | Persistent suspicious lesion detected | Referral to a facility with diagnostic ultrasound and a radiologist (± biopsy) |
| **UNREADABLE** | Scan quality or coverage inadequate | Immediate rescan; after 2 failed rescans → counted as RECALL |

A positive CBE is always referred. The AI can add recalls but never cancels a CBE referral, so the
combined rule recalls **CBE-positive OR AI-RECALL**.

**Components.**
1. **Q, quality and coverage gate:** per-frame adequacy (gain, contact, depth, motion) and per-breast sweep coverage → UNREADABLE.
2. **D, frame-level lesion detector (core model):** probability that a frame contains a suspicious focal lesion, with a box.
3. **A, woman-level aggregation:** sliding-window mean over *w* consecutive frames → maximum over frames per breast → maximum over breasts → threshold.
4. **C1, certified operating point:** the threshold carries a stated added-recall ceiling and sensitivity floor at a new site (Section 4).

Benign-vs-malignant characterisation is not part of the output.

## 2. Claim ceiling

| Question | Source of evidence | Allowed claim |
|---|---|---|
| Does the detector find lesions in still images from other countries? | Public still images, leave-one-site-out | Development evidence only |
| Screening sensitivity, added recall, false recalls per 1,000 | **Field studies only** (Section 8) | Public data **cannot** measure these. No such figure will come from public data. |
| Sensitivity > 0.80 | Pilot accuracy study (150 cancers) | "Pilot evidence" |
| Sensitivity > 0.85 | Pivotal study (not designed in detail here) | Deferred |
| Health-worker operation on low-cost handhelds | Field studies | Not before them |
| Clinical validation or regulatory approval | — | Never claimed by this project |

## 3. Public data: development only

### 3.1 Search result (2026-10-22)
Literature, arXiv and dataset atlases (including github.com/TIanCat/open-medical-ultrasound-datasets)
contain **no public whole-breast handheld-sweep dataset and no public asymptomatic screening breast
ultrasound dataset.**

| Dataset | Content | Normal women/frames? | Use |
|---|---|---|---|
| BreastVid / CVA-Net (MICCAI 2022, China) | 188 lesion-centred clips (113 malignant, 75 benign), 25,272 frames with boxes | **No** (clips start and end at lesion edges) | Temporal aggregation; frame detection sensitivity |
| TDSC-ABUS 2023 (China) | 200 automated 3D volumes, tumour-focused | No | Optional (by application) |
| BUSI (Egypt) | 780 stills incl. 133 normal | Yes (stills) | Detector development |
| BUS-UCLM (Spain) | 683 stills incl. 419 normal; 38 patients | Yes (stills) | Detector development |
| BUS-BRA (Brazil), BrEaST (Poland), BUS-CoT (China), BUSI_WHU, BUS_UC, US3M, QAMEBI | Lesion stills with masks or boxes | Few or none | Lesion-positive frames |

**Statement for the paper:** *Public breast-ultrasound datasets consist of curated still images or
lesion-centred clips, acquired by specialists on mid- or high-end systems in women already referred
for imaging. They contain almost no normal tissue from asymptomatic women and no whole-breast sweeps.
They are used to develop and stress-test the lesion detector. They cannot measure screening
sensitivity, recall rate or false recalls, and no such figure in this work comes from them.*

### 3.2 Development experiments (pre-registered; secondary evidence)
* **P-D1, frame detection, leave-one-site-out (LOSO)** on sites with both normal and lesion frames
  (BUSI, BUS-UCLM; BUS-CoT if it contains normals). Endpoints: frame AUROC (lesion vs normal);
  malignant-frame sensitivity at a normal-frame false-positive rate of 1%, 2% and 5%; per site, with
  patient-clustered bootstrap CIs.
* **P-D2, clip-level aggregation (BreastVid):** detection rate of malignant and benign clips under
  each aggregation rule, at the P-D1 thresholds. Sensitivity only; false-positive behaviour can't be
  estimated without normal sweeps.
* **P-D3, quality gate:** sensitivity under the degradations of Section 10, and the rate at which Q
  flags them.
* **P-D4, C1 proxy evaluation** (Section 4.4).

Detector: frozen encoders on CPU (DINOv2 ViT-B/14; BiomedCLIP ViT-B/16) with linear or patch-level
heads. A fine-tuned detector is an optional, pre-specified secondary analysis if a GPU becomes
available. The encoder and head are chosen by internal CV within training sites only. Comparators:
linear probe per encoder; TabPFN and CatBoost on embedding PCA (does a stronger tabular head add
anything?); a detector trained on a single site.

Duplicate control: pHash (64-bit, Hamming ≤ 6) plus SSIM ≥ 0.95, within and across datasets.
Clusters are kept inside one split; label-discordant clusters are excluded; cross-site duplicates are
removed from the training side. BUSI is audited for its documented defects (duplicates, axillary
images, visible needles); the curated release is preferred.

## 4. Woman-level decision and the methodological contribution (C1)

### 4.1 Aggregation
Sliding-window mean over *w* ∈ {1, 3, 5, 9} frames, chosen in the run-in. A lesion must persist
across *w* consecutive frames to count. Breast score = maximum over frames; woman score = maximum
over breasts.

### 4.2 Operating points (defined on the added-recall scale)
**Added recall** = proportion of cancer-free screened women who are **CBE-negative and AI-RECALL**
(UNREADABLE after rescans counts as RECALL).
* **OP-A5 (primary):** added recall ≤ 5 percentage points (≤ 50 extra recalls per 1,000 women).
* **OP-A10 (secondary):** added recall ≤ 10 percentage points.

### 4.3 C1: certified operating point at a new site
* **C1a, added-recall ceiling (exact).** The lowest threshold whose one-sided 95% Clopper–Pearson
  upper bound on added recall among *m* local cancer-free women is < the cap. For the 5-point cap:

  | Run-in women *m* | Certified if AI-only recalls ≤ | (observed added recall ≤) |
  |---|---|---|
  | 300 | 8 | 2.7% |
  | 400 | 12 | 3.0% |
  | **500** | **16** | **3.2%** |
  | 600 | 20 | 3.3% |

* **C1b, sensitivity floor (exact).** The one-sided 95% Clopper–Pearson lower bound on sensitivity
  from *n* local cancers. **The commonly quoted 29 and 59 cancers assume zero misses.** Bounds by
  number of misses:

  | Local cancers *n* | 0 misses | 1 miss | 2 misses |
  |---|---|---|---|
  | 29 | 0.902 | 0.847 | 0.798 |
  | 46 | 0.937 | 0.901 | 0.869 |
  | 59 | 0.950 | 0.922 | 0.897 |
  | 93 | 0.968 | 0.950 | 0.934 |
  | 124 | 0.976 | 0.962 | 0.950 |
  | 149 | 0.980 | 0.969 | 0.958 |

  Equivalently, certifying ≥ 0.90 needs 29 / 46 / 61 cancers with 0 / 1 / 2 misses, and certifying
  ≥ 0.95 needs 59 / 93 / 124. A screening detector at its target operating point is expected to miss
  some cancers, so the 1- and 2-miss columns are the realistic ones.
* **C1c, spectrum weighting (new).** Local cancers are usually larger and more palpable than
  screen-detected ones. Sensitivity is re-weighted to a prespecified target spectrum (size ≤ 10 /
  11–20 / > 20 mm × palpable yes/no; target proportions taken from published screen-detected series in
  comparable settings and fixed at freeze), with a Kish effective-sample-size Clopper–Pearson bound.
* **C1d, borrowing strength across sites (new; hypothesis H1).** A hierarchical beta-binomial model
  of site-level sensitivity and added recall, updated with local data. **H1:** at matched achieved
  coverage, C1d needs about half the local cancers that C1b needs.

### 4.4 Evaluation of C1
* **Public data (proxy, lesion level; exploratory):** LOSO; the normal-frame false-positive rate
  stands in for added recall, and malignant-lesion detection for sensitivity. Within each held-out
  site, 200 repeated certification draws (*n* ∈ {10, 15, 20, 30, 59} cancers; *m* normals as
  available), evaluated on the disjoint remainder. Endpoint: achieved coverage against the nominal
  95%.
* **Woman level:** the run-in certifies OP-A5 and OP-A10. The pilot accuracy study tests whether the
  certified ceiling and floor held (Section 7.3).

## 5. Label-blind data audit (public data; before freezing)
Allowed: checksums, counts, metadata fields and missingness, verification method, label counts,
duplicate detection, caliper and annotation detection, pixel spacing. Not allowed: training any model
evaluated on a held-out site, or viewing held-out performance. Output: `docs/data_audit.md`.

## 6. Comparators
* **Primary (field):** CBE by the same health worker in the same woman (paired). This is the NP-NCD
  standard of care.
* **Secondary (field, where feasible):** an expert radiologist reading the same stored sweep (an upper
  bound on what the sweep contains); live radiologist diagnostic ultrasound (part of the reference
  standard).
* **Positioning only (not tested here):** iBreastExam (tactile), Thermalytix (thermal), mammography.

## 7. Endpoints and the confirmatory test (pilot accuracy study)

### 7.1 Primary and key secondary endpoints
1. **Sensitivity of CBE + AI at OP-A5** (woman level; cancer diagnosed within 12 months), on the
   enriched cancer arm. Target claim: > 0.80 (D1).
2. **Added recall** and **false recalls per 1,000 women** at OP-A5, in the screening-camp arm.
3. **Confirmatory test**, Section 7.2.
4. **Key subgroup: CBE-negative cancers.** AI sensitivity among cancers missed by CBE. This is the
   quantity the device exists to deliver (added sensitivity = this × (1 − CBE sensitivity)).
5. **Stratum S (≥ 40% of the enriched arm): non-palpable or ≤ 2 cm, imaging-detected cancers.** All
   sensitivity endpoints are reported separately for S and for the rest. Precision with *n* = 60 in
   S: observed 0.70 → 95% CI 0.568–0.812; 0.80 → 0.677–0.892.
6. UNREADABLE rate (before and after rescan); extra recalls per extra cancer detected at 2 and 5 per
   1,000 (at 4% added recall and +10 to +15 points added sensitivity, that is 53–200 extra recalls per
   extra cancer; this cost has to be reported alongside the benefit); OP-A10 versions of 1–5;
   spectrum-weighted versions (C1c).

### 7.2 Confirmatory comparison: intersection-union test (IUT)
Because CBE + AI recalls every woman CBE recalls, **sensitivity cannot decrease** and a McNemar-type
test passes whenever the AI finds even one extra cancer. That says nothing about whether the gain is
worth its recall cost. Both of the following must hold:

* **H1a (benefit):** added sensitivity Δ_sens = P(CBE−, AI+ | cancer) **> δ = 0.05.** Exact one-sided
  97.5% Clopper–Pearson lower bound on the enriched cancer arm (all 150) > 0.05.
* **H1b (cost):** added recall Δ_rec = P(CBE−, AI+ | cancer-free) **< 0.05.** Exact one-sided 97.5%
  Clopper–Pearson upper bound on camp women < 0.05.

The IUT controls the type I error at 2.5% one-sided without multiplicity adjustment. The two samples
are independent, so power is the product. Margin δ = 0.05 means at least 5 extra cancers found per 100
cancers screened.

**Power** (150 cancers, 2,985 cancer-free camp women):

| True Δ_sens | Δ_rec 3.0% | Δ_rec 3.5% | Δ_rec 4.0% |
|---|---|---|---|
| +0.10 | 0.65 | 0.63 | 0.47 |
| **+0.12** | **0.87** | **0.86** | 0.63 |
| +0.15 | 0.985 | 0.97 | 0.71 |
| +0.20 | 1.00 | 0.98 | 0.72 |

Component sizes: H1a needs 131 / 175 cancers (80% / 90% power) at a true Δ_sens of +0.12, 234 / 301
at +0.10, and 75 / 100 at +0.15. H1b needs about 1,500 / 2,050 cancer-free women at a true Δ_rec of
3.5%, and about 3,600 / 4,650 at 4.0%. With δ = 0 (no margin), H1a has power 1.00 in every scenario,
which is why a margin is required.

**Design assumption and its consequence.** The study is powered (~0.86) for a true Δ_sens ≥ +0.12
and Δ_rec ≤ 3.5%. The run-in (500 women) certifies OP-A5 only if observed added recall is ≤ 3.2%,
which keeps the true value in the powered range. **If the AI's true added sensitivity is only about
+0.10, 150 cancers give 65% power. Either accept that, or raise the enriched arm to about 230
cancers.** That is the author's call, and the draft keeps 150 (D1).

### 7.3 C1 validity checks (pre-registered, descriptive)
Whether the run-in-certified added-recall ceiling and sensitivity floor held in the pilot data.

## 8. Field programme (designed here; not conducted by this project)

### 8.1 Phase I: run-in (threshold calibration)
**≥ 500 screening-camp women + ~60 enriched cancers**, scanned by the same health-worker cadre on the
same device model. Fixes the sweep protocol, *w*, the Q thresholds and the C1-certified OP-A5 and
OP-A10 thresholds. Run-in participants are never part of Phase II.

### 8.2 Phase II: pilot accuracy study (this protocol's target)
Prospective, multicentre, paired. Two arms enrolled in parallel at the same sites, by the same health
workers, on the same device model:
* **Screening-camp arm:** ~3,000 consecutive asymptomatic women aged 30–65 at NP-NCD camps → added
  recall, false recalls per 1,000, UNREADABLE rate, and the few camp-detected cancers (6–15 expected at
  2–5 per 1,000; P(≥ 10 cancers) is 0.08 at 2/1,000 and 0.93 at 5/1,000).
* **Enriched cancer arm:** 150 women with biopsy-confirmed cancer before treatment, from partner
  cancer centres, **with ≥ 60 (≥ 40%) non-palpable or ≤ 2 cm cancers recruited from imaging-detected
  cases** (screening or diagnostic mammography, or ultrasound-detected). Plus ~150 women with
  biopsy-proven benign lesions (to measure how often benign lesions trigger RECALL). Health workers
  are blinded to arm where practical (mixed sessions).

### 8.3 Phase III: pivotal study (deferred)
For a claim of sensitivity > 0.85, 185 cancers (true 0.92) to 379 cancers (true 0.90) at 80% power,
and correspondingly more camp women if the specificity bound is tightened (specificity > 0.94 with
true 0.95 needs ~4,225 cancer-free women). Designed after Phase II.

### 8.4 Reference standard and verification (Phases I–II)
* Every RECALL (AI or CBE): radiologist diagnostic ultrasound ± biopsy within 30 days.
* Every camp woman: 12-month follow-up for interval cancer (contact plus registry and hospital linkage).
* Verification sample: a random 10% of CBE-negative/AI-NORMAL camp women plus all UNREADABLE women get
  a radiologist ultrasound at the visit. Begg–Greenes correction for partial verification.
* Blinded expert reading of all stored sweeps.

### 8.5 Other sample-size facts (exact binomial)
Sensitivity with 150 cancers: true 0.90 → observed 135/150, 95% CI 0.840–0.943; power for a lower
bound > 0.80 at true 0.90 is 0.93. False recalls with 2,985 cancer-free women: true 5% → 50 per
1,000 (95% CI 42–58); true 3% → 30 (24–37).

### 8.6 Regulatory and ethics (preconditions for Phase I)
* **PC-PNDT Act:** ultrasound machines must be registered to specific premises, portable use is
  restricted (for example, Delhi guidance permits it only within registered premises), and operators
  are regulated. Health-worker use at camps needs legal advice and state PC-PNDT authority approval
  (registered mobile units, registered facilities, or an exemption).
* CDSCO: AI screening software is a medical device (SaMD) under the Medical Device Rules 2017, so
  Phases I–II are clinical performance investigations needing CDSCO and ethics approval.
* ICMR AI-in-health guidelines (2023); DPDP Act 2023; informed consent in local languages; CTRI
  registration before enrolment.

## 9. Statistics (public development)
Patient-clustered bootstrap (2,000 replicates) per site; Clopper–Pearson at fixed thresholds;
random-effects meta-analysis across LOSO sites (Hartung–Knapp); DeLong for paired AUROC within a
held-out site; 5×5 grouped CV with the Nadeau–Bengio correction within training sites. No
significance stars on exploratory results.

## 10. Robustness: portable-device degradations (public data)
Speckle (multiplicative Rayleigh, σ ∈ {0.1, 0.2, 0.3}); blur (σ ∈ {1, 2} px); downsampling (×0.5,
×0.25); JPEG {50, 20}; gain/contrast ±30%; depth crop 25%; partial probe contact (dark sector).
Endpoints: sensitivity at the frozen frame threshold; the fraction of degraded frames that Q flags.

## 11. Reproducibility
`scripts/reproduce.sh` (to be written for SweepScreen) regenerates every number from raw files whose
SHA-256 checksums are in `data/MANIFEST.sha256`. Held-out evaluation scripts refuse to run unless the
`protocol-v1.0` tag exists and this file matches it. Seeds are fixed.
