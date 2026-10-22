# Pre-registered protocol: AI-read handheld whole-breast ultrasound for first-step screening of asymptomatic women

**Version 0.10 (DRAFT, not frozen; for author review).** Supersedes v0.9 (lesion triage), following
the author's change of scope on 2026-10-22. It becomes v1.0, frozen and tagged `protocol-v1.0`,
only after (a) the author approves this draft and (b) the label-blind data audit in Section 5.
Nothing is evaluated on a held-out site before that tag exists. Every later change goes in
`docs/deviations.md`.

Reporting: TRIPOD+AI (2024), CLAIM (2024 update), STARD 2015; for the field study also
STARD-AI when published, and ICMR's 2023 ethical guidelines for AI in biomedical research.

Every sample-size number below is produced by `scripts/sample_size_field_study.py`
(output: `docs/sample_size_output.txt`).

---

## 1. Intended use

**Setting.** Community screening of asymptomatic women aged 30–65 (the NP-NCD target group) at
screening camps, sub-centres or Ayushman Arogya Mandirs, where there is no mammography and no
radiologist on site.

**Workflow.** A trained health worker performs clinical breast examination (CBE) and a handheld
ultrasound whole-breast sweep of both breasts following a fixed protocol (radial or
quadrant-by-quadrant sweeps plus the axillary tail; protocol to be fixed in the pilot, Section
8.2). The AI reads the sweep and outputs one result per woman:

| Output | Meaning | Action |
|---|---|---|
| **NORMAL** | No suspicious lesion detected in an adequate scan | Routine re-screening per programme interval |
| **RECALL** | Suspicious lesion detected | Referral to a facility with diagnostic ultrasound and a radiologist (± biopsy) |
| **UNREADABLE** | Scan quality or coverage inadequate | Immediate rescan; after 2 failed rescans → treated as RECALL |

CBE is done in every woman. A positive CBE is referred regardless of the AI result (the AI never
overrides a positive CBE).

**Components.**
1. **Q, quality and coverage gate:** per-frame image adequacy (gain, contact, depth, motion) and per-breast coverage of the sweep. Produces UNREADABLE.
2. **D, frame-level lesion detector (the core model):** probability that a frame contains a suspicious focal lesion, with a localisation box.
3. **A, woman-level aggregation:** temporal smoothing over consecutive frames, then the maximum over frames within a breast, then the maximum over both breasts, then a threshold set for the target recall rate (Section 4).
4. **C1, certified operating point:** the recall threshold carries a stated sensitivity floor and recall ceiling at a new site (Section 4).

Malignancy characterisation (benign vs malignant) is **not** part of the screening output. It may
be used internally to rank recalls, but RECALL is triggered by *suspicious lesion presence*.

## 2. Claim ceiling

| Question | Answered by | Claim allowed |
|---|---|---|
| Can the detector find lesions in still images from other countries? | Public still images (Section 3), leave-one-site-out | "Lesion detection on curated still images", development evidence only |
| What is screening sensitivity, recall rate and false recalls per 1,000? | **Only the field study (Section 8)** | Public still-image data **cannot** measure these and will never be reported as screening accuracy. Projections to screening prevalence from public data appear only in a clearly labelled "projection" table, if at all. |
| Does the operating point hold at a new site? | C1 on public data (lesion level, proxy) → pilot → field study | Woman-level certification only from field data |
| Does it work on low-cost handheld probes used by health workers? | Field study (simulated degradations on public data are a robustness check only) | Only after the field study |
| Is it clinically validated or approved? | Neither this protocol nor the field study alone | Never claimed |

## 3. Public data: development only

### 3.1 Search result
A search on 2026-10-22 (peer-reviewed literature, arXiv, dataset atlases including
github.com/TIanCat/open-medical-ultrasound-datasets) found **no public whole-breast
handheld-sweep dataset and no public screening (asymptomatic) breast ultrasound dataset.** The closest
resources:

| Dataset | Content | Normal women/frames? | Use here |
|---|---|---|---|
| BreastVid / CVA-Net (MICCAI 2022, China) | 188 lesion-centred cine clips (113 malignant, 75 benign), 25,272 frames with per-frame boxes | **No.** Each clip starts and ends at the lesion edges | Temporal aggregation; per-frame detection sensitivity |
| TDSC-ABUS 2023 (China) | 200 automated 3D breast volumes, tumour-focused | No normal screening volumes | Optional (by application); 3D slices as pseudo-sweeps |
| BUSI (Egypt) | 780 stills incl. 133 normal | Yes (stills) | Detector training/validation |
| BUS-UCLM (Spain) | 683 stills incl. 419 normal, 38 patients | Yes (stills) | Detector training/validation |
| BUS-BRA (Brazil), BrEaST (Poland), BUS-CoT (China), BUSI_WHU, BUS_UC, US3M, QAMEBI | Lesion stills (with masks/boxes) | Few or none | Lesion-positive frames for the detector |
| ALN-Ultra (videos) | Axillary nodes | — | Not used (controlled access, wrong target) |

**Statement for the paper (verbatim intent):** *Public breast-ultrasound datasets consist of curated
still images or lesion-centred clips acquired by specialists, on mid- or high-end systems, in
women already referred for imaging. They contain almost no normal tissue from asymptomatic women
and no whole-breast sweeps. They can be used to develop and stress-test a lesion detector, but
they cannot measure screening sensitivity, recall rate or false recalls, and no such figure in this
paper comes from them.*

### 3.2 Development experiments on public data (pre-registered; secondary evidence)
* **P-D1, frame-level detection, leave-one-site-out (LOSO).** Sites with both normal and lesion
  frames (BUSI, BUS-UCLM; BUS-CoT if it contains normal frames). Lesion frames from all other sites
  are used in training. Endpoints: frame-level AUROC (lesion vs normal); sensitivity for
  **malignant** lesion frames at a fixed **normal-frame false-positive rate** of 1%, 2% and 5%.
  Both are reported per held-out site, with patient-clustered bootstrap CIs.
* **P-D2, clip-level aggregation (BreastVid).** Fraction of malignant and benign clips detected
  at the frame thresholds fixed in P-D1, for each aggregation rule (Section 4.1). Gives only
  *sensitivity*: with no normal clips, the false-positive behaviour of aggregation cannot be
  estimated from public data.
* **P-D3, quality gate.** Detection sensitivity under the portable-ultrasound degradations of
  Section 10, and whether Q flags them. Q is developed on synthetic degradations of public stills;
  its real calibration happens in the pilot.
* **P-D4, C1 on public data (lesion-level proxy).** Section 4.3.

Frozen encoders on CPU (DINOv2 ViT-B/14; BiomedCLIP ViT-B/16) with linear/patch-level detection
heads are the primary detector. Fine-tuning a detector (for example a small YOLO/DETR) needs a
GPU and is optional, as a pre-specified secondary analysis. The encoder and head are chosen by
internal CV within training sites only.

Duplicate control (all public data): pHash (64-bit, Hamming ≤ 6) plus SSIM ≥ 0.95, within and across
datasets. Clusters are kept inside one split, label-discordant clusters are excluded, and cross-site
duplicates are removed from the training side. BUSI is audited against its documented defects
(duplicates, axillary images, visible needles); the curated release is preferred.

## 4. Woman-level decision and the novel contribution (C1)

### 4.1 Aggregation (fixed before field data; tuned only on pilot data)
Frame score s_t → smoothed score: mean of a sliding window of *w* frames (w ∈ {1, 3, 5, 9}, chosen
in the pilot) → breast score = max over frames → woman score = max over breasts. A suspicious lesion
must be persistent over ≥ *w* consecutive frames. This suppresses single-frame artefacts, which
drive false recalls.

### 4.2 Operating points
Thresholds are chosen to give target recall rates (prevalence is 2–5 per 1,000, so recall rate
≈ false-recall rate):

* **OP-5:** recall ≤ 5% (≤ 50 false recalls per 1,000 women)
* **OP-10:** recall ≤ 10% (≤ 100 false recalls per 1,000 women)

The field study reports **sensitivity at OP-5 and OP-10** as the primary measures.

### 4.3 C1: certified recall operating point under site shift (main methodological contribution)
At a new site (new country, device, operator cadre), the threshold must come with a stated
guarantee. C1 has four parts:

* **C1a, recall ceiling from local normals (exact).** Choose the lowest threshold whose one-sided
  95% Clopper–Pearson upper bound on the recall rate among *m* local screen-negative women is
  ≤ R_max. For example: at an observed 3% recall, certifying ≤ 5% needs about 340 women and certifying
  ≤ 10% about 70; at an observed 7%, certifying ≤ 10% needs about 290.
* **C1b, sensitivity floor from local cancers (exact).** At that threshold, the one-sided 95% lower
  bound on sensitivity from *n* local cancers. With zero misses, 29 cancers certify ≥ 0.90 and 59
  certify ≥ 0.95 (46 and 93 with one miss).
* **C1c, spectrum weighting (new).** Locally available cancers are usually symptomatic and larger
  than screen-detected cancers, so an unweighted sensitivity overstates screening sensitivity.
  Cancers are weighted to a prespecified target spectrum (tumour size ≤ 10 / 11–20 / > 20 mm ×
  palpable yes/no; target proportions from published screen-detected series in similar settings,
  fixed at freeze). The bound uses the effective sample size (Kish). Hypothesis: unweighted
  enriched-sample sensitivity exceeds the spectrum-weighted value, and the weighted value is closer
  to sensitivity among camp-detected cancers (checked in the field study, with the limited number
  of camp cancers).
* **C1d, borrowing strength across sites (new; main hypothesis H1).** A hierarchical
  beta-binomial model of site-level sensitivity and recall at each threshold. Prior sites are
  treated as exchangeable draws, and the model is updated with the local data. **H1:** at matched
  achieved coverage, C1d needs about half the local cancers that C1b needs (≤ 30 instead of 59 for a
  0.95 floor; ≤ 15 instead of 29 for 0.90).

**Evaluation of C1:**
* *On public data (proxy, lesion level):* LOSO, with "recall" replaced by the normal-frame
  false-positive rate and "sensitivity" by malignant-lesion detection. Within each held-out site,
  200 repeated draws of a certification subset (*n* cancers ∈ {10, 15, 20, 30, 59}; *m* normals as
  available), evaluated on the disjoint remainder. Endpoint: **achieved coverage** (the fraction of
  draws in which the remainder meets the certified floor or ceiling), against the nominal 95%. Few
  sites have normals, so this is exploratory.
* *Woman level:* pilot data (Section 8.2) certify the operating point carried into the field study.
  The field study then tests whether the certified floor and ceiling held (Section 8.4).

**Confirmatory status.** On public data C1 is exploratory. In the field study the certified
recall ceiling and sensitivity floor are co-primary validity checks (Section 8.4).

## 5. Label-blind data audit (public data; before freezing)
Allowed: checksums, counts, metadata fields and missingness, verification method, label *counts*,
duplicate detection, annotation/caliper detection, pixel spacing. Not allowed: training any model
evaluated on a held-out site, or looking at held-out performance. The output is
`docs/data_audit.md`, which fixes site roles, units and exclusions.

## 6. Comparators
* **Field study (primary):** CBE by the same health worker in the same woman (paired). This is the
  standard of care in India's NP-NCD.
* **Field study (secondary, where feasible):** expert radiologist reading of the same stored sweep
  (upper bound for the AI on the same data); expert radiologist live diagnostic ultrasound (part of
  the reference standard, Section 8.3).
* **Not head-to-head in this protocol:** iBreastExam (tactile) and Thermalytix (thermal). Their
  published figures are used for positioning only. They could be added as arms if a partner site
  already uses them.
* **Public development:** image-only linear probe on each encoder; TabPFN and CatBoost on
  embedding PCA; a simple baseline of a frame classifier trained only on BUSI. These test whether a
  stronger tabular head adds anything over a linear probe.

HARBOR's availability-aware fusion has no natural role here: the input is the sweep, and screening
women have little structured metadata (age, CBE result, symptoms). It is **dropped from the primary
pipeline**. One pre-registered exploratory test remains: does adding age and the CBE result as inputs
(B5 concatenation vs M1 HARBOR) change field-study sensitivity at OP-10? That test is run in the
field data only. The value-of-information controller is not used (Phase 0).

## 7. Primary endpoints (field study)
1. **Sensitivity at OP-5 and OP-10** (woman level; cancer diagnosed within 12 months of the screen;
   UNREADABLE after rescans counts as RECALL), estimated on the enriched cancer arm with spectrum
   weighting (C1c), and unweighted as a secondary measure.
2. **False recalls per 1,000 women** and **recall rate**, estimated in the screening-camp arm.
3. **UNREADABLE rate** (before and after rescan).
4. **Primary comparison (confirmatory):** sensitivity of **CBE + AI** (RECALL if either is positive)
   against **CBE alone**, paired on the same cancers, at the frozen OP-10 threshold. Exact McNemar,
   two-sided α = 0.05. Co-primary: the one-sided 95% upper bound of the CBE + AI recall rate among
   camp women must be ≤ 10%.
5. Secondary: PPV re-weighted to 2 and 5 per 1,000; sensitivity by tumour size, palpability, age band,
   breast density proxy (ultrasound tissue composition) and device; per-breast localisation
   accuracy for recalled cancers; health-worker scan time.

## 8. Enriched field study (design for the paper; not conducted by this project)

### 8.1 Design overview
A prospective, multicentre, paired diagnostic-accuracy study with two arms enrolled in parallel
at the same sites, by the same health workers, on the same handheld device model:

* **Screening-camp arm:** about **3,000 consecutive asymptomatic women** aged 30–65 attending NP-NCD
  screening camps (estimates recall rate, false recalls per 1,000 and unreadable rate, plus the few
  camp-detected cancers).
* **Enriched cancer arm:** about **150 women with biopsy-confirmed breast cancer**, recruited *before
  treatment* at partner cancer centres, plus a matched sample of about 150 women with biopsy-proven
  benign lesions (to estimate how often benign lesions trigger RECALL). Recruitment is
  **spectrum-stratified**: target ≥ 50% tumours ≤ 20 mm and ≥ 30% clinically non-palpable or only
  equivocally palpable (screen-detected, mammography-detected in urban centres, or incidental
  findings). Health workers are blinded to arm status where practical (mixed scanning sessions).

### 8.2 Pilot (precedes the frozen validation)
About 300 camp women plus about 60 enriched cancers, scanned by the same health-worker cadre. The
pilot data are used to fix the sweep protocol, the aggregation window *w*, the quality gate
thresholds and the C1-certified OP-5 and OP-10 thresholds. **Pilot women are never part of the
validation sample.**

### 8.3 Reference standard and verification
* Every RECALL (by AI or by CBE): diagnostic ultrasound by a radiologist ± biopsy, within 30 days.
* Every screening-camp woman: 12-month follow-up for interval cancer (phone contact plus linkage
  to cancer registries/hospital records where available).
* **Verification sample of screen-negatives:** a random 10% of AI-NORMAL + CBE-negative camp women,
  plus all UNREADABLE women, get a radiologist ultrasound at the visit. This addresses partial
  verification bias; analysis uses inverse-probability weighting (Begg–Greenes).
* An expert reader reads all stored sweeps blinded to AI, CBE and outcome.

### 8.4 Sample size (exact binomial; from `scripts/sample_size_field_study.py`)

**Sensitivity (enriched arm, n = 150 cancers):**

| True sensitivity | Observed | 95% CI (Clopper–Pearson) |
|---|---|---|
| 0.85 | 128/150 | 0.786–0.906 |
| 0.90 | 135/150 | 0.840–0.943 |
| 0.95 | 142/150 | 0.898–0.977 |

| Goal: lower 95% bound above… | True sens. | Power at n = 150 | n for 80% | n for 90% |
|---|---|---|---|---|
| 0.80 | 0.90 | **0.93** | 113 | 149 |
| 0.85 | 0.92 | 0.69 | 185 | 240 |
| 0.85 | 0.90 | 0.35 | 379 | 492 |
| 0.90 | 0.95 | 0.52 | 243 | 316 |

→ 150 cancers support a claim of **sensitivity > 0.80** (if true ≈ 0.90). A claim of > 0.85 needs
185–380 cancers, depending on true performance. **The author should choose the target claim; this
draft proposes > 0.80, with 150 cancers.**

**Primary comparison, CBE + AI vs CBE alone (paired, exact McNemar, n = 150 cancers):**

| Added sensitivity (discordant cells) | Power, n = 150 | Power, n = 100 |
|---|---|---|
| +10 points (13% / 3%) | 0.85 | 0.67 |
| +15 points (18% / 3%) | 0.99 | 0.92 |
| +20 points (22% / 2%) | 1.00 | 0.995 |

**Recall and specificity (camp arm, about 2,985 cancer-free women of 3,000 enrolled):**

| True false-recall rate | Observed | 95% CI | False recalls per 1,000 (95% CI) |
|---|---|---|---|
| 3% | 90/2,985 | 2.4–3.7% | 30 (24–37) |
| 5% | 149/2,985 | 4.2–5.8% | 50 (42–58) |
| 10% | 298/2,985 | 8.9–11.1% | 100 (89–111) |

Power to show specificity lower bound > 0.93 when true 0.95: **0.996**; > 0.94: 0.64 (needs about
4,225 cancer-free women for 80%); > 0.88 when true 0.90: 0.94.

**Camp-detected cancers:** at 2–5 per 1,000, about 6–15 cancers are expected among 3,000 women
(P(≥ 10) is 0.08 at 2/1,000 and 0.93 at 5/1,000). That is too few to estimate screening sensitivity
directly, which is why the enriched arm exists. The camp cancers are used only as a descriptive
check of spectrum weighting (C1c).

### 8.5 Analysis
Sensitivity: spectrum-weighted (primary) and unweighted. Specificity and recall: verification-bias
corrected (Begg–Greenes) and naive. PPV and NPV re-weighted to 2 and 5 per 1,000. Clustered by
health worker and site (GEE or cluster bootstrap). Prespecified subgroups as in Section 7.

### 8.6 Regulatory and ethics (must be resolved before the field study)
* **PC-PNDT Act (India):** ultrasound machines must be registered to specific premises, and
  state guidance (e.g. Delhi) permits portable use only within registered premises. Operator
  qualifications are regulated. Health-worker operation of handheld ultrasound at camps may
  need registered mobile units, operation within registered facilities, or a specific exemption.
  **Legal advice and state PC-PNDT authority approval are a precondition.**
* CDSCO: AI software for screening is regulated as a medical device (SaMD) under the Medical Device
  Rules 2017. The field study would be a clinical performance investigation needing CDSCO and
  ethics approval.
* ICMR ethical guidelines for AI in health (2023); DPDP Act 2023 for data protection; written
  informed consent in local languages; trial registration in CTRI before enrolment.

## 9. Statistics (public development)
Patient-clustered bootstrap (2,000 replicates) per site; Clopper–Pearson at fixed thresholds;
random-effects meta-analysis across LOSO sites (Hartung–Knapp); DeLong for paired AUROC on a held-out
site; 5×5 grouped CV with the Nadeau–Bengio correction within training sites. No significance
stars on exploratory results.

## 10. Robustness: portable-device degradations (public data)
Speckle (multiplicative Rayleigh, σ ∈ {0.1, 0.2, 0.3}), blur (σ ∈ {1, 2} px), downsampling
(×0.5, ×0.25), JPEG quality {50, 20}, gain/contrast ±30%, depth crop 25%, partial probe contact
(random dark sector). Endpoints: detection sensitivity at the frozen frame threshold, and the
fraction of degraded frames that Q flags as unreadable.

## 11. Reproducibility
`scripts/reproduce.sh` regenerates every number from raw files whose SHA-256 checksums are in
`data/MANIFEST.sha256`. Held-out evaluation scripts refuse to run unless the `protocol-v1.0` tag
exists and `docs/protocol.md` matches it. HARBOR's run-to-run nondeterminism (Phase 0) is no longer
relevant to the primary pipeline. Seeds are fixed for all models.
