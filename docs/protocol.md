# Pre-registered protocol: certified-sensitivity AI referral for handheld breast ultrasound

**Version 0.9 (DRAFT, not yet frozen).** It becomes v1.0, frozen and tagged `protocol-v1.0`, after
the label-blind data audit in Section 4. No model is trained on, or evaluated against, any
held-out site before that tag exists. Every change after freezing goes in `docs/deviations.md`,
with date, reason and expected impact.

Reporting follows TRIPOD+AI (2024) and CLAIM (2024 update); STARD 2015 items are used for the
diagnostic-accuracy tables.

---

## 1. Intended use and claim ceiling

**Device concept.** A software module for a handheld or portable breast ultrasound system, for use
in settings without mammography or on-site breast radiology (WHO GBCI "timely diagnosis" pillar;
BHGI basic and limited resource levels). For each woman it outputs **REFER** (send for biopsy or
specialist work-up) or **ROUTINE** (no referral: either no suspicious lesion, or a lesion that can go
to short-interval follow-up). It also reports a calibrated risk and an explicit **ABSTAIN → REFER**
flag when the input lies outside what the model was calibrated on.

**Pipeline (two stages, evaluated separately and then together):**

* **S1, lesion detection (screening stage).** Does this image or sweep contain a suspicious
  focal lesion? Used for asymptomatic women and for whole-breast scanning.
* **S2, malignancy triage (diagnostic stage).** For a detected lesion: refer for biopsy, or
  short-interval follow-up?
* **REFER** if S2 says refer, or if either stage abstains.

**What public retrospective data can and cannot show (stated now so the paper stays within it):**

| Claim | Supportable here? |
|---|---|
| S2 accuracy at prespecified operating points, externally validated across countries | **Yes**: primary evidence |
| A sensitivity guarantee that transfers to a new site, and how many local cases that needs | **Yes**: the novel contribution (Section 3) |
| S1 detection accuracy | **Partly**: only on curated still images; normal images come from few sites (BUSI, BUS-UCLM, possibly BUS-CoT); no whole-breast sweeps |
| End-to-end screening sensitivity, specificity and recall rate in an asymptomatic population | **No.** Projected under stated prevalence assumptions only, labelled as projections, never as results. Needs the prospective study in Section 11. |
| Performance on low-cost handheld devices operated by non-specialists | **No**: simulated degradations only (Section 9) |
| Clinical validation or regulatory readiness | **No.** Never claimed. |

## 2. Data

### 2.1 Sites
A "site" is a source dataset. Development and validation use **leave-one-site-out (LOSO)**: each
site is held out once, and everything else (model fitting, threshold choice, calibration) uses only
the remaining sites.

| Site | Country | Role | Expected use |
|---|---|---|---|
| BUS-BRA | Brazil | LOSO site | S2; BI-RADS comparator |
| BrEaST (Breast-Lesions-USG) | Poland | LOSO site | S2; BI-RADS comparator; clinical metadata (age, composition, symptoms) |
| BUSI (curated release; original audited for duplicates) | Egypt | LOSO site | S2; S1 normals |
| BUS-CoT | China | LOSO site | S2 (largest); descriptors |
| BUS-UCLM | Spain | S1 only (38 patients; too few for an S2 site) | S1 normals |
| GDPH&SYSUCC, UDIAT, PersonalizedUS tabular | China / Spain / Brazil | optional, only if obtained before freeze | S2 |

Final inclusion is fixed at freeze. A site is an S2 LOSO site only if it has **≥ 50 malignant
lesions** after exclusions. Smaller sites are pooled into training only and reported descriptively.

### 2.2 Unit of analysis
* **Lesion** where lesion IDs exist; otherwise **patient** where patient IDs exist; otherwise **image**, flagged in every table.
* Multiple images of one lesion: the lesion score is the **mean** of image probabilities (prespecified; max reported as a sensitivity analysis).
* Woman-level output (end-to-end): REFER if any lesion is referred.

### 2.3 Reference standard
Histopathology (core biopsy or excision) for malignant. Benign by histopathology, or by stable
follow-up where the dataset documents it (≥ 2 years preferred; the dataset's own definition
recorded per site). Every table reports the verification method per site. Lesions without
verification are excluded.

### 2.4 Exclusions (applied label-blind where possible)
1. Non-breast images (axilla), images with visible biopsy needles, Doppler-only or elastography-only frames.
2. **Duplicates and near-duplicates:** perceptual hash (pHash, 64-bit) with Hamming distance
   ≤ 6, plus SSIM ≥ 0.95 on 256 px greyscale, within and across sites. Near-duplicate clusters are
   kept inside one split. Clusters with discordant labels are excluded and counted. Cross-site
   duplicates are removed from the *training* side of each LOSO fold. The threshold is set on
   pairs a human has reviewed during the audit and then frozen.
3. Images with burned-in annotations (calipers, text, arrows) are kept, but annotations are
   masked by a fixed preprocessing step. Caliper presence is recorded as a covariate, because
   calipers can leak suspicion into the label.

## 3. Primary hypothesis and novel contribution

**Problem.** Public breast-ultrasound models lose about 0.08 AUROC across datasets (Diagnostics
2026 benchmark), and a threshold tuned for 98% sensitivity at one site gives an unknown,
usually lower, sensitivity at another. For a referral device the missed-cancer rate is the
safety-critical quantity, and no published ultrasound work certifies it at a new site.

**Contribution C1, certified-sensitivity referral under site shift (the main novelty).** A
threshold-selection procedure that, at a new site, comes with a stated guarantee on the
missed-cancer rate and a stated local-data requirement:

* **C1a, local certification (exact).** Given *n* local biopsy-proven cancers, pick the highest
  REFER threshold whose one-sided 95% Clopper–Pearson lower bound on sensitivity is ≥ the target.
  Finite-sample and distribution-free (needs only the local cancers to be representative). Its
  cost is known in closed form: with zero misses, **59 cancers certify ≥ 0.95, and 149 certify ≥ 0.98**
  (93 and 236 if one miss is allowed).
* **C1b, source-weighted conformal (approximate, needs no local labels).** Site-weighted / Mondrian
  conformal calibration on the training sites, with the threshold set so that the
  weighted miss rate is ≤ 1 − target. Valid only under exchangeability with the new site, so its
  validity is *tested* under LOSO, not assumed.
* **C1c, borrowing strength (the hypothesis that would be genuinely new).** A hierarchical
  beta-binomial model of site-level sensitivity at each threshold (source sites as exchangeable
  draws from the site population), updated with the *n* local cancers. The threshold is set so the
  posterior probability that sensitivity ≥ target is ≥ 0.95. **Hypothesis H1:** at matched
  achieved coverage, C1c needs substantially fewer local cancers than C1a, for example ≤ 30
  instead of 59 for the 0.95 floor.

**Primary endpoint (prespecified).** For each LOSO held-out site, with the threshold chosen by the
method under test and applied unchanged:

1. **Achieved held-out sensitivity** against the 0.95 target (validity), and
2. **Specificity**, equivalently the **fraction of benign lesions spared referral** (utility).

**Primary comparison (the only one with confirmatory status).** C1b (source-only, no local labels)
against the **naive transfer threshold** (the threshold giving 0.95 sensitivity on pooled training
sites, applied as-is), on the **across-site shortfall**: the mean over held-out sites of
max(0, 0.95 − achieved sensitivity). H0: equal shortfall. Test: paired permutation over
sites, with lesion-level bootstrap CIs per site. **With 4–5 sites this test has low power. It is
pre-registered anyway, and a null result will be reported as such.**

**Secondary, utility comparison against standard of care.** On held-out sites that record
radiologist BI-RADS: the proposed S2 against the rule **"biopsy if BI-RADS ≥ 4"**. Endpoint:
benign biopsies avoided at **non-inferior sensitivity** (margin −0.02; paired lesion-level
bootstrap, one-sided 97.5%). Also against "biopsy if BI-RADS ≥ 4B" where the subcategory exists.

**C1a/C1c evaluation design.** Within each held-out site, repeat 200 times: draw a certification
subset with *n* ∈ {10, 20, 30, 59, 100, 149} cancers (and benign lesions in the site's ratio), set
the threshold, and evaluate on the disjoint remainder. Report the achieved coverage (the fraction of
repeats in which remainder sensitivity ≥ target) and the specificity, as functions of *n*.

## 4. Label-blind data audit (done before freezing; allowed)
Allowed before freeze: file integrity and checksums, image counts, metadata fields and missingness
per site, verification-method counts, label *counts* per site, duplicate detection (Section 2.4),
caliper/annotation detection, image size and pixel-spacing availability. **Not allowed** before
freeze: fitting any model whose outputs are evaluated on a held-out site, or looking at
performance on a held-out site. The audit output (`docs/data_audit.md`) fixes the site list, the
unit of analysis per site and the exclusion counts. Then the protocol is frozen.

## 5. Inputs and models

### 5.1 Image representation (frozen encoders on CPU; no fine-tuning in the primary analysis)
* E1: DINOv2 ViT-B/14 (general). E2: BiomedCLIP ViT-B/16 (medical). An ultrasound-specific encoder only if its weights are obtainable before freeze.
* Input: **whole image** with annotations masked (deployable: needs no lesion outline). Secondary: lesion ROI crop from dataset masks or boxes, reported as an **optimistic upper bound**, since deployment would need an automatic detector.
* The encoder is chosen *inside* the training sites by grouped internal CV (mean of S2 AUROC over training sites). It is never chosen on a held-out site.

### 5.2 Clinical and report fields (availability varies by site)
Age; palpable/symptomatic; breast composition; radiologist BI-RADS category; BI-RADS descriptors
(shape, margin, orientation, echo pattern, posterior features, calcifications); device. A field is
used only if present at ≥ 1 training site. **When BI-RADS is used as a model input, the model is
not compared against BI-RADS as an independent comparator.** Two model families are therefore
reported: *image + clinical without BI-RADS* (compared against BI-RADS) and *image + clinical +
BI-RADS* (decision support for a reader).

### 5.3 Heads (all fitted only on training sites)
| ID | Model | Role |
|---|---|---|
| B0 | Radiologist BI-RADS rule (≥ 4; ≥ 4B) | standard of care |
| B1 | Logistic regression on clinical fields | baseline |
| B2 | Image-only: L2 logistic regression on frozen embeddings | baseline |
| B3 | TabPFN v2 on [PCA-64 of embedding; clinical] | baseline (strong tabular) |
| B4 | CatBoost on the same inputs | baseline |
| B5 | Concatenation fusion with median imputation plus missingness indicators | baseline |
| B6 | Modality dropout (random field masking during training) | comparator for H2 |
| M1 | HARBOR availability-aware fusion: embedding as one token group, each clinical field as a token; availability channel active | tested in H2 |

The classifier behind the C1 procedures is the best of B2–B6/M1 by **training-site internal CV**,
chosen before any held-out evaluation. If two are within 0.005 AUROC, the simpler one is used.

## 6. Secondary hypotheses (each with its ablation; dropped if it fails under LOSO)

* **H2, availability-aware fusion.** M1 shows a smaller held-out AUROC drop than B5 and B6 when the
  held-out site's set of recorded fields differs from training. Ablation: M1 without the
  availability channel (value-only). Measured as held-out minus internal AUROC per site. *Expected
  power: low (4–5 sites). If M1 is not better than B5, HARBOR is removed from the method.*
* **H3, biopsy reduction in BI-RADS 3/4A.** Among lesions rated BI-RADS 3–4A, the fraction of benign
  lesions spared referral at the C1-certified threshold. Reported only where BI-RADS subcategories
  exist.
* **H4, calibration transfer.** Calibration intercept and slope on the held-out site before and after
  logistic recalibration on *m* ∈ {25, 50, 100, 200} local lesions (repeated draws, evaluated on the
  remainder). Endpoint: the smallest *m* for which the median |intercept| < 0.1 and the slope lies in
  [0.9, 1.1].
* **H5, S1 detection.** Lesion-vs-normal AUROC and specificity at sensitivity 0.95, LOSO over sites
  with normal images. Descriptive only; too few normal-image sites for inference.

The value-of-information controller is **not** part of this protocol. Phase 0 showed it did not
beat simple fixed-panel mixtures on its own benchmark (`docs/phase0_audit.md`), and these data have
no sequential test decision to optimise. It may come back only as a pre-registered amendment.

## 7. Statistics
* Lesion- or patient-clustered bootstrap (2,000 replicates) for every per-site CI.
  Clopper–Pearson for sensitivity and specificity at fixed thresholds.
* Across sites: random-effects meta-analysis (logit scale; DerSimonian–Laird, with Hartung–Knapp
  CIs given the few sites) of sensitivity and specificity, plus the per-site forest plot. Pooling
  is never done by concatenating predictions across sites.
* AUROC comparisons on a held-out site: DeLong (paired). Internal CV within training sites:
  5×5 grouped CV with the Nadeau–Bengio corrected resampled t-test.
* One confirmatory test (Section 3). Everything else is exploratory, reported with CIs and no
  significance stars.
* PPV/NPV are reported at each site's observed prevalence **and** re-weighted to an assumed
  screening prevalence (5 per 1,000) and a symptomatic-clinic prevalence (10%), labelled as
  re-weighted.

## 8. Calibration and decision analysis
Reliability plots (10 quantile bins, lesion-level), calibration intercept and slope, ECE as a
secondary measure. Decision curves (net benefit) over threshold probabilities of 1–20%, relevant to
biopsy referral, against "refer all", "refer none" and BI-RADS rules.

## 9. Robustness: portable-ultrasound degradations
Applied to held-out images only, at prespecified levels: additive speckle (Rayleigh multiplicative
noise, σ ∈ {0.1, 0.2, 0.3}), Gaussian blur (σ ∈ {1, 2} px), downsampling to {0.5, 0.25} resolution
and back, JPEG quality {50, 20}, gain/contrast ±30%, depth crop (bottom 25% removed). Endpoint:
change in sensitivity at the frozen threshold, and whether the abstention rule fires.

## 10. Subgroups and error analysis
Age bands (< 40, 40–49, 50–69, ≥ 70) where age exists; breast composition; device; lesion size
(mask area in mm² where pixel spacing exists, else in pixels, flagged); BI-RADS category;
histological subtype for the missed cancers. Every false negative at the certified threshold is
listed with its site, BI-RADS, size and histology, and up to 12 example images go in the
supplement.

## 11. Prospective study the paper will specify (not run)
Paired-design diagnostic accuracy study at ≥ 3 LMIC clinics, with women presenting for CBE
(symptomatic and asymptomatic arms). Handheld ultrasound by trained non-radiologists with the
device, followed by standard-of-care ultrasound and biopsy or 2-year follow-up. Sample size (exact
binomial, Clopper–Pearson, 80% power, true sensitivity 0.97): **100 cancers** to show a lower 95%
bound > 0.90; **240 cancers** to show a lower bound > 0.93. At 10% prevalence in the symptomatic arm
that is about 1,000 or 2,400 women. The asymptomatic arm is powered for recall rate, not
sensitivity, because at about 5 cancers per 1,000 it cannot be powered for sensitivity at a
feasible size.

## 12. Software and reproducibility
Every number in the paper is regenerated by `scripts/reproduce.sh` from raw files whose SHA-256
checksums are listed in `data/MANIFEST.sha256`. Seeds are fixed; HARBOR's run-to-run
nondeterminism (Phase 0) is either removed or quantified by repeated fits. Held-out evaluation
scripts refuse to run unless `git tag protocol-v1.0` exists and the protocol file matches the tagged
version.
