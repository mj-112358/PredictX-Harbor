# Niche decision memo (Phase 1)

Status: **draft for decision.** Nothing in this memo has been run on new data yet. The literature
claims come from a 2023–2026 search done on 2026-10-22 (sources listed at the end). In this
session only search snippets were reachable: Zenodo, TCIA, Mendeley, PubMed/PMC, nature.com and
arXiv full texts were all blocked. Every dataset size, field list and licence below is therefore
marked **verified** (official documentation read), **reported** (taken from a paper abstract or
secondary source) or **unknown**. Each one gets re-checked against the files once they are
attached.

## 1. Bottom line

* **Primary track (recommended): Option A, reframed.** *Site-robust, safety-constrained triage of
  breast lesions on ultrasound: which lesions can safely go to short-interval follow-up instead of
  biopsy, with external validation across countries.* This is an **early-diagnosis triage** problem
  (the WHO Global Breast Cancer Initiative's "timely diagnosis" pillar), not population screening.
  No public retrospective dataset can support a claim of screening accuracy, and the paper must not
  make one.
* **Secondary track: none by default.** Option D (NHANES liver-fibrosis triage) fits the
  existing acquisition controller better than anything in breast. If you want the controller to
  stay a headline contribution, D is the only honest home for it, but it would be a second disease
  and a second paper's worth of work. My advice is to commit to A alone, and decide on D only after
  the Phase 3 baselines for A are in.
* **Options B and C are not recommended.** B: the public BCSC Risk Factors dataset has **no
  outcome variable**, and the outcome-bearing BCSC file is aggregated to 3,071 rows, so there is
  nothing to learn at the individual level. C: hundreds of GB, needs a GPU (we have none), and the
  field is crowded with prospectively validated, cleared products (MASAI, PRAIM, Transpara, Lunit).
* **Blocking constraint.** This session cannot download any dataset, pretrained image encoder, or
  the TabPFN weights (Section 6). Phase 3 cannot start until you attach files or widen the network
  policy.

## 2. Constraints found in Phase 0 that shape the choice

| Resource | Status |
|---|---|
| Compute | 4 CPU cores, 15 GB RAM, ~30 GB free disk, **no GPU** |
| PyPI | reachable (pinned requirements installed) |
| GitHub, raw.githubusercontent.com | reachable for code and documentation |
| huggingface.co | **blocked**: TabPFN v2 weights and all common pretrained image encoders (DINOv2, BiomedCLIP, SigLIP, ultrasound foundation models) |
| download.pytorch.org, dl.fbaipublicfiles.com | **blocked**: torchvision/DINOv2 weights |
| Zenodo, figshare, Mendeley Data, TCIA, Kaggle, PhysioNet, UCI, CDC/NHANES, BCSC | **blocked** |

Implications: (i) any image work uses **frozen** encoders on CPU (≈0.1–0.3 s per image for a
ViT-B on 4 cores, so ~1–2 h for 15k images). Fine-tuning is off the table without a GPU. (ii) The
encoder weights have to come from you, or from an allow-listed huggingface.co.

## 3. Comparison

Scale: ++ strong, + adequate, 0 weak, − disqualifying.

| Criterion | A. Ultrasound lesion triage | B. Supplemental-screening selection | C. Screening mammography | D. NHANES fibrosis triage |
|---|---|---|---|---|
| Clinical need | ++ timely diagnosis in LMICs; unnecessary benign biopsies (BI-RADS 3/4A) | + real (dense breasts) | ++ | ++ MASLD burden; FIB-4 has many false positives |
| Standard of care to beat | ++ radiologist BI-RADS category (available in BUS-BRA, BrEaST, BUS-CoT) | 0 BCSC / Tyrer–Cuzick already exist | ++ double reading | ++ FIB-4 → elastography two-step pathway |
| Public data we can actually get | + several small-to-mid public sets (Section 4), all blocked here and needing you to attach them | − no individual-level outcome in public BCSC files | − 300+ GB, GPU | ++ NHANES 2017–2020 and 2021–2023 both include FibroScan (blocked here; you would attach) |
| External validation across sites | ++ 4–6 datasets from 4–5 countries → leave-one-dataset-out | − | + (CMMD, CBIS-DDSM, RSNA) | + temporal (2017–20 → 2021–23); one country, one survey |
| Feasibility on our compute | + frozen embeddings plus tabular head on CPU | ++ | − | ++ |
| Fit with HARBOR (availability-aware fusion) | + metadata differs by site (Section 5); missingness is mostly *site-level*, not per-patient | 0 | 0 | ++ lab panels genuinely partially missing per patient |
| Fit with acquisition controller | 0 the pathway decision is mostly binary (biopsy vs follow-up); no cohort has CBE-only patients, so "order ultrasound or not" can't be studied | 0 | 0 | ++ natural: labs → order elastography or not |
| Novelty room (2023–2026) | + see Section 4.3: the cross-site generalisation gap is documented but unsolved; conformal risk for ultrasound exists only within one multicentre Brazilian cohort | 0 | − | 0 crowded: SAFE was *derived in NHANES*; LiverRisk, MAF-5 |
| Matches your stated direction (breast) | ++ | ++ | ++ | − |

## 4. Option A in detail

### 4.1 Clinical question (proposed wording)

> Among women with a breast lesion found on clinical breast examination or ultrasound, in a
> setting with ultrasound but no mammography or on-site pathology AI, can an ultrasound-plus-clinical
> model identify lesions that can safely go to short-interval follow-up instead of biopsy, at a
> prespecified sensitivity of at least 0.98 for malignancy, when deployed at a hospital whose
> population, scanner and recorded metadata were not seen in training?

Primary endpoint (proposal, to be frozen in `docs/protocol.md`): **specificity, equivalently the
fraction of benign lesions spared biopsy, at sensitivity ≥ 0.98**, with the threshold fixed on
training sites and applied unchanged to the held-out site. Report the sensitivity achieved on the
held-out site as well, since it will drift. Secondary: the same at ≥ 0.95, AUROC, calibration, net
benefit.

### 4.2 Candidate datasets

| Dataset (country) | Size | Labels | Per-case metadata | Licence / access | Status |
|---|---|---|---|---|---|
| **BUS-BRA** (Brazil, INCA Rio) | 1,875 images, 1,064 patients; 1,268 benign / 607 malignant images; BI-RADS 2/3/4/5 = 562/463/693/157 images | biopsy-proven (verification of BI-RADS 2–3 lesions **unknown**: biopsy or follow-up) | BI-RADS category, histology, scanner (4 systems, reported), side; patient age **unknown** | Zenodo, cite Gómez-Flores et al., Med Phys 2024; licence field **unknown** | size verified from GitHub README; rest reported |
| **BrEaST / Breast-Lesions-USG** (Poland) | 256 patients/scans: 154 benign, 98 malignant, 4 normal | biopsy **or follow-up**, 33 diagnoses | **age, breast tissue composition, signs and symptoms**, BI-RADS category and descriptors, verification method | CC BY 4.0 (reported) | reported |
| **BUSI** (Egypt, Baheya) | 780 images, 600 women; ~437 benign / 210 malignant / 133 normal | label source not clearly documented | **none per image** | Data in Brief 2020; a curated version (Aumente-Maestro et al., 2025) exists | **known defects**: duplicate images (some in both classes), ~70 axillary images, biopsy needles visible. Use the curated release or deduplicate ourselves. |
| **BUS-CoT** (China) | 11,439 images, 10,019 lesions, 4,838 patients, 99 histopathology types | histopathology | observation and BI-RADS-style feature descriptions, diagnosis | figshare 10.6084/m9.figshare.29036876; licence **unknown** | reported; single centre **unknown** |
| **GDPH&SYSUCC** (China, 2 hospitals) | 2,405 images; 886 benign (526 pts) / 1,519 malignant (676 pts) | pathology (biopsy or surgery) | none reported | stated public; download location **unknown** | reported |
| **BUS-UCLM** (Spain) | 683 images from **38 patients** (174 benign / 90 malignant / 419 normal) | segmentation masks; label source unclear | none | Mendeley 10.17632/7fvgj4jsp7.3 | reported; too few patients to be a validation site |
| **UDIAT Dataset B** (Spain) | 163 images (110 benign / 53 malignant) | — | none | by request (Yap et al. 2018) | too small; optional |
| Buzatto et al. / **PersonalizedUS** (Brazil, multicentre) | 1,936 biopsied BI-RADS ≥ 3 lesions, retrospective plus prospective cohorts | core/excisional biopsy | tabular: age, palpability, BI-RADS descriptors (shape, margin, …) | released with the AAAI 2025 paper; host **unknown** | reported. **Also the closest prior art** (Section 4.3). |
| CADBUSI (Mayo, USA) | 79,281 exams | pathology | BI-RADS | **not public** (internal use only) | excluded |

The realistic multi-site core is **BUS-BRA + BrEaST + curated BUSI + BUS-CoT (+ GDPH&SYSUCC if
obtainable)**: four or five datasets from Brazil, Poland, Egypt and China.
Leave-one-dataset-out then gives 4–5 external-validation folds with roughly 100 (BrEaST) to several
thousand (BUS-CoT) cancers per held-out site.

### 4.3 What the 2023–2026 literature says, and where the gap is

* **Specialist breast AI has moved to imaging and pathology, as the review said.** Prospective
  screening-mammography evidence is now strong: in MASAI (Sweden, >105k women) AI-supported
  screening increased detection 29% with 44% less reading workload, and the final readout
  (Lancet 2025/26) reports fewer interval cancers. A tabular cytology classifier is not competitive
  there and the paper should stop trying.
* **Ultrasound triage in low-resource settings is a recognised, open use case.** Berg et al.
  (Radiology 2023, Jalisco, Mexico) ran a commercial AI (Koios) on portable-ultrasound images of
  758 masses in 300 women with palpable lumps and reported 96% sensitivity for women with cancer,
  framed explicitly as triage to cut unnecessary referrals. Trials of AI-supported point-of-care
  breast ultrasound in LMICs are registered (NCT07462351, NCT06932133), and NCT05561842 (a
  non-AI mobile-ultrasound study in Nigeria) recruits until 2027. A 2025 systematic review covers
  34 studies of AI handheld breast ultrasound.
* **Cross-site generalisation of public ultrasound models is poor and documented.** A
  patient-leakage-aware benchmark over BUSI, BUS-UCLM, BUS-BRA and BrEaST (Diagnostics 2026)
  reports mean AUROC falling from 0.80 internally to 0.72 cross-dataset, mean external ECE of
  0.18, and ROI input helping. A ten-dataset leave-one-dataset-out study reports strong dependence
  on training composition. These papers diagnose the problem; none reports a **sensitivity
  guarantee or a biopsy-sparing policy at a fixed operating point** on the held-out site.
* **Conformal risk estimation for breast ultrasound exists, but only within one cohort.**
  PersonalizedUS (AAAI 2025) uses conformal prediction with subgroup-local coverage on tabular
  BI-RADS descriptors from a Brazilian multicentre study, and cites up to 65% fewer biopsies in
  BI-RADS 4a. It does not test transfer to a new country, scanner or metadata schema.
* **Biopsy reduction in BI-RADS 4A** is studied mostly in small single-centre retrospective
  "hypothetical downgrade" analyses (for example 14 of 43 4A biopsies avoided with no miss under a
  conservative rule). Commercial systems (Koios, S-Detect) are the comparators in the clinical
  literature, but we can't run them.

**The defensible contribution in one sentence:** *a biopsy-sparing ultrasound triage policy whose
missed-cancer rate is controlled at a prespecified level on a new site, together with an honest
measurement of how much local data a new site needs to recalibrate, evaluated by
leave-one-country-out across four or five public datasets.* That answers the reviewer's question
("what is the contribution if TabPFN matches you?"): the contribution is the safety-constrained
decision policy and its external evidence, not a new classifier.

### 4.4 How the existing components fit (honestly)

* **HARBOR / availability-aware fusion: testable, with modest expectations.** Recorded metadata
  really does differ by site: BrEaST has age, composition, symptoms and descriptors; BUS-BRA has
  BI-RADS and device; BUSI has nothing; BUS-CoT has descriptors. But the missingness is almost
  entirely **site-level** (a field is absent for a whole hospital), not the per-patient informative
  missingness HARBOR's theory addresses. With 4–5 sites the leave-one-out comparison has 4–5
  units, so a "smaller drop on the held-out site" claim will have little power. I expect the image
  embedding plus BI-RADS to carry most of the signal, and I'll test HARBOR against mask
  concatenation, imputation plus indicators and modality dropout exactly as you specified. If it
  doesn't help, the paper says so and drops it.
* **Value-of-information controller: weak fit.** Every public cohort contains only patients who
  already had ultrasound, so "should this patient get ultrasound after CBE?" can't be evaluated. What
  is left is a single biopsy-vs-follow-up decision, where VoI reduces to a threshold on predicted
  risk. That makes the decision curve and the sensitivity-constrained threshold the right tools,
  and makes VoI branding a stretch. I'd fold the controller's decision theory into contribution 3
  (selective referral / abstention) rather than present it as test acquisition. Phase 0 also
  re-examines whether the controller beat simple fixed-panel strategies in the coronary
  experiment (see `docs/phase0_audit.md`).
* **Typed biomarker metadata (units, assay, timing, quality): still untested.** None of these
  datasets record assay or quality metadata per measurement either. The closest real analogue is
  device/scanner (BUS-BRA) and image quality, which we can study through the
  portable-ultrasound degradation robustness check.

### 4.5 Ceiling on the claims (stated up front so nobody is surprised later)

* All candidate cohorts are **retrospective and enriched** (lesions that reached biopsy or a
  curated archive), not screening populations. Prevalence of malignancy is 30–60%, far above
  screening prevalence, so PPV/NPV must be re-weighted to target prevalences and labelled as such.
* No cohort was acquired with **portable or handheld** devices in a low-resource clinic. Low-resource
  robustness will be shown only by simulated degradation. A true claim needs a prospective
  portable-device study, which the paper will specify (setting, sample size, endpoints) but can't
  do.
* **Radiologist BI-RADS as comparator.** In some datasets BI-RADS may have been assigned or
  revised retrospectively, with knowledge of histology (incorporation bias). That would inflate
  the standard of care and make our comparison conservative, but it must be checked per dataset
  and stated.
* The best achievable claim is **"strong retrospective external evidence at prespecified
  operating points, with a protocol for prospective validation"**, never "validated" or
  "approved".

## 5. Option D in brief (why it is the better fit for your existing method, and why I still put it second)

NHANES 2017–March 2020 (pre-pandemic files) and August 2021–August 2023 (`LUX_L`) both contain
FibroScan liver stiffness and CAP measurements in the general population (≈6,700 adults with valid
elastography in 2017–2020, reported). The standard of care is the guideline two-step pathway (FIB-4
rule-out < 1.3 / rule-in > 2.67, elastography for the indeterminate band). A cost-aware controller
choosing who goes on to elastography is a natural and measurable use of `acquisition.py`, and lab
panels really are partially missing per participant, which is HARBOR's setting. Against it:
(i) different disease from the one you asked for; (ii) the reference standard is elastography, a
surrogate, not biopsy; (iii) the space is crowded: SAFE was *derived in NHANES*, and LiverRisk
and MAF-5 have NHANES comparisons, so the bar is "beat SAFE at matched elastography referrals",
which may not be met; (iv) it is one country and one survey, so external validation is only
temporal.

## 6. What I need from you

### 6.1 Decision
Which track to commit to: **A alone (recommended)**, A + D, D alone, or something else.

### 6.2 Network access (preferred over manual downloads)
Network access is set in the environment settings (cloud environment menu → Edit → Network
access; docs: https://code.claude.com/docs/en/cloud-environments#network-access). Allowing these
hosts would let me fetch everything myself, with checksums logged:

* `huggingface.co` and `cdn-lfs.huggingface.co` (plus `*.hf.co`): TabPFN v2 weights and pretrained image encoders
* `zenodo.org`, `figshare.com`, `ndownloader.figshare.com`: BUS-BRA, curated BUSI, BUS-CoT
* `www.cancerimagingarchive.net` and `nbia.cancerimagingarchive.net` (TCIA): BrEaST / Breast-Lesions-USG
* `data.mendeley.com`: BUS-UCLM (optional)
* for Option D only: `wwwn.cdc.gov`

### 6.3 Or attach these files (Option A)
I'm giving DOIs and record pages rather than guessed filenames. Please download **all files** of each record:

1. **BUS-BRA**: https://doi.org/10.5281/zenodo.8231412 (all files: images, masks, the case-level CSV, any fold-partition files)
2. **BrEaST / Breast-Lesions-USG**: the TCIA collection "Breast-Lesions-USG" (images plus the clinical-data spreadsheet), or the dataset link in Pawłowska et al., *Sci Data* 11:148 (2024), doi:10.1038/s41597-024-02984-z
3. **BUSI, curated**: the Zenodo record "Curated BUSI dataset – Curated Breast Ultrasound Images" (Aumente-Maestro et al., 2025). Also the **original** BUSI ("Dataset_BUSI_with_GT", Al-Dhabyani et al., *Data in Brief* 28:104863, 2020) so I can audit duplicates myself.
4. **BUS-CoT**: https://doi.org/10.6084/m9.figshare.29036876 (all files; please note the licence shown on the page)
5. *Optional:* GDPH&SYSUCC (Mo et al., *IEEE TMI* 2023, doi:10.1109/TMI.2023.3236011, see its data-availability link); the PersonalizedUS / Buzatto et al. tabular release (link in the AAAI 2025 paper, arXiv:2408.15458); BUS-UCLM (https://doi.org/10.17632/7fvgj4jsp7.3).

### 6.4 Model weights (needed whichever track)
* **TabPFN v2 classifier**: the checkpoint the pinned `tabpfn==2.2.1` requests is
  `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` from https://huggingface.co/Prior-Labs/TabPFN-v2-clf
  (this may require accepting the Prior Labs licence on Hugging Face). Without it, no TabPFN
  number can be reproduced or produced.
* **Image encoders** (frozen, CPU), prespecified as one general model and one medical model:
  `facebook/dinov2-base` (Apache-2.0) and `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
  (MIT): the whole Hugging Face repo of each. An ultrasound-specific encoder (EchoCare,
  github.com/CAIR-HKISI/EchoCare) is a possible third, if its weights are hosted somewhere you can reach.

### 6.5 Option D files (only if you choose D)
NHANES 2017–March 2020 pre-pandemic files: `P_LUX.xpt`, `P_DEMO.xpt`, `P_BIOPRO.xpt`, `P_CBC.xpt`,
`P_BMX.xpt`, `P_ALQ.xpt`, `P_DIQ.xpt`, `P_GHB.xpt`, `P_GLU.xpt`, `P_HEPB_S.xpt`, `P_HEPC.xpt`,
`P_BPXO.xpt`; and August 2021–August 2023: `LUX_L.xpt`, `DEMO_L.xpt`, plus the `_L` versions of
the same lab and exam files, if released. All from https://wwwn.cdc.gov/nchs/nhanes/.

## 7. Sources (2023–2026 search, 2026-10-22)

* BUS-BRA: [GitHub README](https://github.com/wgomezf/BUS-BRA), [Zenodo record](https://zenodo.org/records/7730709); BI-RADS counts from [arXiv 2403.09828](https://arxiv.org/pdf/2403.09828)
* BrEaST: [Pawłowska et al., Sci Data 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10830496/)
* BUSI defects: [Letter to the editor, PMC10293973](https://pmc.ncbi.nlm.nih.gov/articles/PMC10293973); [arXiv 2508.17768](https://arxiv.org/pdf/2508.17768); [Curated BUSI](https://portalinvestigacion.uniovi.es/documentos/69c416ee51799c7f541aa040)
* BUS-UCLM: [Mendeley Data](https://data.mendeley.com/datasets/7fvgj4jsp7), [Sci Data 2025](https://www.nature.com/articles/s41597-025-04562-3)
* BUS-CoT: [arXiv 2509.17046](https://arxiv.org/abs/2509.17046v1)
* GDPH&SYSUCC: [HoVer-Trans, arXiv 2205.08390](https://arxiv.org/pdf/2205.08390)
* CADBUSI (not public): [J Digit Imaging 2025](https://link.springer.com/article/10.1007/s10278-025-01784-0)
* Cross-site gap: [Breast Ultrasound AI Under Dataset Shift, Diagnostics 2026](https://opinvisindi.is/items/436637f1-2eb8-4bbe-8064-afc136105896); [Training Set Diversity…](https://research.mpu.edu.mo/en/publications/training-set-diversity-a-key-factor-inai-driven-breast-ultrasound/)
* Conformal / biopsy reduction: [PersonalizedUS, AAAI 2025 / arXiv 2408.15458](https://arxiv.org/html/2408.15458v1); [AI downgrading of BI-RADS 4A](https://www.hospimedica.com/artificial-intelligence/articles/294787437/adding-artificial-intelligence-ai-system-to-breast-ultrasound-can-reduce-unnecessary-biopsies-finds-study.html)
* Low-resource triage: [RSNA news on Berg et al., Radiology 2023](https://www.rsna.org/news/2023/may/ai-for-triaging-breast-masses); [Systematic review of AI handheld breast US](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12453205/); [NCT07462351](https://clinicaltrials.gov/study/NCT07462351); [NCT06932133](https://clinicaltrials.gov/study/NCT06932133); [NCT05561842](https://clinicaltrials.gov/study/NCT05561842); [AI-enabled POCUS, South Africa](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12587214/)
* Policy: [WHO GBCI framework 2023](https://www.iccp-portal.org/system/files/resources/9789240065987-eng.pdf); [BHGI resource stratification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5493239/)
* Mammography AI: [MASAI final results, Lund University](https://www.lunduniversity.lu.se/article/ai-support-breast-cancer-screening-fewer-missed-cancer-cases); [MASAI 2025 Lancet Digital Health](https://www.itnonline.com/content/ai-enhanced-mammography-screening-significantly-improves-breast-cancer-detection-reduces)
* Open ultrasound encoders: [EchoCare](https://arxiv.org/pdf/2509.11752); [OpenUS](https://arxiv.org/abs/2511.11510v1); [SonoBase](https://arxiv.org/pdf/2609.19230)
* Option B: [BCSC Risk Factors dataset](https://www.bcsc-research.org/datasets/rf); [BCSC screening performance dataset](https://www.bcsc-research.org/index.php/datasets/mammo-screen-data)
* Option D: [NHANES LUX_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/LUX_L.htm); [SAFE score in NHANES](https://www.elsevier.es/en-revista-annals-hepatology-16-articulo-the-steatosis-associated-fibrosis-estimator-safe--S1665268124003107); [LiverRisk/SAFE/FIB-4 comparison](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11357697/)
