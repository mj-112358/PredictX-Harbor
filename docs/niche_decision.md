# Niche decision memo

**Version 2 (2026-10-22), for author review before the protocol is frozen.**
v1 (lesion triage, Option A as first framed) is kept in git history (commit `e88cc56`). This version
reflects two author decisions: Track A (breast ultrasound), then a change of scope to **first-step
screening of asymptomatic women**.

Status markers: **verified** (official documentation read), **reported** (from a paper abstract or
secondary source; full text was blocked in this session), **unknown**.

## 1. Decision record

| Date | Decision | By |
|---|---|---|
| 2026-10-22 | Track A (breast ultrasound) over B (BCSC risk), C (screening mammography) and D (NHANES liver) | author |
| 2026-10-22 | Not lesion triage. The device must support screening and carry a novel contribution | author |
| 2026-10-22 | Use case: health worker does a handheld whole-breast sweep with or after CBE; AI outputs NORMAL / RECALL / UNREADABLE per woman; primary endpoints at screening prevalence | author |

## 2. The use case in one paragraph

At an NP-NCD screening camp, a health worker examines a woman aged 30–65 (CBE) and then sweeps both
breasts with a handheld ultrasound probe. The AI checks the scan is adequate (else UNREADABLE →
rescan) and looks for persistent suspicious lesions across the sweep. It returns **NORMAL** or
**RECALL** for the woman. RECALL sends her to a facility with a radiologist. A positive CBE is
always referred. The product question: **does adding an AI-read handheld sweep to CBE find
substantially more cancers at an acceptable recall rate (≤ 5–10%), when operated by health workers?**

## 3. Why this is a real gap (positioning, India)

| Tool | Modality | Who operates | Evidence (screening-relevant) | Limitation the proposal addresses |
|---|---|---|---|---|
| **CBE (NP-NCD standard of care)** | Palpation | ASHA/ANM, nurses | Mumbai cluster RCT (151,538 women, 20 years): ~30% lower breast-cancer mortality in women ≥ 50 (subgroup; the all-ages reduction was not significant) (**reported**, Mittra et al., BMJ 2021) | Misses non-palpable cancers; uptake is very low (~1% of women aged 30–49 report ever having had CBE, NFHS-5 analysis, **reported**) |
| **iBreastExam** (UE LifeSciences) | Tactile pressure sensor, handheld | Health workers | Bangalore, 1,019 women: sens 84%, spec 94% vs CBE 65%/94% (developer-run study, **reported**). Nigeria (Lancet Glob Health 2022, nurses): sens 86% vs CBE 83%, **spec 50% vs CBE 86%** (**reported**). Large deployment (millions of scans, press-reported) | Tactile, not imaging; detects masses like palpation does; low specificity in independent evaluation |
| **Thermalytix** (Niramai) | Thermal imaging + AI | Technician | Punjab state programme: 15,069 women, 183 locations (npj Digit Med 2024, **reported**; outcome figures not retrieved) | Thermal physiology, not anatomy; evidence mostly company-associated |
| **Mammography** | X-ray | Radiographer + radiologist | Strong RCT evidence in HICs | ~1.3% use among women ≥ 45 in India (LASI, **reported**); needs fixed infrastructure and radiologists |
| **AI handheld ultrasound** (Koios, research systems) | B-mode ultrasound | Mostly sonographers | Mexico (Berg et al., Radiology 2023): palpable lumps, 300 women, **diagnostic not screening**. Systematic review (PLOS Digit Health 2025, 34 studies): 79% high/unclear risk of bias; no validated screening use | **No health-worker-operated, AI-read whole-breast ultrasound screening study in asymptomatic women with screening-prevalence endpoints was found** |
| Ultrasound as a screening modality | — | Radiologists | China multicentre RCT, high-risk women (Shen et al., BJC 2015): ultrasound detected 14/14 cancers vs mammography 8/14 in the combined arm (**reported**) | Requires radiologists; the operator bottleneck is what the AI addresses |

**Gap statement:** Ultrasound sees non-palpable cancers that CBE and tactile devices can't, but in
India it is operator- and radiologist-limited. No published system has been validated for
*health-worker-operated, AI-read whole-breast ultrasound screening* with recall-rate-anchored
endpoints. This proposal targets exactly that, and adds a method to certify the operating point
when the device moves to a new site.

**The measurable contribution (testable claims):**
1. **C1, certified recall operating point under site shift:** an exact recall ceiling plus
   sensitivity floor for a new site, with a spectrum-weighted sensitivity bound for enriched cancer
   samples (new), and a hierarchical borrowing-strength method hypothesised to halve the local
   cancers needed (new; H1).
2. **A screening-ready pipeline** (quality gate → frame detector → persistence-based woman-level
   aggregation), developed on public data across countries, with a quality-reject option.
3. **A pre-registered, exactly powered field-study design** (Section 8 of the protocol) whose
   primary comparison is CBE + AI vs CBE alone at a fixed recall rate.

What it is not: not a mammography competitor, not a malignancy classifier, and not a validated
screening device until the field study is done.

## 4. Data reality (the main limitation, stated plainly)

* **No public whole-breast sweep dataset and no public asymptomatic screening ultrasound dataset
  exists** (search of literature and dataset atlases, 2026-10-22). The only public breast video set
  (BreastVid, 188 clips) is lesion-centred, with no normal women. ABUS volumes (TDSC-ABUS, 200) are
  tumour-focused.
* Public still-image sets (BUSI, BUS-UCLM, BUS-BRA, BrEaST, BUS-CoT and others) are **for development
  only**. They cannot measure screening sensitivity, recall rate or false recalls per 1,000. Only
  BUSI (133) and BUS-UCLM (419 images, 38 patients) contain a meaningful number of normal frames, so
  even the frame-level false-positive rate can only be estimated at two sites.
* Therefore **the core screening evidence must come from new data.** The protocol adds a pilot
  (~300 camp women + ~60 cancers) to fix the sweep protocol, aggregation and thresholds, and then the
  frozen enriched field study (~3,000 camp women + ~150 cancers + ~150 benign).
* **What this project can deliver without new data:** a detector and quality gate developed and
  stress-tested across 4–6 countries' still images; the C1 method with its lesion-level proxy
  evaluation; and the pre-registered, exactly powered field-study protocol. A paper written now
  would be a *methods-and-protocol* paper with development evidence, not a screening-accuracy paper.

## 5. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **PC-PNDT Act** restricts ultrasound machines to registered premises and regulates operators. Health workers carrying handhelds to camps may be unlawful without registration or exemption | **High** (could block deployment in India) | Legal advice and state PC-PNDT authority engagement before the pilot; camps at registered facilities or registered mobile units; check whether breast-only devices can be exempted |
| No public sweep data → false-recall behaviour of aggregation unknown until the pilot | High | Pilot before freezing field-study thresholds; persistence-based aggregation |
| Spectrum bias: enriched cancers larger and more palpable than screen-detected ones | High | Spectrum-stratified recruitment; spectrum-weighted sensitivity (C1c); camp cancers as a check |
| Benign lesions and cysts trigger RECALL, inflating false recalls (cysts are common at 30–50) | Medium | Detector trained to flag *suspicious* lesions; simple-cyst handling decided in the pilot; benign arm measures it |
| Low-end handheld image quality | Medium | Quality gate; degradation robustness; device-specific certification (C1) |
| Pretrained encoder licences and access (Hugging Face blocked here) | Low/medium | Network access change requested; fallbacks listed |

## 6. What I need from you

1. **Approve or amend** the protocol draft (`docs/protocol.md` v0.10). The key choices to confirm:
   * sensitivity claim target: **> 0.80 with 150 cancers** (a > 0.85 claim needs 185–380 cancers);
   * recall operating points: **OP-5 and OP-10**;
   * primary comparison: **CBE + AI vs CBE alone at OP-10** (85% power for +10 points with 150 cancers);
   * HARBOR and the VoI controller moved out of the primary pipeline (one exploratory HARBOR test kept).
2. **Network access.** Still blocked in this session (every data host returns 403). If you changed
   the setting, please start a new session on this branch, since the change likely applies only to new
   containers. Hosts needed: `huggingface.co`, `cdn-lfs.huggingface.co`, `*.hf.co`, `zenodo.org`,
   `figshare.com`, `ndownloader.figshare.com`, `data.mendeley.com`, `www.cancerimagingarchive.net`,
   `nbia.cancerimagingarchive.net`. GitHub is already reachable (BreastVid / CVA-Net repository).
3. **Field-data partner (for later).** The screening claim can't be made without one: a hospital or
   NGO running NP-NCD camps, with a cancer centre for the enriched arm. Even a small retrospective set of
   *stored handheld sweeps from asymptomatic women* would be more valuable than any public dataset.

### 6.1 Datasets to download (all files of each record)
1. BUSI original (Al-Dhabyani 2020; Mendeley `k8t3gnx9h6`) **and** the curated release (Aumente-Maestro et al., 2025)
2. BUS-UCLM: https://doi.org/10.17632/7fvgj4jsp7.3
3. BUS-BRA: https://doi.org/10.5281/zenodo.8231412
4. BrEaST / Breast-Lesions-USG (TCIA) or the link in Pawłowska et al., *Sci Data* 2024
5. BUS-CoT: figshare https://doi.org/10.6084/m9.figshare.30838715 (atlas) or 10.6084/m9.figshare.29036876 (arXiv v2), whichever is current
6. BreastVid: per github.com/jhl-Det/CVA-Net (download link in the README)
7. Optional: BUSI_WHU (Mendeley `k6cpmwybk3/3`), BUS_UC (Mendeley `3ksd7w7jkx/1`), TDSC-ABUS 2023 (application form)
8. Weights: `facebook/dinov2-base`; `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`; TabPFN v2 classifier checkpoint (only for the tabular-head comparison)

## 7. Sources

* Public datasets: [open-medical-ultrasound-datasets atlas](https://github.com/TIanCat/open-medical-ultrasound-datasets); [BreastVid / CVA-Net, MICCAI 2022](https://arxiv.org/pdf/2207.00141); [TDSC-ABUS](https://arxiv.org/abs/2501.15588v1); [BUS-UCLM](https://www.nature.com/articles/s41597-025-04562-3); [BrEaST](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10830496/); [BUS-BRA](https://github.com/wgomezf/BUS-BRA); [BUS-CoT](https://arxiv.org/abs/2509.17046v1); [BUSI defects](https://pmc.ncbi.nlm.nih.gov/articles/PMC10293973)
* India programme and uptake: [NP-NCD screening and uptake study (BMC Public Health 2025, via The Week)](https://www.theweek.in/wire-updates/national/2025/04/03/lst2-research-cancer-screening.html); [AI screening not in NP-NCD (Rajya Sabha reply, via CliniExperts)](https://cliniexperts.com/regulatory-update/government-clarifies-status-of-ai-based-breast-cancer-screening-under-the-national-programme-for-prevention-and-control-of-non-communicable-diseases-np-ncd/); [DHR HTA on breast cancer screening](https://dhr.gov.in/sites/default/files/HTA%20_Breast%20Cancer%20Screening.pdf)
* CBE evidence: [Mittra et al., BMJ 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7977439/)
* iBreastExam: [Lancet Glob Health 2022 (Nigeria)](https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(22)00030-4/fulltext); [Annals of Oncology (India three-arm study)](https://www.annalsofoncology.org/article/S0923-7534(19)30468-5/fulltext); [validation study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5084434/)
* Thermalytix: [npj Digital Medicine 2024, Punjab](https://www.nature.com/articles/s41746-024-01368-2)
* Ultrasound screening: [Shen et al., BJC 2015](https://dash.harvard.edu/bitstream/handle/1/15034941/4366890.pdf?sequence=1); [AI handheld breast US systematic review](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001019); [Berg et al. 2023 via RSNA](https://www.rsna.org/news/2023/may/ai-for-triaging-breast-masses); [AI US system without sonographer, Sci Rep 2025](https://www.nature.com/articles/s41598-025-29554-6)
* PC-PNDT: [Delhi PC-PNDT key directions](https://dcwestrev.delhi.gov.in/sites/default/files/dcwestrev/generic_multiple_files/key_direction.pdf); [Delhi PC-PNDT orders](https://pcpndt.delhi.gov.in/orders-and-circulars); [PCPNDT and cancer care commentary](https://kpiasacademy.com/pcpndt-act-cancer-care-reforms/)

## Appendix: v1 option comparison (summary)
B was rejected because the public BCSC files have no individual-level outcome. C was rejected for
needing hundreds of GB and a GPU, in a crowded field. D (NHANES fibrosis) fits the old acquisition
controller best but is a different disease. Full v1 text is in commit `e88cc56`.
