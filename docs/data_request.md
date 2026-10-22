# Data and model weights needed for SweepScreen

Put each item, exactly as downloaded (zips are fine), under `data/raw/<folder>/`. The loaders read
these layouts directly, and `python -m sweepscreen.data.manifest build` records a SHA-256 for every
file.

| # | Folder | Dataset | What to download | Expected layout (as documented) | Licence / terms |
|---|---|---|---|---|---|
| 1 | `data/raw/busi/` | **BUSI** (Al-Dhabyani et al., *Data in Brief* 28:104863, 2020; Egypt) | `Dataset_BUSI.zip` from Mendeley Data `k8t3gnx9h6` v1 (https://data.mendeley.com/datasets/k8t3gnx9h6/1) | `Dataset_BUSI_with_GT/{benign,malignant,normal}/<class> (N).png` and `<class> (N)_mask.png` (plus `_mask_1.png` for a second lesion); 437 / 210 / 133 images | CC BY 4.0 |
| 2 | `data/raw/busbra/` | **BUS-BRA** (Gómez-Flores et al., *Med Phys* 51:3110, 2024; Brazil) | **All files** of Zenodo record https://doi.org/10.5281/zenodo.8231412 | images, masks, a case-level CSV (ID, case/patient, histology, pathology, BI-RADS, device, …), fold-partition CSVs; 1,875 images / 1,064 patients | as stated on Zenodo (please note it) |
| 3 | `data/raw/breast/` | **BrEaST / Breast-Lesions-USG** (Pawłowska et al., *Sci Data* 11:148, 2024; Poland) | From the TCIA collection page "Breast-Lesions-USG": the **images-and-masks** archive **and** the **clinical-data spreadsheet** (.xlsx) | PNG images + tumour masks; spreadsheet with age, tissue composition, symptoms, BI-RADS, descriptors, verification, histology; 256 patients | CC BY 4.0 |
| 4 | `data/raw/udiat/` | **UDIAT Dataset B** (Yap et al., *IEEE JBHI* 22:1218, 2018; Spain) | Request through the form at https://helward.mmu.ac.uk/STAFF/m.yap/dataset.php (needs a signed agreement) | `original/*.png`, `GT/*.png`, and a spreadsheet with benign/malignant labels; 163 images (110 benign / 53 malignant) | research use under the signed agreement |
| 5 | `data/raw/breastvid/` | **BreastVid / CVA-Net "miccai_buv"** (Lin et al., MICCAI 2022; China) | Google Drive file `1LVXK34OJhC2LkqqyMmVFnXsXQavvZdeF` (link in github.com/jhl-Det/CVA-Net README; Baidu mirror also listed) | `miccai_buv/rawframes/{benign,malignant}/<video_id>/*.png`, `train.json`, `val.json` (COCO-style boxes); 188 videos, ~25k frames | non-commercial research and education |
| 6 *(recommended)* | `data/raw/busuclm/` | **BUS-UCLM** (Vallez et al., *Sci Data* 2025; Spain) | All files of https://doi.org/10.17632/7fvgj4jsp7.3 | 683 images incl. **419 normal**, RGB masks | CC BY 4.0 |

Why item 6: of items 1–5, only BUSI contains normal (lesion-free) images (133). The quality gate and
the detector's lesion-free class are much better trained and tested with BUS-UCLM's 419 normal frames
added. Everything runs without it.

Known corrections the BreastVid loader applies automatically (from the CVA-Net README, 2023-03-25):
drop `rawframes/benign/x66ef02e7f1b9a0ef` (a duplicate of `malignant/x3b88488853e8b7d1`); relabel
`benign/x63c9ba1377f35bf6` and `benign/x5a1c46ec6377e946` as malignant; group those two with
`malignant/2390fbea047347b` and `malignant/7a39ab5d4970bf89` as one patient.

## Model weights

| Folder | Model | Files | Version / licence |
|---|---|---|---|
| `models/pretrained/dinov2-small/` | **DINOv2 ViT-S/14** (Oquab et al., 2023), the single pretrained image encoder (22M parameters; runs on CPU) | From https://huggingface.co/facebook/dinov2-small: `config.json`, `model.safetensors`, `preprocessor_config.json` | Apache-2.0. Any current revision; its SHA-256 is recorded in the manifest. |

**TabPFN is not used.** The malignancy head is an L2-regularised logistic regression on the image
embedding plus available metadata. That keeps the stack CPU-only and dependency-light, and avoids
the TabPFN licence's attribution requirement in the user interface. The Phase 0 audit found TabPFN
gave no significant gain over simpler models on small tables.

## Software (already pinned in `requirements.txt`)
Python 3.13; torch 2.5.1 (CPU use); transformers 4.57.1 (DINOv2 implementation, loaded from the local
folder above, with no network access); opencv-python-headless, imagehash, scikit-learn, FastAPI.

## Alternative: allow the hosts instead of downloading by hand
`data.mendeley.com`, `zenodo.org`, `www.cancerimagingarchive.net`, `nbia.cancerimagingarchive.net`,
`drive.google.com`, `drive.usercontent.google.com`, `huggingface.co`, `cdn-lfs.huggingface.co`,
`*.hf.co`. (UDIAT still needs the signed request.) Network changes apply to new sessions.
