# Phase 0: audit and reproduction

Date: 2026-10-22. Code at commit `458111e` (identical to the attached `PredictX-Harbor-main.zip`,
checked with a recursive diff). The paper was read from `paper/sections/*.tex` and `paper/main.pdf`,
because the separate PDF was not attached. Every re-run below wrote to a scratch directory, so
nothing under `results/` or `paper/` was modified.

## 1. Environment

| Item | Finding |
|---|---|
| Compute | 4 CPU cores, 15 GB RAM, ~30 GB free disk, no GPU |
| Python | 3.13; all pins in `requirements.txt` installed from PyPI into `.venv` (scikit-learn 1.6.1 as pinned) |
| Tests | `pytest`: 13 passed before the first failure. The failure is `test_service.py::test_predict_returns_a_probability[heart]`, raised because the TabPFN v2 weights can't be downloaded (huggingface.co answers 403). Every test that loads a TabPFN-containing artifact fails for the same reason. **No test failed for a code reason.** |
| Network | PyPI, GitHub and raw.githubusercontent.com reachable. **Blocked (403 at the proxy):** huggingface.co, download.pytorch.org, dl.fbaipublicfiles.com, zenodo.org, figshare.com, data.mendeley.com, TCIA, kaggle.com, physionet.org, archive.ics.uci.edu, CDC/NHANES, bcsc-research.org, PubMed/PMC, arxiv.org |

## 2. Reproduction

| Check | Result |
|---|---|
| `experiments/report.py` regenerated from the stored per-split predictions | All 9 `tab_*.tex` and `numbers.json` **byte-identical** to `paper/generated/`. The table arithmetic is correct given the stored predictions. |
| Split identity (breast grouped CV under sklearn 1.6.1; heart CV; breast 50/50) | Test-row sets identical for every (rep, fold) |
| Breast P1, re-fitted from scratch | LR 0.9945, CatBoost 0.9924 (**bit-identical predictions**); HARBOR 0.9922 vs 0.9922 stored (predictions not bit-identical: mean abs. difference ≈ 0.001, max 0.10, from nondeterministic training) |
| Breast P2 (50/50) | LR 0.9956, CatBoost 0.9938 identical; HARBOR 0.9931 vs 0.9932 |
| Heart P1 | LR 0.9092, CatBoost 0.9027, LR+aux 0.9113, CatBoost+aux 0.9039 identical; HARBOR 0.9096 vs 0.9092 (+0.0004); HARBOR+aux 0.9140 vs 0.9139 |
| Acquisition P3 (all 25 splits × 21 policies) | **Exactly reproduced**: full panel 0.9027 AUROC at cost 5.0; VoI (information, λ=0) 0.8855 at cost 2.66 (98.1% of full-panel AUROC at 53.3% of cost) |
| TabPFN v2, and therefore every PredictX ensemble number | **Not reproducible in this session**: the weights download is blocked. The stored TabPFN predictions were used, unverified, for the analyses in Section 3. |

**Discrepancies:** none beyond HARBOR's run-to-run nondeterminism (≤ 0.0004 AUROC in mean). The
stored seeds alone don't make HARBOR bit-reproducible on this machine. This should be fixed (or
stated) before any HARBOR number is quoted to four decimals.

## 3. The review's weaknesses, checked

Paired differences are over the 25 P1 folds, with 95% Nadeau–Bengio corrected intervals,
computed from the stored predictions.

| # | Weakness | Verdict | Evidence |
|---|---|---|---|
| 1 | Breast endpoint is the wrong problem; everything scores ≈ 0.99 | **Confirmed** | LR 0.9945, SVM 0.9948, TabPFN 0.9950, PredictX 0.9943. PredictX − LR = −0.0002 [−0.0028, +0.0024]. Data: 699 rows, 645 sample IDs (46 repeated, up to 6 rows per ID), 4 IDs with conflicting labels, 284 rows that duplicate another row's features and label exactly (discrete 1–10 scores). |
| 2 | No niche | **Confirmed** | Five unrelated 1988–2000s benchmark cohorts; no clinical pathway, user or operating point is defined anywhere in code or paper. |
| 3 | Typed-biomarker headline is switched off | **Confirmed, and worse than stated** | These components aren't zero-weighted; **they aren't implemented at all.** `predictx/harbor.py` and `preprocessing.py` contain no time, specimen, assay, unit or quality embedding, no clinical-graph prior (`α_d A_prior`), no `κ` prior term, no reliability loss `L_rel`, no consistency loss, and no typed missingness reasons `e_μ` (a single `e_miss` vector is used). The reliability gate exists, but its input is only `[h0; m]`. The paper's Section 4 equations describe a model that was never built. |
| 4 | HARBOR doesn't measurably help | **Confirmed** | Ensemble minus ensemble-without-HARBOR: heart +0.0000 [−0.0042, +0.0042], diabetes +0.0009 [−0.0024, +0.0041], breast −0.0002 [−0.0010, +0.0006], liver −0.0015 [−0.0089, +0.0059], kidney 0. Controlled missingness (P4): two-channel − value-only on WDBC-MNAR +0.013 [−0.001, +0.027], and mask-concat ≥ two-channel in all 4 settings. **New point:** plain logistic regression with median imputation plus indicators beats *every* HARBOR variant in all four P4 settings (e.g. WDBC-MNAR LR 0.988 vs two-channel 0.979). |
| 5 | The ensemble is essentially TabPFN | **Confirmed** | PredictX − TabPFN: heart +0.0001 [−0.0081, +0.0082], diabetes −0.0017, breast −0.0007, liver −0.0015, kidney 0.0000. No interval excludes zero. TabPFN's mean is ≥ PredictX's on 4 of 5 cohorts (kidney tied). An average of TabPFN and LR alone matches the four-member ensemble everywhere (differences −0.0015 to 0.0000). |
| 6 | Propositions are textbook | **Confirmed** | Prop. 1 (cost) is the standard attention complexity; Prop. 2 (ℓ1 under softmax) holds because rows sum to 1; Prop. 3 is standard entropy calculus; Prop. 4 is Jensen's inequality plus the Brier variance decomposition. All correct, none new. The tests in `tests/test_harbor.py` and `test_ensemble.py` check these identities numerically. |
| 7 | No external validation, prospective component, calibration, DCA or subgroups | **Confirmed** | Only internal CV (P1) and a resampled 50/50 protocol (P2). The "cross-site" experiment adds the HU/CH/VA cohorts to *training* and still tests on Cleveland, so it isn't external validation. Calibration is reported only as the Brier score; there are no reliability curves, calibration intercept or slope, decision curves or subgroup analyses. |
| 8 | "If TabPFN alone matches you and HARBOR adds nothing, what is the contribution?" | **Confirmed as the central problem** | Follows from 3–5. |

### Additional problems found during the audit

1. **The headline coronary result compares unequal training data.** "PredictX attains the highest
   AUROC for coronary disease (0.915)" and the coronary column of the paired-comparison table use
   PredictX *with* the three auxiliary heart cohorts, against comparators trained on Cleveland only.
   Given the same auxiliary data, TabPFN scores 0.9134 against PredictX's 0.9146.
2. **The value-of-information controller doesn't beat trivial strategies at matched cost on the
   paper's own metric.** Using the same per-pattern CatBoost predictors and splits, I compared
   VoI with *random mixtures of adjacent fixed panels* (each patient randomly gets panel A or B, so
   that the mean cost matches):

   | Mean cost | VoI AUROC | Fixed-panel mixture AUROC | VoI accuracy | Mixture accuracy |
   |---|---|---|---|---|
   | 0.17 (decision, λ=0.1) | 0.832 | 0.837 | 0.781 | 0.772 |
   | 0.83 (decision, λ=0.02) | 0.852 | 0.861 | 0.801 | 0.784 |
   | 1.72 (information, λ=0.02) | 0.872 | 0.879 | 0.805 | 0.802 |
   | 2.66 (information, λ=0, paper headline) | 0.886 | 0.894 | 0.822 | 0.822 |

   On **AUROC** the mixture beats VoI at every one of the 13 VoI operating points. On **accuracy**
   (closer to what a decision utility optimises), the decision-utility VoI gains about 1–2 points at
   low budgets (cost 0.17–1.2), and the information-utility VoI ties at the headline point. These
   are point estimates over 25 folds, without intervals. Conclusion: the "98% of AUROC at 53% of
   cost" claim doesn't show added value over simple panels. A small decision-level benefit at low
   budgets is plausible but unproven. A naive uncertainty-band rule (Ex → Fl → Th while
   |p − 0.5| < w) is *worse* than VoI, so VoI is better than a bad heuristic, but not better than a
   simple one.
3. **The "exceeds the best published result" claims (P2)** compare against the 2022 TabPFN v1
   benchmark, with different random splits. On breast, PredictX (0.9953) is *below* its own TabPFN v2
   member (0.9955); on liver, PredictX (0.7438) is above TabPFN v2 (0.7430) by 0.0008 over 5 splits.
   Any margin over the published numbers comes from TabPFN v2, not from the paper's method.
4. **"Significantly better in 12 of 50 comparisons"**: all 12 wins are against the weaker comparators
   (naive Bayes ×3, k-NN, RF and MLP ×2 each, LR, tuned XGBoost and CatBoost ×1 each). None is against
   TabPFN or HARBOR.
5. **HARBOR training is not bit-reproducible** (Section 2).

## 4. Implications for the rewrite

* The four-member ensemble can't be the contribution. Any new method must be compared directly
  against TabPFN alone and against LR, at clinically meaningful operating points.
* HARBOR has never been tested on the problem it was designed for (real, heterogeneous metadata). It
  either gets a fair test on such data (Phase 4, hypothesis 1) or is dropped.
* The acquisition controller needs matched-cost baselines (fixed panels, their mixtures, simple
  sequential rules) and a decision-relevant metric (sensitivity-constrained specificity, net
  benefit) before it can be claimed as a contribution.
* The breast cytology endpoint should be dropped (see `docs/niche_decision.md`).
