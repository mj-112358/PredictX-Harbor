# Changelog

Running log of what changed in the project and why. Newest first.

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
