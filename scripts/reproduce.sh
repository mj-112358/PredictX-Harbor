#!/bin/bash
# Rebuild every result, table, figure and number of the SweepScreen paper from the raw data.
# Needs the files listed in docs/data_request.md under data/raw/ and models/pretrained/dinov2-small/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

python -m sweepscreen.data.manifest build               # SHA-256 of every input file -> data/MANIFEST.sha256
python -m experiments.run_features --rebuild            # record table, dedup, clean + augmented feature stores
python -m experiments.run_loso                          # leave-one-site-out + internal CV
python -m experiments.run_loso --no-aug --no-internal --results results/ablation_noaug   # augmentation ablation
python -m experiments.run_video                         # temporal aggregation on BreastVid
python -m experiments.run_quality                       # quality gate per held-out site
python -m experiments.run_certify                       # threshold certification at a new site
python -m experiments.run_robustness                    # handheld-probe degradations
python -m experiments.run_robustness --results results/ablation_noaug   # same, models trained without augmentation
python -m experiments.train_final                       # deployable model -> models/registry.json
python -m experiments.run_runtime                       # CPU timings with the registered model
python -m experiments.report                            # tables, figures, numbers.json, macros.tex
python -m pytest -q
(cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
echo "done: paper/main.pdf"
