#!/bin/bash
# Reproduce every result, table and figure of the paper, then build the manuscript.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

for c in heart diabetes breast liver kidney; do python -m experiments.benchmark --cohort "$c"; done
for c in diabetes breast liver; do
  python -m experiments.benchmark --cohort "$c" --protocol half --models LR SVM KNN NB RF MLP XGB CAT PFN HARBOR
done
for c in liver kidney; do python -m experiments.benchmark --cohort "$c" --features web; done
python -m experiments.acquisition
for d in WDBC ILPD; do for m in MCAR MNAR; do python -m experiments.missingness --data "$d" --mechanism "$m"; done; done
python -m experiments.report
python -m service.train_models
python -m pytest
(cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
echo "done: paper/main.pdf"
