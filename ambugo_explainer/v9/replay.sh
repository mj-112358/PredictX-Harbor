#!/usr/bin/env bash
# Regenerate data/ from AmbuGo_First_Draft_Reproducibility_v9.zip.
#   bash replay.sh /path/to/AmbuGo_First_Draft_Reproducibility_v9.zip
# The frozen controller is copied into a scratch working copy and imported unmodified;
# instrument.py only adds logging (and, for the branch, drops one request).
set -euo pipefail
ZIP=$(realpath "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
pip install -q "eclipse-sumo==1.28.0" "libsumo==1.28.0" "sumolib==1.28.0" "traci==1.28.0" numpy

unzip -q "$ZIP" -d "$WORK/x"
SRC=$(find "$WORK/x" -maxdepth 2 -type d -name 'AmbuGo_First_Draft_Reproducibility_v9' | head -1)
R="$WORK/replay"
mkdir -p "$R/outputs" "$R/work/v9/vendor"
cp -r "$SRC/outputs/AmbuGo_v7_Synthetic_Data" "$R/outputs/"
cp "$SRC/work/v9/sumo_study.py" "$SRC/work/v9/selected_policy.json" "$R/work/v9/"
cp -r "$SRC/work/v9/sumo_network" "$R/work/v9/"
ln -s "$(python3 -c 'import sumo, os; print(os.path.dirname(sumo.__file__))')" "$R/work/v9/vendor/sumo"
cp "$HERE/instrument.py" "$R/work/v9/"

echo "frozen controller sha256 (expect 471a2291...cecf2):"
sha256sum "$R/work/v9/sumo_study.py"

REC="$SRC/work/v9/sumo_main/records.json"
cd "$R/work/v9"
python3 instrument.py --seed 40 --load 0.95 --missions 1 --policy P3 --out "$WORK/ep/P3" --records "$REC"
python3 instrument.py --seed 40 --load 0.95 --missions 1 --policy P0 --out "$WORK/ep/P0" --records "$REC"
python3 instrument.py --seed 40 --load 0.95 --missions 1 --policy P3 --out "$WORK/ep/P3_withholdJ21" \
        --branch withhold:J21:E_J11_J21:4132

D="$HERE/data"
mkdir -p "$D/episodes" "$D/network" "$D/fixture" "$D/archive"
for n in P3 P0 P3_withholdJ21; do rm -rf "$D/episodes/$n"; cp -r "$WORK/ep/$n" "$D/episodes/$n"; done
cp "$SRC/work/v9/sumo_network/district.net.xml" "$D/network/"
cp "$SRC"/outputs/AmbuGo_v7_Synthetic_Data/*.csv "$D/fixture/"
cp "$SRC/work/v9/selected_policy.json" "$D/archive/"
cp "$SRC/work/v9/paper_sumo_results.txt" "$D/archive/" 2>/dev/null || true
cp "$REC" "$D/archive/sumo_main_records.json"
python3 - "$REC" "$SRC/work/v9/sumo_analysis.json" "$D/archive" <<'EOF'
import json, sys
rec = json.load(open(sys.argv[1]))
json.dump([r for r in rec if r["seed"] == 40 and r["load"] == 0.95 and r["missions"] == 1],
          open(sys.argv[3] + "/records_s40_x095_m1.json", "w"), indent=1)
a = json.load(open(sys.argv[2]))
json.dump({k: a[k] for k in ("aggregate", "aggregate_contrasts")}, open(sys.argv[3] + "/sumo_analysis_aggregate.json", "w"), indent=1)
EOF
echo "data/ regenerated; now: python3 verify.py && python3 film.py"
