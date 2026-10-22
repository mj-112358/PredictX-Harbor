"""Build the unified record table.

    python -m sweepscreen.data.build [--sites BUSI BUS-BRA ...]

Loads every site whose folder exists under ``data/raw``, hashes images after annotation removal and
field-of-view cropping, assigns groups (patient + video + near-duplicate components), and writes

* ``data/processed/records.csv``: one row per image/frame, schema of :mod:`sweepscreen.data.schema` plus
  ``group_id``, ``dup_cluster``, ``discordant``, ``cross_site``;
* ``data/processed/dedup_summary.json``: per-site duplicate statistics.

Records in label-discordant duplicate clusters are kept in the table but flagged; experiments
exclude them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..preprocess import crop_to_content, remove_annotations
from . import dedup
from .loaders import LOADERS
from .loaders.common import read_raw
from .schema import to_frame

ROOT = Path(__file__).resolve().parents[2]
RAW, EXTRACTED, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "extracted", ROOT / "data" / "processed"


def _hash_input(path: str):
    gray = remove_annotations(read_raw(path))
    return crop_to_content(gray).apply(gray)


def build(sites=None, max_distance: int = 6, frame_stride: int = 10, raw: Path = RAW, extracted: Path = EXTRACTED,
          processed: Path = PROCESSED) -> pd.DataFrame:
    raw, extracted, processed = Path(raw), Path(extracted), Path(processed)
    records = []
    for site, (module, folder) in LOADERS.items():
        if sites and site not in sites:
            continue
        if not (raw / folder).exists():
            print(f"[skip] {site}: {raw / folder} not present")
            continue
        recs = module.load(str(raw / folder), str(extracted / folder))
        print(f"[load] {site}: {len(recs)} records")
        records += recs
    if not records:
        raise SystemExit("no dataset found under data/raw (see docs/data_request.md)")
    hashes = {}
    for r in records:
        if r.video_id is not None and (r.frame_index or 0) % frame_stride:
            continue
        hashes[r.uid] = dedup.phash(_hash_input(r.image_path))
    groups = dedup.assign_groups(records, hashes, max_distance)
    table = to_frame(records).merge(groups[["uid", "group_id", "dup_cluster", "discordant", "cross_site"]], on="uid")
    processed.mkdir(parents=True, exist_ok=True)
    table.to_csv(processed / "records.csv", index=False)
    (processed / "dedup_summary.json").write_text(json.dumps(
        {"max_hamming": max_distance, "frame_stride": frame_stride, "sites": dedup.summary(groups)}, indent=1))
    return table


def load_table(processed: Path = PROCESSED) -> pd.DataFrame:
    return pd.read_csv(Path(processed) / "records.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sites", nargs="*")
    parser.add_argument("--max-hamming", type=int, default=6)
    args = parser.parse_args(argv)
    t = build(args.sites, args.max_hamming)
    print(t.groupby(["site", "label"]).size().to_string())


if __name__ == "__main__":
    main()
