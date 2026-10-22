"""Paths and helpers shared by the experiment scripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sweepscreen.data.build import PROCESSED, load_table
from sweepscreen.features import FEATURES, FeatureStore
from sweepscreen.train import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def parser(doc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", type=Path, default=PROCESSED)
    p.add_argument("--features", type=Path, default=FEATURES / "clean")
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--fast", action="store_true", help="tiny settings for smoke tests")
    return p


def load(args):
    table = load_table(args.processed)
    store = FeatureStore(args.features)
    if list(store.index.uid) != list(table.uid):
        raise SystemExit("feature store rows do not match records.csv; re-run experiments.run_features")
    return table, store


def config(args) -> TrainConfig:
    return TrainConfig(detector_epochs=3, video_stride=1, malignancy_video_stride=1) if args.fast else TrainConfig()


def save_predictions(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(path, orient="records", lines=True)


def read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_json(path, orient="records", lines=True)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
