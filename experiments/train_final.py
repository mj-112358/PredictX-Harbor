"""Fit the deployable model on every site and register it.

    python -m experiments.train_final

Writes ``models/sweepscreen-<version>.joblib`` with a JSON model card next to it, and appends the
version to ``models/registry.json`` (the service loads the entry marked ``current``).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sweepscreen.features import FeatureStore
from sweepscreen.models.encoder import Encoder
from sweepscreen.pipeline import AggregationConfig, SweepScreen
from sweepscreen.train import fit

from .common import ROOT, config, load, parser


def register(bundle: Path, version: str, registry: Path) -> None:
    entries = json.loads(registry.read_text()) if registry.exists() else []
    for e in entries:
        e["current"] = False
    entries.append({"version": version, "file": bundle.name, "card": bundle.with_suffix(".json").name,
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "current": True})
    registry.write_text(json.dumps(entries, indent=1))


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = p.parse_args(argv)
    table, store = load(args)
    aug_root = args.features.parent / "aug"
    aug = FeatureStore(aug_root) if (aug_root / "index.csv").exists() else None
    models = fit(store, table, np.arange(len(table)), config(args), aug)
    enc = Encoder(config=Encoder.tiny_config()) if args.fast else Encoder()
    app = SweepScreen(models, enc, AggregationConfig())
    tmp = args.models_dir / "sweepscreen-tmp.joblib"
    version = app.save(tmp)
    final = args.models_dir / f"sweepscreen-{version}.joblib"
    shutil.move(str(tmp), str(final))
    shutil.move(str(tmp.with_suffix(".json")), str(final.with_suffix(".json")))
    register(final, version, args.models_dir / "registry.json")
    print(f"registered {final.name}")


if __name__ == "__main__":
    main()
