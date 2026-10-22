"""Build the record table (if needed), the clean feature store, and the augmented feature store.

    python -m experiments.run_features

``clean``: every image and video frame. ``aug``: one copy of every still image with random handheld
degradations (:class:`sweepscreen.preprocess.augment.RandomAugment`, seed 0), used as extra detector
training data.
"""
from __future__ import annotations

from pathlib import Path

from sweepscreen.data.build import RAW, build, load_table
from sweepscreen.data.loaders import STILL_SITES
from sweepscreen.preprocess.augment import RandomAugment
from sweepscreen.features import extract
from sweepscreen.models.encoder import Encoder

from .common import parser


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--raw", type=Path, default=RAW)
    args = p.parse_args(argv)
    if args.rebuild or not (args.processed / "records.csv").exists():
        build(raw=args.raw, extracted=args.processed.parent / "extracted", processed=args.processed)
    table = load_table(args.processed)
    enc = Encoder(config=Encoder.tiny_config()) if args.fast else Encoder()
    extract(table, enc, "clean", out_root=args.features.parent)
    stills = table[table.site.isin(STILL_SITES)].reset_index(drop=True)
    if len(stills):
        extract(stills, enc, "aug", degrade=RandomAugment(seed=0), out_root=args.features.parent)


if __name__ == "__main__":
    main()
