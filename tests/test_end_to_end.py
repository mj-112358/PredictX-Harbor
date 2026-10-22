"""Plumbing test of the whole chain on synthetic fixture data with a tiny randomly initialised encoder."""
import shutil

import numpy as np
import pytest

from sweepscreen.data import build as build_mod
from sweepscreen.data import splits
from sweepscreen.features import FeatureStore, extract
from sweepscreen.models.encoder import Encoder
from sweepscreen.pipeline import AggregationConfig, SweepScreen
from sweepscreen.train import TrainConfig, fit, predict
from tests.conftest import synthetic_frame


@pytest.fixture
def raw_root(tmp_path, busi_tree, breastvid_tree, busbra_tree):
    raw = tmp_path / "raw"
    raw.mkdir()
    for tree, name in ((busi_tree, "busi"), (breastvid_tree, "breastvid"), (busbra_tree, "busbra")):
        shutil.move(str(tree), str(raw / name))
    return raw


def test_full_chain(tmp_path, raw_root):
    table = build_mod.build(raw=raw_root, extracted=tmp_path / "ext", processed=tmp_path / "proc")
    assert set(table.site) == {"BUSI", "BreastVid", "BUS-BRA"}
    assert (tmp_path / "proc" / "records.csv").exists()

    enc = Encoder(config=Encoder.tiny_config(), threads=1, batch_size=8)
    store = FeatureStore(extract(table, enc, out_root=tmp_path / "feat", chunk=8, qfail_every=2))
    assert store.array("tokens").shape == (len(table), 16, 16, 32)
    assert len(store.index) == len(table)

    for site, tr, te in splits.leave_one_site_out(table):
        splits.check_disjoint(table, tr, te)
    tr = np.flatnonzero(table.site.values != "BUSI")
    te = np.flatnonzero(table.site.values == "BUSI")
    models = fit(store, table, tr, TrainConfig(detector_epochs=3, video_stride=1, malignancy_video_stride=1))
    pred = predict(models, store, table, te)
    assert len(pred) == len(te) and pred.frame_prob.between(0, 1).all()
    assert {"candidates", "gt_boxes", "mal_det", "quality"} <= set(pred.columns)

    app = SweepScreen(models, enc, AggregationConfig(window=2, min_frames=3, min_fraction=0.5))
    version = app.save(tmp_path / "models" / "bundle.joblib")
    loaded = SweepScreen.load(tmp_path / "models" / "bundle.joblib", enc)
    assert loaded.version == version and len(version) == 12
    rng = np.random.default_rng(0)
    frames = [synthetic_frame(rng, center=(120, 100))[0] for _ in range(6)]
    out = loaded.analyze_sweep({"left": frames, "right": frames[:2]})
    assert out["decision"] in {"NORMAL", "RECALL", "UNREADABLE"}
    assert len(out["frames"]["left"]) == 6
    if out["decision"] != "RECALL":
        assert out["decision"] == "UNREADABLE"     # right breast has only 2 frames (< min_frames)
