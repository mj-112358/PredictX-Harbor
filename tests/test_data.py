import numpy as np
import pandas as pd
import pytest

from sweepscreen.data import dedup, manifest, splits
from sweepscreen.data.loaders import breast, breastvid, busbra, busi, busuclm, udiat
from sweepscreen.data.loaders.common import LayoutError
from sweepscreen.data.schema import Record, boxes_from_mask, from_frame, to_frame


def test_boxes_from_mask_one_box_per_component():
    m = np.zeros((50, 50), np.uint8)
    m[5:10, 5:15] = 1
    m[30:40, 30:35] = 1
    assert sorted(boxes_from_mask(m)) == [(5, 5, 15, 10), (30, 30, 35, 40)]


def test_record_validation():
    with pytest.raises(ValueError):
        Record(uid="x:1", site="x", image_path="a.png", label="cancer", patient_id="x:1")
    with pytest.raises(ValueError):
        Record(uid="x:1", site="x", image_path="a.png", label="benign", patient_id="x:1", boxes=[(5, 5, 5, 9)])


def test_frame_roundtrip():
    r = Record(uid="s:1", site="s", image_path="a.png", label="benign", patient_id="s:p", boxes=[(1, 2, 3, 4)],
               meta={"birads": "4a"}, age=51.0)
    back = from_frame(to_frame([r]))[0]
    assert back == r


def test_busi_loader(busi_tree, tmp_path):
    recs = busi.load(str(busi_tree), str(tmp_path / "x"))
    assert len(recs) == 7
    assert sum(r.label == "normal" for r in recs) == 2
    two = [r for r in recs if r.uid == "BUSI:benign_id_1"][0]
    assert len(two.mask_paths) == 2 and len(two.boxes) == 2
    assert all(not r.boxes for r in recs if r.label == "normal")


def test_busi_original_layout(busi_original_tree, tmp_path):
    recs = busi.load(str(busi_original_tree), str(tmp_path / "x"))
    assert sorted(r.label for r in recs) == ["benign", "benign", "malignant", "normal"]


def test_busuclm_loader(busuclm_tree, tmp_path):
    recs = busuclm.load(str(busuclm_tree), str(tmp_path / "x"))
    assert len(recs) == 3                                   # the Doppler image is excluded
    assert {r.patient_id for r in recs} == {"BUS-UCLM:ALWI", "BUS-UCLM:BEMA"}
    assert [len(r.boxes) for r in recs] == [1, 0, 1]
    assert [r.label for r in recs] == ["benign", "normal", "malignant"]


def test_breastvid_frames_without_annotation_entry_are_excluded(breastvid_tree, tmp_path):
    recs = breastvid.load(str(breastvid_tree), str(tmp_path / "x"))
    assert not any(r.image_path.endswith("00099.png") for r in recs)


def test_busbra_loader_patient_ids(busbra_tree, tmp_path):
    recs = busbra.load(str(busbra_tree), str(tmp_path / "x"))
    assert len(recs) == 6
    assert len({r.patient_id for r in recs}) == 3        # two images per case
    assert {r.label for r in recs} == {"benign", "malignant"}
    assert all(len(r.boxes) == 1 for r in recs)


def test_breast_loader_metadata(breast_tree, tmp_path):
    recs = breast.load(str(breast_tree), str(tmp_path / "x"))
    assert [r.label for r in recs] == ["benign", "malignant", "benign", "normal"]
    assert recs[0].age == 41.0 and recs[0].meta["Tissue_composition"] == "heterogeneous"
    assert recs[3].boxes == []


def test_udiat_loader(udiat_tree, tmp_path):
    recs = udiat.load(str(udiat_tree), str(tmp_path / "x"))
    assert len(recs) == 4 and sum(r.malignant for r in recs) == 1


def test_breastvid_loader_applies_corrections(breastvid_tree, tmp_path):
    recs = breastvid.load(str(breastvid_tree), str(tmp_path / "x"))
    videos = {r.video_id for r in recs}
    assert "BreastVid:benign/x66ef02e7f1b9a0ef" not in videos       # dropped duplicate
    a = sorted([r for r in recs if r.video_id == "BreastVid:benign/vidA"], key=lambda r: r.frame_index)
    assert [bool(r.boxes) for r in a] == [False] + [True] * 6 + [False]


def test_missing_layout_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(LayoutError):
        busbra.load(str(tmp_path / "empty"), str(tmp_path / "x"))


def test_hamming_and_groups():
    a = np.array([0, 1, 0b111], dtype=np.uint64)
    assert dedup.hamming_matrix(a, a).tolist() == [[0, 1, 3], [1, 0, 2], [3, 2, 0]]
    recs = [Record(uid=f"s:{i}", site="s" if i < 3 else "t", image_path="x", label=lab, patient_id=f"s:p{i}")
            for i, lab in enumerate(["benign", "malignant", "benign", "benign"])]
    hashes = {"s:0": 0, "s:1": 1, "s:2": 0xFFFF_FFFF, "s:3": 0}
    g = dedup.assign_groups(recs, hashes, max_distance=2)
    assert g.group_id[0] == g.group_id[1] == g.group_id[3] != g.group_id[2]
    assert g.discordant[0] and g.discordant[1] and not g.discordant[2]
    assert g.cross_site[3]


def _table(n_sites=3, per=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sites):
        for i in range(per):
            rows.append({"site": f"S{s}", "patient_id": f"S{s}:{i // 2}", "group_id": f"g{s}_{i // 2}",
                         "label": rng.choice(["benign", "malignant"])})
    t = pd.DataFrame(rows)
    t.loc[[0, 1], "group_id"] = t.loc[per, "group_id"]   # patient S0:0 is a cross-site duplicate of S1:0
    return t


def test_group_kfold_disjoint():
    t = _table()
    for _, tr, te in splits.group_kfold(t, 5):
        splits.check_disjoint(t, tr, te)


def test_loso_excludes_cross_site_groups():
    t = _table()
    for site, tr, te in splits.leave_one_site_out(t):
        assert set(t.site.values[te]) == {site}
        assert not set(t.group_id.values[tr]) & set(t.group_id.values[te])


def test_manifest_roundtrip(tmp_path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    f = tmp_path / "data" / "raw" / "a.bin"
    f.write_bytes(b"abc")
    entries = manifest.collect(["data/raw"], root=tmp_path)
    out = tmp_path / "M.sha256"
    manifest.write(entries, out)
    assert manifest.verify(out, root=tmp_path) == []
    f.write_bytes(b"abd")
    assert manifest.verify(out, root=tmp_path) == ["changed: data/raw/a.bin"]
