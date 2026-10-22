"""Every experiment script and the report run end to end (fast settings, fixture data, tiny random encoder)."""
import json
import shutil

import pytest

from experiments import (report, run_certify, run_features, run_loso, run_quality, run_robustness, run_runtime,
                         run_video, train_final)


@pytest.fixture
def workspace(tmp_path, busi_tree, busbra_tree, breast_tree, udiat_tree, breastvid_tree):
    raw = tmp_path / "raw"
    raw.mkdir()
    for tree, name in ((busi_tree, "busi"), (busbra_tree, "busbra"), (breast_tree, "breast"),
                       (udiat_tree, "udiat"), (breastvid_tree, "breastvid")):
        shutil.move(str(tree), str(raw / name))
    return tmp_path


def test_all_experiments_and_report(workspace):
    w = workspace
    common = ["--processed", str(w / "proc"), "--features", str(w / "feat" / "clean"), "--results", str(w / "res"), "--fast"]
    run_features.main(common + ["--raw", str(w / "raw")])
    assert (w / "feat" / "aug" / "index.csv").exists()
    run_loso.main(common)
    for site in ("BUSI", "BUS-BRA", "BrEaST", "UDIAT", "BreastVid"):
        assert (w / "res" / "predictions" / f"loso_{site}.jsonl").exists()
    run_video.main(common)
    run_quality.main(common)
    run_certify.main(common)
    run_robustness.main(common + ["--sites", "BUSI"])
    train_final.main(common + ["--models-dir", str(w / "models")])
    reg = json.loads((w / "models" / "registry.json").read_text())
    assert reg[-1]["current"] and (w / "models" / reg[-1]["file"]).exists()
    run_runtime.main(common + ["--models-dir", str(w / "models"), "--threads", "1"])
    report.main(common + ["--out", str(w / "gen")])
    for name in ("tab_datasets.tex", "tab_loso.tex", "tab_video.tex", "tab_quality.tex", "tab_certify.tex",
                 "tab_runtime.tex", "tab_robustness.tex", "macros.tex", "numbers.json", "fig_froc.pdf"):
        assert (w / "gen" / name).exists(), name
