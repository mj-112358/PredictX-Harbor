"""Download every public dataset and the encoder weights, verify checksums, and record provenance.

    python scripts/download_data.py [--only busi busbra ...]

Each source goes into its own folder under ``data/raw/`` (weights under ``models/pretrained/``); a
folder must be empty or contain only this source's files. For every file the script records the URL,
licence, citation, size and SHA-256 in ``data/SOURCES.json``, and checks the publisher's checksum
where one is published (Zenodo MD5, Mendeley SHA-256, Hugging Face LFS SHA-256). A mismatch aborts.
UDIAT Dataset B is distributed by signed request and is not downloaded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES_JSON = ROOT / "data" / "SOURCES.json"
HF_REV = "ed25f3a31f01632728cabb09d1542f84ab7b0056"   # facebook/dinov2-small, pinned

SOURCES = {
    "busi": {
        "dir": "data/raw/busi", "licence": "CC BY 4.0",
        "citation": "Al-Dhabyani et al., Data in Brief 28:104863 (2020); curated release: Aumente-Maestro et al., Zenodo 19047974 (2025)",
        "files": [{"name": "Curated BUSI.zip", "url": "https://zenodo.org/records/19047974/files/Curated%20BUSI.zip?download=1",
                   "zenodo": ("19047974", "Curated BUSI.zip")}]},
    "busi_audit": {
        "dir": "data/raw/busi_audit", "licence": "CC BY 4.0",
        "citation": "Supplementary files of the letter to the editor on BUSI (Mendeley Data k8t3gnx9h6 v1)",
        "files": [{"name": "dataset_comment_list.csv",
                   "url": "https://data.mendeley.com/public-files/datasets/k8t3gnx9h6/files/41d7e047-07b3-4a4a-b89f-b9b70f6e9285/file_downloaded",
                   "sha256": "e3c2abc0ce027401dc717bf8a7ec76442e30c3cda2df32a0bc8aa197ff1c74a4"}]},
    "busbra": {
        "dir": "data/raw/busbra", "licence": "CC BY 4.0",
        "citation": "Gómez-Flores et al., Med Phys 51:3110-3123 (2024); Zenodo 8231412",
        "files": [{"name": "BUSBRA.zip", "url": "https://zenodo.org/records/8231412/files/BUSBRA.zip?download=1",
                   "zenodo": ("8231412", "BUSBRA.zip")}]},
    "breast": {
        "dir": "data/raw/breast", "licence": "CC BY 4.0",
        "citation": "Pawłowska et al., Sci Data 11:148 (2024); TCIA Breast-Lesions-USG, doi:10.7937/9WKK-Q141",
        "files": [{"name": "BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip",
                   "url": "https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip"},
                  {"name": "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx",
                   "url": "https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"}]},
    "breastvid": {
        "dir": "data/raw/breastvid", "licence": "Non-commercial research and education use only (CVA-Net README)",
        "citation": "Lin et al., MICCAI 2022, pp. 614-623; github.com/jhl-Det/CVA-Net",
        "files": [{"name": "miccai_buv.7z", "gdrive": "1LVXK34OJhC2LkqqyMmVFnXsXQavvZdeF"}]},
    "busuclm": {
        "dir": "data/raw/busuclm", "licence": "CC BY 4.0",
        "citation": "Vallez et al., Sci Data (2025); Mendeley Data 7fvgj4jsp7 v3",
        "files": [{"name": "7fvgj4jsp7-3.zip", "url": "https://data.mendeley.com/public-api/zip/7fvgj4jsp7/download/3"}]},
    "dinov2-small": {
        "dir": "models/pretrained/dinov2-small", "licence": "Apache-2.0",
        "citation": f"Oquab et al., DINOv2 (TMLR 2024); huggingface.co/facebook/dinov2-small@{HF_REV}",
        "files": [{"name": n, "url": f"https://huggingface.co/facebook/dinov2-small/resolve/{HF_REV}/{n}", "hf": n}
                  for n in ("config.json", "model.safetensors", "preprocessor_config.json")]},
}


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(url: str):
    out = subprocess.run(["curl", "-sSfL", "-m", "60", url], capture_output=True, check=True).stdout
    return json.loads(out)


def published_checksum(f: dict):
    """(algorithm, hex) published by the host, or None."""
    if "sha256" in f:
        return "sha256", f["sha256"]
    if "zenodo" in f:
        rec, key = f["zenodo"]
        for item in _json(f"https://zenodo.org/api/records/{rec}")["files"]:
            if item["key"] == key:
                algo, value = item["checksum"].split(":")
                return algo, value
    if "hf" in f:
        info = _json(f"https://huggingface.co/api/models/facebook/dinov2-small/revision/{HF_REV}?blobs=true")
        for s in info["siblings"]:
            if s["rfilename"] == f["hf"] and s.get("lfs"):
                return "sha256", s["lfs"]["sha256"]
    return None


def fetch(f: dict, dest: Path) -> None:
    if "gdrive" in f:
        import gdown
        gdown.download(id=f["gdrive"], output=str(dest), quiet=False)
    else:
        subprocess.run(["curl", "-fL", "--retry", "4", "--retry-delay", "2", "-m", "7200", "-o", str(dest), f["url"]],
                       check=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*")
    args = p.parse_args(argv)
    record = json.loads(SOURCES_JSON.read_text()) if SOURCES_JSON.exists() else {}
    for key, src in SOURCES.items():
        if args.only and key not in args.only:
            continue
        folder = ROOT / src["dir"]
        folder.mkdir(parents=True, exist_ok=True)
        expected = {f["name"] for f in src["files"]}
        stray = [p.name for p in folder.iterdir() if p.name not in expected]
        if stray:
            raise SystemExit(f"{folder} contains unexpected files {stray}; each source needs its own folder")
        entries = []
        for f in src["files"]:
            dest = folder / f["name"]
            if not dest.exists():
                print(f"[get] {key}: {f['name']}", flush=True)
                fetch(f, dest)
            pub = published_checksum(f)
            status = "no published checksum (SHA-256 recorded on first download)"
            if pub:
                got = digest(dest, pub[0])
                if got != pub[1]:
                    raise SystemExit(f"{dest}: {pub[0]} mismatch (got {got}, published {pub[1]})")
                status = f"verified against published {pub[0]}"
            entries.append({"file": str(dest.relative_to(ROOT)), "url": f.get("url") or f"gdrive:{f['gdrive']}",
                            "bytes": dest.stat().st_size, "sha256": digest(dest, "sha256"), "checksum": status})
            print(f"[ok] {key}: {f['name']} ({dest.stat().st_size / 1e6:.1f} MB) {status}", flush=True)
        record[key] = {"licence": src["licence"], "citation": src["citation"], "files": entries,
                       "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        SOURCES_JSON.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
