"""Checksummed manifest of every raw data file and pretrained weight.

    python -m sweepscreen.data.manifest build    # writes data/MANIFEST.sha256 and data/MANIFEST.json
    python -m sweepscreen.data.manifest verify   # fails if any file changed or is missing

``MANIFEST.json`` lists every file with its SHA-256, size, source, licence and citation (from
``data/SOURCES.json``, written by ``scripts/download_data.py``), and whether the publisher's checksum
was verified at download.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRS = ("data/raw", "models/pretrained")
DEFAULT_OUT = ROOT / "data" / "MANIFEST.sha256"


def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def collect(dirs: Iterable[str], root: Path = ROOT) -> Dict[str, str]:
    out = {}
    for d in dirs:
        base = root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                out[str(p.relative_to(root))] = sha256(p)
    return out


def write(entries: Dict[str, str], path: Path = DEFAULT_OUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{h}  {f}\n" for f, h in sorted(entries.items())))


def read(path: Path = DEFAULT_OUT) -> Dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        h, f = line.split("  ", 1)
        out[f] = h
    return out


def write_json_manifest(entries: Dict[str, str], sources_path: Path, out: Path) -> dict:
    """Per-file manifest with licence and provenance. Files that no source lists are flagged ``unlisted``."""
    import json

    sources = json.loads(sources_path.read_text()) if sources_path.exists() else {}
    by_file = {}
    for key, src in sources.items():
        for f in src["files"]:
            by_file[f["file"]] = (key, src, f)
    rows = []
    for f, h in sorted(entries.items()):
        key, src, rec = by_file.get(f, (None, {}, {}))
        if rec and rec.get("sha256") != h:
            raise SystemExit(f"{f}: SHA-256 differs from the value recorded at download")
        rows.append({"file": f, "sha256": h, "bytes": (ROOT / f).stat().st_size, "source": key or "unlisted",
                     "licence": src.get("licence"), "citation": src.get("citation"), "url": rec.get("url"),
                     "checksum": rec.get("checksum")})
    licences = {}
    for r in rows:
        licences.setdefault(r["licence"] or "unlisted", set()).add(r["source"])
    doc = {"files": rows, "licences": {k: sorted(v) for k, v in licences.items()}}
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    return doc


def verify(path: Path = DEFAULT_OUT, root: Path = ROOT) -> list:
    """List of problems (empty when every listed file exists with the recorded hash)."""
    problems = []
    for f, h in read(path).items():
        p = root / f
        if not p.exists():
            problems.append(f"missing: {f}")
        elif sha256(p) != h:
            problems.append(f"changed: {f}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "verify"])
    args = parser.parse_args(argv)
    if args.command == "build":
        entries = collect(DEFAULT_DIRS)
        write(entries)
        doc = write_json_manifest(entries, ROOT / "data" / "SOURCES.json", DEFAULT_OUT.with_suffix(".json"))
        print(f"{len(entries)} files -> {DEFAULT_OUT}; licences: {doc['licences']}")
        return 0
    problems = verify()
    print("\n".join(problems) or "manifest OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
