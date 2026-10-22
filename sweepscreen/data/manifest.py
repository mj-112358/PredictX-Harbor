"""Checksummed manifest of every raw data file and pretrained weight.

    python -m sweepscreen.data.manifest build    # writes data/MANIFEST.sha256
    python -m sweepscreen.data.manifest verify   # fails if any file changed or is missing
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
        print(f"{len(entries)} files -> {DEFAULT_OUT}")
        return 0
    problems = verify()
    print("\n".join(problems) or "manifest OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
