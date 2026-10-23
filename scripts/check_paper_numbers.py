"""Fail if the paper uses a number macro that ``experiments.report`` did not generate.

    python scripts/check_paper_numbers.py [paper_dir]

Every result quoted in the paper is a ``\\N...`` macro defined in ``paper/generated/macros.tex``.
"""
import re
import sys
from pathlib import Path

paper = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "paper")
defined = set(re.findall(r"\\newcommand\{\\(N[A-Za-z]+)\}", (paper / "generated" / "macros.tex").read_text()))
used = {}
for tex in [paper / "main.tex", *sorted((paper / "sections").glob("*.tex"))]:
    for name in re.findall(r"\\(N[A-Za-z]+)", tex.read_text()):
        used.setdefault(name, tex.name)
missing = sorted(n for n in used if n not in defined)
for n in missing:
    print(f"undefined: \\{n} (used in {used[n]})")
print(f"{len(used)} macros used, {len(defined)} defined, {len(missing)} missing")
sys.exit(1 if missing else 0)
