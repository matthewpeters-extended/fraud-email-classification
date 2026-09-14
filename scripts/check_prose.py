"""Check that markdown prose contains no hyphen characters.

House style for this project: the written deliverables carry no "-" in prose. Compounds
are rewritten ("out of sample", "mean reverting"), ranges use "to", bullets use "*", and
tables are HTML because the markdown delimiter row requires dashes.

The character is allowed only where it is functionally required:

  * inside fenced code blocks
  * inside inline `code` spans and HTML <code> elements
  * inside URLs
  * inside markdown link targets and anchors

Run over every tracked markdown file, or over named paths.

Usage:
    python scripts/check_prose.py
    python scripts/check_prose.py README.md docs/sources.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FENCE = re.compile(r"^\s*```")
# Order matters. Code spans are removed before HTML tags, so that the contents of a
# <code> element are treated as code rather than exposed as prose by tag stripping.
CODE_SPAN = re.compile(r"`[^`]*`")
HTML_CODE = re.compile(r"<code>.*?</code>", re.DOTALL | re.IGNORECASE)
URL = re.compile(r"https?://\S+|\bwww\.\S+")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
HTML_TAG = re.compile(r"<[^>]*>")

SKIP_DIRS = {".venv", "data", ".pytest_cache", ".git", "__pycache__"}

ALL_DASHES = "-‐‑‒–—―−"
DASH_NAMES = {
    "-": "hyphen minus",
    "‐": "hyphen",
    "‑": "non breaking hyphen",
    "‒": "figure dash",
    "–": "en dash",
    "—": "em dash",
    "―": "horizontal bar",
    "−": "minus sign",
}


def strip_allowed(line: str) -> str:
    """Remove every region where a dash is permitted."""
    for pattern in (HTML_CODE, CODE_SPAN, URL, LINK_TARGET, HTML_TAG):
        line = pattern.sub(" ", line)
    return line


def check(path: Path) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    in_fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        prose = strip_allowed(line)
        for char in ALL_DASHES:
            if char in prose:
                findings.append((number, DASH_NAMES[char], line.strip()))
                break
    return findings


def main(argv: list[str]) -> int:
    if argv:
        targets = [Path(a) if Path(a).is_absolute() else ROOT / a for a in argv]
    else:
        targets = sorted(
            p
            for p in ROOT.rglob("*.md")
            if not SKIP_DIRS & set(p.parts)
        )

    total = 0
    for path in targets:
        if not path.exists():
            print(f"missing: {path}")
            return 1
        findings = check(path)
        rel = path.relative_to(ROOT)
        if findings:
            total += len(findings)
            print(f"{rel}: {len(findings)} dash character(s) in prose")
            for number, name, text in findings:
                print(f"  L{number} ({name}): {text[:100]}")
        else:
            print(f"{rel}: clean")

    print()
    if total:
        print(f"FAIL: {total} dash character(s) in prose across {len(targets)} file(s)")
        return 1
    print(f"PASS: {len(targets)} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
