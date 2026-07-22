"""Identify every punctuation marker used in a text file.

Classifies characters by Unicode general category: anything in a Punctuation
category (``P*``) or Symbol category (``S*``) is treated as a marker. This
catches CJK punctuation (。，「」『』《》 …) as well as ASCII punctuation without
maintaining a hand-written list.

Usage (from scripts/):
    .venv/bin/python find_punctuation.py [path]     # default: mengzi.txt
"""

from __future__ import annotations

import sys
import unicodedata
from collections import Counter
from pathlib import Path

DEFAULT_FILE = Path("mengzi.txt")


def is_marker(ch: str) -> bool:
    """True for Punctuation (P*) or Symbol (S*) categories."""
    return unicodedata.category(ch)[0] in ("P", "S")


def find_punctuation(text: str) -> Counter[str]:
    """Count every punctuation / symbol character in *text*."""
    return Counter(ch for ch in text if is_marker(ch))


def report(counts: Counter[str]) -> str:
    lines = [
        f"{'char':^6} {'code':^8} {'cat':^4} {'count':>7}  name",
        "-" * 60,
    ]
    for ch, n in counts.most_common():
        name = unicodedata.name(ch, "<unnamed>")
        lines.append(f"{ch:^6} U+{ord(ch):04X}  {unicodedata.category(ch):^4} {n:>7}  {name}")
    lines.append("-" * 60)
    lines.append(f"{len(counts)} distinct markers, {sum(counts.values())} total occurrences")
    return "\n".join(lines)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    text = path.read_text(encoding="utf-8")
    print(f"file: {path}  ({len(text)} characters)\n")
    print(report(find_punctuation(text)))


if __name__ == "__main__":
    main()
