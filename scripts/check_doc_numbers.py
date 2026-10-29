"""Number check between the English documents and their Korean originals (maintainer tooling).

    python scripts/check_doc_numbers.py            # every pair below
    python scripts/check_doc_numbers.py docs/GUARANTEES.md

The English text of docs/GUARANTEES.md, docs/capabilities.md, docs/api/PUBLIC_API.md,
docs/architecture.md and docs/migrations/0001-initial-extraction.md is canonical; the Korean original is kept beside it as `*.ko.md`. A
translation must carry the same numbers: for each pair, the multiset of numeric tokens is the same.

A numeric token is a maximal run of digits with inner `.` or `,` groups (`1,536`, `0.1.0`,
`4.61`), wherever it stands (`dev53`, `U31`, `p50`, dates and hashes count too), with its sign
when a `−` or `+` stands right before it after a non-word character (`−3.23`, `+0.33`; an ASCII
hyphen is never a sign, so `EVID-1` and `2026-10-21` carry none). Units and words around a number
(`%`, `%p`, `pp`, `초`, `s`) are not part of it. A translation that changes only a unit (`523만` →
`5.23 million`) is listed in CONVERSIONS and reported. Exit status 1 when a pair differs."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    "docs/GUARANTEES.md",
    "docs/capabilities.md",
    "docs/api/PUBLIC_API.md",
    "docs/architecture.md",
    "docs/migrations/0001-initial-extraction.md",
)
NUMBER = re.compile(r"(?<![\d.,])\d+(?:[.,]\d+)*")
# Korean token -> English token where the translation changes only the unit (만 = 10,000)
CONVERSIONS = {
    "docs/capabilities.md": {"523": "5.23"},  # fullwiki "523만 문단" = "5.23 million passages"
}


def korean(path):
    return path.with_name(path.stem + ".ko.md")


def tokens(text):
    out = []
    for m in NUMBER.finditer(text):
        sign = ""
        if m.start() >= 1 and text[m.start() - 1] in "−+":
            before = text[m.start() - 2] if m.start() >= 2 else " "
            if not (before.isalnum() or before == "_"):
                sign = text[m.start() - 1]
        out.append(sign + m.group(0))
    return out


def lines_with(text, token):
    bare = token.lstrip("−+")
    return [
        f"    {i}: {line.strip()[:110]}"
        for i, line in enumerate(text.splitlines(), 1)
        if token in tokens(line) or (bare != token and bare in line)
    ][:3]


def check(relative):
    english = ROOT / relative
    original = korean(english)
    en_text, ko_text = english.read_text(encoding="utf-8"), original.read_text(encoding="utf-8")
    conversions = CONVERSIONS.get(relative, {})
    en = Counter(tokens(en_text))
    ko = Counter(conversions.get(t, t) for t in tokens(ko_text))
    for source, target in conversions.items():
        print(f"{relative}: accepted conversion {source} (Korean) -> {target} (English)")
    missing, extra = ko - en, en - ko
    print(
        f"{relative}: {sum(en.values())} numbers in English, {sum(ko.values())} in Korean, "
        f"{sum(missing.values())} missing, {sum(extra.values())} extra"
    )
    for token, n in sorted(missing.items()):
        print(f"  missing in English ×{n}: {token}")
        print("\n".join(lines_with(ko_text, token)))
    for token, n in sorted(extra.items()):
        print(f"  extra in English ×{n}: {token}")
        print("\n".join(lines_with(en_text, token)))
    return not missing and not extra


def main(argv):
    pairs = argv or PAIRS
    ok = [check(p) for p in pairs]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
