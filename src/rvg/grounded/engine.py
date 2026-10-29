"""Exact quote alignment used by extractive repair. The legacy grounded answer engine stayed in the
research repository; the Universal engine is the one answer path."""

from __future__ import annotations


def align_quote(source: str, proposed: str) -> str | None:
    if not proposed.strip():
        return None
    if source.count(proposed) == 1:
        return proposed
    # Resolve whitespace-only copying differences to an exact original span; do not edit the source.
    needle = "".join(c for c in proposed if not c.isspace())
    positions = [i for i, c in enumerate(source) if not c.isspace()]
    compact = "".join(source[i] for i in positions)
    if not needle or compact.count(needle) != 1:
        return None
    offset = compact.index(needle)
    return source[positions[offset] : positions[offset + len(needle) - 1] + 1]
