"""Text normalisation for the exact-text comparisons of the cascade. The same-answer graders
(they read a gold reference) are an evaluation instrument and stayed in the research
repository. U22 (parent docs/UNIVERSAL_U22_PROTOCOL.md §1) adds two deterministic,
language-neutral comparisons of two short answers for the second-stage agreement rule — no model
call, no language or domain constant."""

from __future__ import annotations

import re
import unicodedata


def flat(text):
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


# a digit group separator between digits ("1,234", "1.234.567", "1'234", "1 234")
_GROUPED = re.compile(r"(?<=\d)[,.' \u00a0\u202f](?=\d{3}(?!\d))")
_DECIMAL_ZEROS = re.compile(r"(\d+[.,]\d*?)0+(?!\d)")
NUMBER_TOKEN = re.compile(r"\d+(?:[.,]\d+)*")


def _trim_decimal(match):
    kept = match.group(1)
    return kept[:-1] if kept[-1] in ".," else kept


def normalized(text):
    """norm-v1: NFKC, casefold, digit group separators dropped, trailing decimal zeros dropped,
    everything but letters and digits to one space."""
    s = unicodedata.normalize("NFKC", text or "").casefold()
    s = _GROUPED.sub("", s)
    s = _DECIMAL_ZEROS.sub(_trim_decimal, s)
    return re.sub(r"[\W_]+", " ", s).strip()


def normalized_equal(a, b):
    """The same short answer after `normalized` (both non-empty)."""
    x, y = normalized(a), normalized(b)
    return bool(x) and x == y


def _numbers(text):
    return sorted(t.replace(",", "") for t in NUMBER_TOKEN.findall(text or ""))


def _contains_all(bigger, smaller):
    pool = list(bigger)
    for item in smaller:
        if item not in pool:
            return False
        pool.remove(item)
    return True


def _ngrams(text, n=3):
    return {text[i : i + n] for i in range(len(text) - n + 1)} if len(text) >= n else {text}


def same_text(a, b, theta=0.7):
    """U12-b's `same_text` of the research repository, unchanged: after casefolding and dropping
    everything but letters and digits, empty never matches; numbers present on both sides must
    nest (one side's numbers all occur in the other's); then equality or containment (at least
    two characters) is the same answer, else the character-3-gram containment (shared over the
    smaller set) must reach `theta`."""
    x, y = re.sub(r"[\W_]+", "", (a or "").casefold()), re.sub(r"[\W_]+", "", (b or "").casefold())
    if not x or not y:
        return False
    na, nb = _numbers(a), _numbers(b)
    if na and nb and not (_contains_all(na, nb) or _contains_all(nb, na)):
        return False
    if x == y:
        return True
    if min(len(x), len(y)) >= 2 and (x in y or y in x):
        return True
    ga, gb = _ngrams(x), _ngrams(y)
    return len(ga & gb) / min(len(ga), len(gb)) >= theta
