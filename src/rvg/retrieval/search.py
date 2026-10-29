"""Lexical retrieval of the product path: the policy tokenizer and full-document BM25. The
corpus retrievers and rank fusion of the SciFact retrieval research stayed in the research
repository."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from ..errors import ContractError
from ..universal import universal

LEXICAL = universal()["retrieval"].get("lexical", "word")


def tokens(text: str, mode: str | None = None) -> list[str]:
    """`word`: \\w+ runs (assumes word boundaries). `trigram`: every window of three characters
    of the case-folded text with whitespace collapsed — no word boundary or inflection assumed,
    the same shape as FTS5's trigram tokenizer, so it is the language-neutral channel. The mode
    is a global policy (`rvg/resources/policies/universal.json` retrieval.lexical), never a per-KB choice."""
    mode = mode or LEXICAL
    folded = unicodedata.normalize("NFKC", text).casefold()
    if mode == "word":
        return re.findall(r"\w+", folded)
    if mode == "trigram":
        flat = " ".join(folded.split())
        return (
            [flat[i : i + 3] for i in range(len(flat) - 2)]
            if len(flat) >= 3
            else [flat] * bool(flat)
        )
    raise ContractError("unknown_lexical_tokenizer")


class BM25:
    """Full-document BM25; TF preserved, log1p IDF, explicit tokenizer/version."""

    def __init__(
        self,
        documents: list[dict],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: str | None = None,
    ):
        self.ids = [doc["id"] for doc in documents]
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or LEXICAL
        postings = defaultdict(list)
        lengths = []
        for index, doc in enumerate(documents):
            words = tokens(doc["title"] + " " + doc["text"], self.tokenizer)
            lengths.append(len(words))
            for word, count in Counter(words).items():
                postings[word].append((index, count))
        self.lengths = np.asarray(lengths, dtype=float)
        self.average = max(float(np.mean(self.lengths)), 1)
        self.postings = {
            word: (np.asarray([p[0] for p in rows]), np.asarray([p[1] for p in rows], dtype=float))
            for word, rows in postings.items()
        }

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        scores = np.zeros(len(self.ids))
        for word, qtf in Counter(tokens(query, self.tokenizer)).items():
            if word not in self.postings:
                continue
            indices, tf = self.postings[word]
            idf = math.log1p((len(self.ids) - len(indices) + 0.5) / (len(indices) + 0.5))
            denominator = tf + self.k1 * (
                1 - self.b + self.b * self.lengths[indices] / self.average
            )
            scores[indices] += qtf * idf * tf * (self.k1 + 1) / denominator
        candidates = np.flatnonzero(scores > 0)
        ordered = sorted(candidates, key=lambda i: (-float(scores[i]), self.ids[i]))[:limit]
        return [(self.ids[i], float(scores[i])) for i in ordered]
