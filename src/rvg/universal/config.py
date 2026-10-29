"""The one policy (`rvg/resources/policies/universal.json`), read once, hashed once.

Every value that decides what the system retrieves, writes, checks and releases lives in that
one file. Product modules take their defaults from it; nothing reads a per-knowledge-base,
per-language or per-request policy. The file is a package resource: it is read from the
installed package, never relative to a working directory or a source checkout."""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from importlib.resources import files

PATH = files("rvg") / "resources" / "policies" / "universal.json"


@lru_cache(maxsize=1)
def universal() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def universal_hash() -> str:
    return hashlib.sha256(PATH.read_bytes()).hexdigest()


def dimension_warning(embedding):
    """D-DIM-DECL (documented, not fixed; parent docs/PROD_1_PROTOCOL.md §6): the policy file's
    `models.embedding_dimensions` is a declaration no code reads — the embedding space is the
    configured provider's (every measurement so far: 1,536 dimensions). Returns a warning when the
    declaration and the space differ, else None. Nothing is changed by it."""
    if embedding is None:
        return None
    declared = (universal().get("models") or {}).get("embedding_dimensions")
    try:
        actual = embedding.describe().dimensions
    except AttributeError:  # a provider without a described space: nothing to compare
        return None
    if declared is None or actual is None or int(declared) == int(actual):
        return None
    return (
        f"policy models.embedding_dimensions={declared} is not read; the embedding space in use "
        f"has {actual} dimensions (D-DIM-DECL, behaviour unchanged)"
    )


_NOTED = set()  # the D-DIM-DECL notes this process has logged


def warn_dimensions(embedding):
    """Log `dimension_warning` at INFO, once per process for each distinct note (0.1.0, parent
    PUB-2: it was a WARNING at every session's open, written into the caller's output)."""
    message = dimension_warning(embedding)
    if message and message not in _NOTED:
        _NOTED.add(message)
        logging.getLogger("rvg.policy").info(message)
    return message
