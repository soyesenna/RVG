"""Canonical JSON hashing shared by the stores and indexes. The dataset tooling that sat beside
it in the research repository (SciFact download, normalisation, manifests) stayed there."""

from __future__ import annotations

import hashlib
import json


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
