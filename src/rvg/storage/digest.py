"""The knowledge digest an evaluation run is pinned to, defined on logical content so every
backend gives the same value for the same knowledge: document revisions (their stored JSON body),
revocations, and the vectors of one embedding space as the JSON list of their float64 values —
the text the SQLite store has always kept, rebuilt from canonical bytes elsewhere."""

from __future__ import annotations

import hashlib
import json

import numpy as np


def vector_json(data: bytes) -> str:
    return json.dumps(np.frombuffer(data, dtype="<f8").tolist())


def knowledge_digest(documents, revocations, vectors=None):
    """`documents`: (id, revision, body) in (id, revision) order; `revocations`: ids in order;
    `vectors`: (input_hash, float64 bytes) in hash order, or None for no space."""
    h = hashlib.sha256()
    for identifier, revision, body in documents:
        h.update(f"d|{identifier}|{revision}|".encode())
        h.update(hashlib.sha256(body.encode()).digest())
    for identifier in revocations:
        h.update(f"r|{identifier}".encode())
    if vectors is not None:
        for key, data in vectors:
            h.update(f"v|{key}|".encode())
            h.update(hashlib.sha256(vector_json(data).encode()).digest())
    return h.hexdigest()
