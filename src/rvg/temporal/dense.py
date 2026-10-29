"""Exact dense ranking of a tenant from a reusable float64 matrix (N1) — kept verbatim as the
N1F-1 comparison baseline (arm E0). The product path uses `rvg.storage.local_exact` behind
the vector index port since PORT-0; this module reads a SQLite store directly and is imported only
by the benchmark scripts.

The row-by-row search read every document and every stored vector (JSON) of the tenant per query.
Here the tenant's documents and vectors are read once per *stamp* — the tenant's event sequence
(any upsert or revocation) and its vector set in one embedding space (count and newest rowid, so a
new or re-written vector is seen) — into a snapshot: documents in store order `(id, revision)`,
their family and observation time, and the row-normalised float64 matrix of the documents that
have a vector. A query then filters (revocation, `known_at`, allowed ids) with masks over that
order, scores the eligible rows block by block and keeps the stored-order tie rule of the stable
argsort. The semantics are the row-by-row search's: filters before top-k, documents without a
vector ranked lexically after the dense seeds, the whole family of every seed returned, the same
errors. Nothing here depends on a knowledge base or a language.

A snapshot is never mutated: a changed stamp builds a new one (the memmap arm writes it to a new
file). Reading the store and parsing happen once per stamp; scoring runs off the event loop."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from ..retrieval.data import canonical_hash
from .models import TemporalDocument

BLOCK_ROWS = 8192  # rows scored per block: 8192 x 1536 float64 = 96 MiB at most


EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_MICRO = timedelta(microseconds=1)


def micros(value: datetime) -> int:
    return (value - EPOCH) // _ONE_MICRO


@dataclass(frozen=True)
class DenseSnapshot:
    stamp: tuple
    documents: tuple  # TemporalDocument in store order (id, revision), revoked ones included
    ids: np.ndarray  # object array of document ids, store order
    families: np.ndarray  # object array
    observed: np.ndarray  # int64 microseconds
    inserted: np.ndarray  # int64 event number of the insert, -1 when not recorded (always seen)
    matrix_row: np.ndarray  # int64: row in `matrix`, -1 when the document has no vector
    invalid: np.ndarray  # bool: the stored vector is malformed (raised only if it is eligible)
    matrix: np.ndarray  # float64 (rows, dims), each row divided by its own norm

    @property
    def nbytes(self):
        return int(self.matrix.nbytes)


def input_text(doc):
    return doc.title + "\n" + doc.text


class DenseCache:
    """Snapshots of one store, keyed by (tenant, space); rebuilt when the stamp moves."""

    def __init__(self, store, *, memmap_dir: Path | None = None):
        self.store, self.memmap_dir = store, memmap_dir
        self._entries, self._lock, self._building = {}, threading.Lock(), {}
        self.builds, self.last_matrix_bytes, self.last_build_seconds = 0, None, None

    def stamp(self, tenant, space):
        conn = self.store.conn
        events = self.store.snapshot(tenant)
        vectors = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(rowid),0) FROM passage_vectors WHERE tenant=? AND space=?",
            (tenant, space),
        ).fetchone()
        return (events, int(vectors[0]), int(vectors[1]))

    def read(self, tenant, space):
        """Everything a build needs, read on the caller's thread (the store's connection is not
        shared across threads): raw document rows in store order and the tenant's vectors."""
        conn = self.store.conn
        rows = conn.execute(
            "SELECT body,event_seq FROM temporal_documents WHERE tenant=? ORDER BY id,revision",
            (tenant,),
        ).fetchall()
        vectors = dict(
            conn.execute(
                "SELECT input_hash, vector FROM passage_vectors WHERE tenant=? AND space=?",
                (tenant, space),
            ).fetchall()
        )
        return [(r[0], r[1]) for r in rows], vectors

    def build(self, stamp, bodies, vectors, dims, tenant, space):
        documents = tuple(TemporalDocument.model_validate_json(b) for b, _ in bodies)
        inserted = np.array([-1 if e is None else e for _, e in bodies], dtype=np.int64)
        n = len(documents)
        matrix_row = np.full(n, -1, dtype=np.int64)
        invalid = np.zeros(n, dtype=bool)
        found = []
        for i, doc in enumerate(documents):
            raw = vectors.get(canonical_hash(input_text(doc)))
            if raw is None:
                continue
            value = np.array(json.loads(raw), dtype=float)
            if value.shape != (dims,) or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
                invalid[i] = True
                continue
            matrix_row[i] = len(found)
            found.append(value)
        matrix, final = self._allocate(len(found), dims, tenant, space, stamp)
        for start in range(0, len(found), BLOCK_ROWS):
            block = np.asarray(found[start : start + BLOCK_ROWS])
            matrix[start : start + len(block)] = block / np.linalg.norm(
                block, axis=1, keepdims=True
            )
        if final is not None:
            # Published by rename once complete: a reader never maps a half-written file.
            matrix.flush()
            shape, tmp = matrix.shape, matrix.filename
            del matrix
            os.replace(tmp, final)
            matrix = np.memmap(final, dtype=np.float64, mode="r", shape=shape)
        return DenseSnapshot(
            stamp=stamp,
            documents=documents,
            ids=np.array([d.id for d in documents], dtype=object),
            families=np.array([d.family for d in documents], dtype=object),
            observed=np.array([micros(d.observed_at) for d in documents], dtype=np.int64),
            inserted=inserted,
            matrix_row=matrix_row,
            invalid=invalid,
            matrix=matrix,
        )

    def _allocate(self, rows, dims, tenant, space, stamp):
        if self.memmap_dir is None or rows == 0:
            return np.empty((rows, dims), dtype=np.float64), None
        folder = Path(self.memmap_dir) / canonical_hash([tenant, space])[:24]
        path = folder / f"{canonical_hash(list(stamp))[:32]}.f64"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        return np.memmap(tmp, dtype=np.float64, mode="w+", shape=(rows, dims)), path

    async def get(self, tenant, space, dims, *, run=None):
        stamp = self.stamp(tenant, space)
        key = (tenant, space)
        with self._lock:
            entry = self._entries.get(key)
        if entry is not None and entry.stamp == stamp:
            return entry
        # One build per (tenant, space, stamp): queries that arrive while it runs wait for it
        # instead of each building the same snapshot (N1 R3: concurrent first queries).
        flight = (tenant, space, stamp)
        pending = self._building.get(flight)
        if pending is not None and pending.get_loop() is asyncio.get_running_loop():
            return await asyncio.shield(pending)
        future = self._building[flight] = asyncio.get_running_loop().create_future()
        try:
            snapshot = await self._build(tenant, space, dims, stamp, run)
        except BaseException as exc:
            future.set_exception(exc)
            future.exception()  # retrieved: a failure with no waiter is not a warning
            raise
        finally:
            if self._building.get(flight) is future:
                del self._building[flight]
        future.set_result(snapshot)
        return snapshot

    async def _build(self, tenant, space, dims, stamp, run):
        key = (tenant, space)
        started = time.monotonic()
        bodies, vectors = self.read(tenant, space)
        build = lambda: self.build(stamp, bodies, vectors, dims, tenant, space)
        snapshot = await run(build) if run is not None else build()
        with self._lock:
            self._entries[key] = snapshot
            self.builds += 1
            current = getattr(snapshot.matrix, "filename", None)
            if current:
                # Older snapshots of this tenant and space are superseded: their files go.
                for stale in Path(current).parent.glob("*.f64"):
                    if stale != Path(current):
                        stale.unlink(missing_ok=True)
            self.last_matrix_bytes = snapshot.nbytes
            self.last_build_seconds = round(time.monotonic() - started, 3)
        return snapshot


def eligible(snapshot, *, revoked, known_at, allowed_ids, hidden=(), at=None):
    """Store-order mask of the documents the row-by-row search would have listed: known by
    `known_at`, inserted by knowledge snapshot `at` (if given), not revoked (the caller passes
    the revocations of that snapshot), allowed, and visible to the principal."""
    mask = snapshot.observed <= micros(known_at)
    if at is not None:
        mask &= (snapshot.inserted < 0) | (snapshot.inserted <= at)
    if hidden:
        mask &= ~np.fromiter(
            (i in hidden for i in snapshot.ids), dtype=bool, count=len(snapshot.ids)
        )
    if revoked:
        mask &= ~np.fromiter(
            (i in revoked for i in snapshot.ids), dtype=bool, count=len(snapshot.ids)
        )
    if allowed_ids is not None:
        mask &= np.fromiter(
            (i in allowed_ids for i in snapshot.ids), dtype=bool, count=len(snapshot.ids)
        )
    return mask


def rank(snapshot, mask, query, top_k):
    """(store-order indexes of the top-k eligible documents with a vector, their scores): scores
    of eligible rows only, in store order, then the stable descending order of the row-by-row
    search (ties keep store order), found with a partial selection that keeps every row tied
    with the k-th score."""
    ready = np.flatnonzero(mask & (snapshot.matrix_row >= 0))
    if ready.size == 0:
        return ready, np.empty(0)
    q = np.asarray(query, dtype=float)
    q = q / np.linalg.norm(q)
    rows = snapshot.matrix_row[ready]
    total = snapshot.matrix.shape[0]
    if 2 * rows.size >= total:
        # Most rows are eligible: score every row in contiguous blocks (no copies) and pick the
        # eligible ones; the few extra dot products cost less than gathering scattered rows.
        full = np.empty(total, dtype=np.float64)
        for start in range(0, total, BLOCK_ROWS):
            full[start : start + BLOCK_ROWS] = snapshot.matrix[start : start + BLOCK_ROWS] @ q
        scores = full[rows]
    else:
        scores = np.empty(ready.size, dtype=np.float64)
        for start in range(0, ready.size, BLOCK_ROWS):
            part = rows[start : start + BLOCK_ROWS]
            if part[-1] - part[0] == part.size - 1:  # a contiguous run of rows: no copy
                block = snapshot.matrix[part[0] : part[-1] + 1]
            else:
                block = snapshot.matrix[part]
            scores[start : start + part.size] = block @ q
    if top_k < scores.size:
        kth = np.partition(-scores, top_k - 1)[top_k - 1]
        candidates = np.flatnonzero(-scores <= kth)  # ascending = store order among ties
        order = candidates[np.argsort(-scores[candidates], kind="stable")][:top_k]
    else:
        order = np.argsort(-scores, kind="stable")[:top_k]
    return ready[order], scores[order]
