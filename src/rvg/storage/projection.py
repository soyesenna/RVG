"""The projector (PORT-2): brings a remote vector index up to the authoritative stores and then
moves the projection frontier — never before.

One projector per (tenant, space, index) holds a lease with a fencing token in the state store.
A sync reads the frontier (the `vector_seq` and `document_seq` the index is known to contain),
the vectors changed since then (the blob store's change feed) and the revisions inserted since
then (the catalog), writes one point per revision whose current vector is valid — the point id
is `UnitKey.point_id(space)`, the payload carries the vector's `blob_seq` — deletes the points of
vectors that turned invalid, checks between batches that its token is still current, and only
then advances the frontier under that token. A crash after the writes and before the advance
leaves the index ahead of the frontier, which is safe: rows past the frontier are scored from
the authority (overlay). Revocations and ACL are never projected; they are filters from the
authority at query and release time."""

from __future__ import annotations

import os
import socket
import time

import numpy as np

from .catalog import micros


class Projector:
    def __init__(
        self, state, blobs, backend, catalogs, *, holder=None, ttl=120.0, batch=256, clock=None
    ):
        self.state, self.blobs, self.backend, self.catalogs = state, blobs, backend, catalogs
        self.holder = holder or f"{socket.gethostname()}:{os.getpid()}:{id(self)}"
        self.ttl, self.batch = ttl, batch
        self.clock = clock or time.time
        self.stats = {"syncs": 0, "upserted": 0, "deleted": 0, "fenced": 0, "busy": 0}

    def sync(self, tenant, space, dims, *, fault=None):
        token = self.state.projection_lease(
            tenant, space, self.backend.name, self.holder, now=self.clock(), ttl=self.ttl
        )
        if token is None:
            self.stats["busy"] += 1
            return {"status": "busy"}
        started = time.monotonic()
        self.backend.ensure(space, dims)
        frontier = self.state.projection_frontier(tenant, space, self.backend.name)
        catalog = self.catalogs.build(tenant)  # parsed revisions are reused across builds
        target_documents = catalog.generation
        initial = frontier["vector_seq"] == 0 and frontier["document_seq"] == 0
        with self.blobs.scan_changes(tenant, space, frontier["vector_seq"]) as scan:
            target_vectors = scan.vector_seq
            changed = {r[0] for r in scan.rows}
        meta = self.blobs.vector_meta(tenant, space)
        if initial:
            selected = set(range(len(catalog)))
        else:
            selected = set(np.flatnonzero(catalog.inserted > frontier["document_seq"]).tolist())
            for h in changed:
                selected.update(catalog.rows_of_hash.get(h, np.empty(0, dtype=int)).tolist())
        upserts, deletes, kept = [], [], []
        for row in sorted(selected):
            h = catalog.input_hashes[row]
            point = catalog.unit_keys[row].point_id(space)
            m = meta.get(h)
            if m is None:
                continue  # no vector yet: the pending path covers it
            if m[0] != dims:
                deletes.append(point)
                continue
            kept.append((row, point, h, m[1]))
        docs = catalog.docs([k[0] for k in kept])  # N1F-6: one batched read
        for (row, point, h, seq), doc in zip(kept, docs, strict=True):
            upserts.append((point, h, doc, seq, int(catalog.inserted[row])))
        vectors = self.blobs.vectors(tenant, space, {u[1] for u in upserts}) if upserts else {}
        written = 0
        for start in range(0, len(upserts), self.batch):
            if self.state.projection_frontier(tenant, space, self.backend.name)["token"] != token:
                self.stats["fenced"] += 1
                return {"status": "fenced", "written": written}
            chunk = upserts[start : start + self.batch]
            m = np.asarray([vectors[h] for _, h, _, _, _ in chunk], dtype=np.float64)
            m = m / np.linalg.norm(m, axis=1, keepdims=True)
            self.backend.upsert(
                space,
                [
                    (
                        point,
                        row,
                        {
                            "tenant": tenant,
                            "document_id": doc.id,
                            "revision": doc.revision,
                            "input_hash": h,
                            "observed_us": micros(doc.observed_at),
                            "inserted": max(0, seq_inserted),
                            "blob_seq": int(blob_seq),
                        },
                    )
                    for (point, h, doc, blob_seq, seq_inserted), row in zip(chunk, m, strict=True)
                ],
            )
            written += len(chunk)
            if fault is not None:
                fault("after_batch")
        self.backend.delete(space, deletes)
        if not self.state.advance_projection(
            tenant, space, self.backend.name, token, target_vectors, target_documents
        ):
            self.stats["fenced"] += 1
            return {"status": "fenced", "written": written}
        self.stats["syncs"] += 1
        self.stats["upserted"] += written
        self.stats["deleted"] += len(deletes)
        return {
            "status": "ok",
            "written": written,
            "deleted": len(deletes),
            "frontier": {"vector_seq": target_vectors, "document_seq": target_documents},
            "seconds": round(time.monotonic() - started, 4),
        }
