"""A vector index held by another service (pgvector, Qdrant) behind the vector index port (PORT-2).

The service is a projection of the authoritative stores, never their replacement:

- Which rows may be ranked is decided by the catalog mask (knowledge snapshot, revocations,
  principal, allowed ids) from the state store. The mask is pushed down as a filter (tenant,
  observation time, insert number, and the eligible or excluded point ids), so the service ranks
  inside the eligible set — never a global top-k filtered afterwards.
- A point is trusted only for the vector bytes it was written with: every point carries the
  `vector_seq` of its bytes (`blob_seq`), and a returned point whose version is not the
  authority's current one is treated as not projected.
- Rows changed after the projection frontier (vectors written or documents inserted since the
  last completed projection) are scored locally and exactly from the blob store (overlay). Too
  large an overlay first triggers one projection; still too large is `index_not_caught_up` (an
  ERROR), never "no evidence".

Precision (declared in `capabilities`): `native` ranks by the service's own scores over its
float32 vectors (`EXACT_NATIVE` with an exact scan, `params.exact` / no ANN index).
`canonical` (`EXACT_CANONICAL`) uses the service only to propose candidates: they are rescored
from the canonical float64 vectors, and more are fetched (20 → 100 → 400 → 1,600) until the
native score of the last candidate is below the k-th canonical score by more than twice the
float32 error bound, which proves no unfetched row can enter the top-k; if the ladder ends
first, the eligible set is scanned exactly from the blob store (counted as a fallback)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import ContractError
from .local_exact import canonical_scores, unit
from .ports import EXACT_CANONICAL, EXACT_NATIVE

LADDER = (20, 100, 400, 1600)
UNIT32 = 2.0**-24


def native_error_bound(dims):
    """A bound on |float32 cosine - float64 cosine| for unit vectors: storage and query rounding
    plus a float32 dot product of `dims` terms, with a factor 4 of margin."""
    return 4.0 * (dims + 2) * UNIT32


@dataclass
class RemoteView:
    has_vector: np.ndarray
    invalid: np.ndarray
    generation: dict
    _index: object = field(repr=False)
    _catalog: object = field(repr=False)
    _tenant: str = field(repr=False)
    _space: str = field(repr=False)
    _dims: int = field(repr=False)
    _seq: np.ndarray = field(repr=False)
    _covered: np.ndarray = field(repr=False)
    _points: list = field(repr=False)

    def rank(self, mask, query, top_k):
        return self._index._rank(self, mask, query, top_k)


class RemoteIndex:
    def __init__(
        self,
        state,
        blobs,
        backend,
        *,
        precision="native",
        projector=None,
        overlay_limit=20000,
        ladder=LADDER,
    ):
        if precision not in ("native", "canonical"):
            raise ValueError("unknown_remote_precision")
        self.state, self.blobs, self.backend = state, blobs, backend
        self.precision, self.projector = precision, projector
        self.overlay_limit, self.ladder = overlay_limit, tuple(ladder)
        self.stats = {
            "remote_queries": 0,
            "remote_fetches": 0,
            "overlay_rows": 0,
            "stale_points": 0,
            "fallback_scans": 0,
            "catch_up_projections": 0,
            "candidates_rescored": 0,
        }
        self._points_cache = {}

    def capabilities(self):
        return {
            "index": self.backend.name,
            "search": EXACT_CANONICAL if self.precision == "canonical" else EXACT_NATIVE,
            "stored_dtype": "float32",
            "metric": "cosine",
            "filter": "eligible_ids_and_payload_pushdown_before_top_k",
            "snapshot": "authority_generations_and_point_versions",
            "projection": "outbox_frontier_with_exact_overlay",
            "candidate_ladder": list(self.ladder) if self.precision == "canonical" else None,
            "backend": self.backend.describe(),
        }

    def _point_ids(self, tenant, space, catalog):
        key = (tenant, space, catalog.generation, id(catalog))
        cached = self._points_cache.get(key)
        if cached is None:
            ids = [k.point_id(space) for k in catalog.unit_keys]
            cached = (ids, {p: i for i, p in enumerate(ids)}, catalog)
            self._points_cache = {key: cached}
        return cached

    async def view(self, tenant, space, dims, catalog, *, run=None):
        def build():
            meta = self.blobs.vector_meta(tenant, space)
            frontier = self.state.projection_frontier(tenant, space, self.backend.name)
            n = len(catalog)
            dims_of = np.fromiter(
                (meta.get(h, (-1, 0))[0] for h in catalog.input_hashes), dtype=np.int64, count=n
            )
            seq = np.fromiter(
                (meta.get(h, (-1, 0))[1] for h in catalog.input_hashes), dtype=np.int64, count=n
            )
            has_vector = dims_of == dims
            invalid = (dims_of >= 0) & (dims_of != dims)
            covered = (
                has_vector
                & (seq <= frontier["vector_seq"])
                & ((catalog.inserted < 0) | (catalog.inserted <= frontier["document_seq"]))
            )
            return has_vector, invalid, seq, covered, frontier

        has_vector, invalid, seq, covered, frontier = (
            await run(build) if run is not None else build()
        )
        lag = int((has_vector & ~covered).sum())
        if lag > self.overlay_limit and self.projector is not None:
            self.stats["catch_up_projections"] += 1
            sync = lambda: self.projector.sync(tenant, space, dims)
            await run(sync) if run is not None else sync()
            has_vector, invalid, seq, covered, frontier = (
                await run(build) if run is not None else build()
            )
            lag = int((has_vector & ~covered).sum())
        if lag > self.overlay_limit:
            raise ContractError("index_not_caught_up", detail={"rows": lag})
        return RemoteView(
            has_vector=has_vector,
            invalid=invalid,
            generation={
                "vector": [self.blobs.vector_seq(tenant, space)],
                "frontier": frontier,
                "kind": self.backend.name,
            },
            _index=self,
            _catalog=catalog,
            _tenant=tenant,
            _space=space,
            _dims=dims,
            _seq=seq,
            _covered=covered,
            _points=self._point_ids(tenant, space, catalog),
        )

    # --- ranking -------------------------------------------------------------------------------
    def _canonical(self, view, rows, q):
        """Canonical float64 scores of catalog rows from the blob store's bytes."""
        if len(rows) == 0:
            return np.empty(0)
        catalog = view._catalog
        hashes = [catalog.input_hashes[i] for i in rows]
        found = self.blobs.vectors(view._tenant, view._space, set(hashes))
        m = np.asarray([found[h] for h in hashes], dtype=np.float64)
        m = m / np.linalg.norm(m, axis=1, keepdims=True)
        return canonical_scores(m, q)

    def _rank(self, view, mask, query, top_k):
        self.stats["remote_queries"] += 1
        q = unit(query)
        eligible = mask & view.has_vector
        remote = eligible & view._covered
        overlay = set(np.flatnonzero(eligible & ~view._covered).tolist())
        ids, row_of_point, _ = view._points
        count = int(remote.sum())
        candidates = {}
        if count:
            # The service ranks exactly the eligible, projected rows: listed if they are the
            # smaller side, otherwise every other row of the catalog is excluded by id.
            inside = np.flatnonzero(remote)
            if count <= len(ids) - count:
                include, exclude = [ids[i] for i in inside], None
            else:
                include, exclude = None, [ids[i] for i in np.flatnonzero(~remote)]
            steps = [s for s in self.ladder if s < count] + [count]
            for limit in steps:
                self.stats["remote_fetches"] += 1
                hits = self.backend.search(
                    view._tenant, view._space, q, limit, include=include, exclude=exclude
                )
                native, stale = {}, set()
                for point, score, blob_seq in hits:
                    row = row_of_point.get(point)
                    if row is None or not remote[row]:
                        continue
                    if blob_seq != view._seq[row]:
                        stale.add(row)  # older bytes than the authority's: score it locally
                        continue
                    native[row] = score
                exhausted = limit >= count or len(hits) < limit
                if self.precision == "native":
                    candidates = native
                    ranked = sorted(native.values(), reverse=True)
                    # every row tied with the k-th must have been fetched
                    if exhausted or (len(ranked) >= top_k and ranked[top_k - 1] > hits[-1][1]):
                        break
                else:
                    rows = sorted(native)
                    candidates = dict(
                        zip(rows, self._canonical(view, rows, q).tolist(), strict=True)
                    )
                    self.stats["candidates_rescored"] += len(rows)
                    if exhausted:
                        break
                    ordered = sorted(candidates.values(), reverse=True)
                    bound = 2 * native_error_bound(view._dims)
                    if len(ordered) >= top_k and hits[-1][1] < ordered[top_k - 1] - bound:
                        break
            if stale:
                self.stats["stale_points"] += len(stale)
            overlay |= stale
        self.stats["overlay_rows"] += len(overlay)
        if overlay:
            rows = sorted(overlay)
            for row, score in zip(rows, self._canonical(view, rows, q).tolist(), strict=True):
                candidates[row] = score
        if not candidates:
            return np.empty(0, dtype=np.int64), np.empty(0)
        ranked = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return (
            np.array([r for r, _ in ranked], dtype=np.int64),
            np.array([s for _, s in ranked], dtype=np.float64),
        )
