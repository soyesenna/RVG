"""Exact dense search over an external, read-only float16 vector set (N1F-5, opt-in).

A deployment that already holds a corpus's vectors as float16 shards (one `.npy` file per shard)
ranks them in place: nothing is copied, converted on disk or written. A manifest names the shards
and their sha256, the embedding space they were computed in and a file with, per row, the input
hash of the RVG embedding input (`title-newline-text/1`) the row is the embedding of — all zeros
for a row whose input was something else (an empty title, a truncated text), which then has no
vector here. A catalog row reaches an external row through its input hash, as it reaches a row
of the canonical matrix.

Scores: the scan raises the float16 row exactly to float64 (or float32, `score_dtype`) and takes
one matrix-vector product per block of a fixed partition — each shard cut at multiples of
`BLOCK_ROWS` — divided by the row's float64 norm; the query is normalised in float64. The rows that
can reach the top-k are then rescored in float64 in a fixed summation order (`canonical_scores`,
`certified_top`), so a ranking and its scores never depend on the mask, on the other rows, on the
number of threads, on `score_dtype` or on the platform's BLAS. The ranking rule is `rank_rows`'s:
eligible rows only, ties in catalog order, every row tied with the k-th score kept. The precision
class is `EXACT_NATIVE`: exhaustive over the stored float16 representation, not over the
provider's float64 output. Vectors the state store holds for inputs the set does not have
(documents inserted later) are ranked in the same ranking from the resident canonical matrix of
the local exact index.

Query projection (opt-in, E9): a set embedded with fewer dimensions of the same model and inputs
(a Matryoshka model asked for `dimensions=d`) can serve queries of the deployment's space of N > d
dimensions — the space's fingerprint must differ from the set's in `dimensions` only — by keeping
the query's first d coordinates and normalising them in float64. Vectors the state store holds in
that space are cut and normalised the same way. The blob reads of that space give no vector to the
release check (it cannot compare an N-dimensional query with a d-dimensional row), so a change to
the knowledge base during a question leaves its scope unknown.

The optional candidate stage (N1F-4's TurboQuant codes, built from the external rows) rescores
its candidates from the same external rows in float64 and declares `APPROXIMATE`.

Writing a set (docs/external-vectors.md): `write_external_set` writes the shards, the row hash
file and the manifest from `(title, text, vector)` or `(input hash, vector)` rows;
`export_external_set` writes the vectors a state store holds for one embedding space. Both check
the result with `ExternalVectorSet`, the reader a deployment uses.

Nothing here depends on a knowledge base or a language."""

from __future__ import annotations

import functools
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..errors import ContractError
from ..retrieval.data import canonical_hash
from .identity import EMBEDDING_INPUT, embedding_input, input_hash
from .local_exact import (
    BLOCK_ROWS,
    LocalExactIndex,
    _parse,
    canonical_scores,
    certified_top,
    scan_bound,
    unit,
)
from .ports import APPROXIMATE, EXACT_NATIVE

FORMAT = "rvg-external-vectors/1"
SCORE_DTYPES = ("float64", "float32")
PROJECTIONS = ("truncate_l2",)
HASH_BYTES = 32


def space_fingerprint(space):
    """The fingerprint of an embedding space description (`EmbeddingSpace.fingerprint`)."""
    return hashlib.sha256(
        json.dumps(space, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _keys(raw):
    """Four native uint64 columns whose lexicographic order is the byte order of 32-byte hashes."""
    words = np.ascontiguousarray(raw, dtype=np.uint8).view(">u8").reshape(-1, 4)
    return tuple(words[:, i].astype(np.uint64) for i in range(4))


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExternalVectorSet:
    """The shards and row hashes a manifest names, mapped read-only."""

    def __init__(self, manifest):
        path = Path(manifest)
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ContractError("external_vectors_manifest_unreadable") from None
        if (
            body.get("format") != FORMAT
            or body.get("dtype") != "float16"
            or body.get("input_spec") != EMBEDDING_INPUT
        ):
            raise ContractError("external_vectors_manifest_unsupported")
        self.manifest, self.base = body, path.parent
        self.dims, self.space = int(body["dims"]), body["space_fingerprint"]
        description = body.get("space")
        if (
            not isinstance(description, dict)
            or space_fingerprint(description) != self.space
            or description.get("dimensions") != self.dims
        ):
            raise ContractError("external_vectors_space_description_mismatch")
        self.shards, starts, total = [], [], 0
        for item in body["shards"]:
            try:
                shard = np.load(self._path(item["file"]), mmap_mode="r")
            except (OSError, ValueError):
                raise ContractError("external_vectors_shard_unreadable") from None
            if (
                shard.dtype != np.float16
                or shard.shape != (int(item["rows"]), self.dims)
                or not shard.flags.c_contiguous
            ):
                raise ContractError("external_vectors_shard_mismatch")
            self.shards.append(shard)
            starts.append(total)
            total += shard.shape[0]
        self.rows, self.starts = total, np.asarray(starts, dtype=np.int64)
        hashes = self._path(body["hashes"]["file"])
        try:
            if hashes.stat().st_size != total * HASH_BYTES:
                raise ContractError("external_vectors_hashes_mismatch")
            self._raw = (
                np.memmap(hashes, dtype=np.uint8, mode="r", shape=(total, HASH_BYTES))
                if total
                else np.empty((0, HASH_BYTES), dtype=np.uint8)
            )
        except OSError:
            raise ContractError("external_vectors_hashes_unreadable") from None
        # The fixed block partition: (shard, first local row, end local row, first global row).
        blocks = [
            (s, lo, min(lo + BLOCK_ROWS, shard.shape[0]), int(self.starts[s]) + lo)
            for s, shard in enumerate(self.shards)
            for lo in range(0, shard.shape[0], BLOCK_ROWS)
        ]
        self.blocks = blocks
        self.block_start = np.asarray([b[3] for b in blocks], dtype=np.int64)
        self.fingerprint = canonical_hash(
            {
                "format": FORMAT,
                "space": self.space,
                "dims": self.dims,
                "dtype": "float16",
                "input_spec": EMBEDDING_INPUT,
                "shards": [[int(s["rows"]), s["sha256"]] for s in body["shards"]],
                "hashes": body["hashes"]["sha256"],
            }
        )
        self._lock = threading.Lock()
        self._sorted = self._norms = self._bad = None
        self.timings = {}

    def _path(self, name):
        path = Path(name)
        return path if path.is_absolute() else self.base / path

    def space_with(self, dims):
        """The fingerprint of the set's space with another number of dimensions."""
        return space_fingerprint(self.manifest["space"] | {"dimensions": int(dims)})

    def describe(self):
        return {
            "set": self.fingerprint,
            "space": self.space,
            "rows": self.rows,
            "dims": self.dims,
            "shards": len(self.shards),
        }

    def verify(self):
        """Every shard and the hash file against the manifest's sha256 (reads every byte)."""
        bad = [
            item["file"]
            for item in self.manifest["shards"]
            if _file_sha256(self._path(item["file"])) != item["sha256"]
        ]
        hashes = self.manifest["hashes"]
        if _file_sha256(self._path(hashes["file"])) != hashes["sha256"]:
            bad.append(hashes["file"])
        if bad:
            raise ContractError("external_vectors_sha256_mismatch", detail={"files": bad})
        return {"files": len(self.manifest["shards"]) + 1, "ok": True}

    # --- linking -------------------------------------------------------------------------------
    def _lookup(self):
        with self._lock:
            if self._sorted is None:
                started = time.monotonic()
                keys = _keys(self._raw)
                linked = np.flatnonzero(keys[0] | keys[1] | keys[2] | keys[3])
                order = linked[
                    np.lexsort(tuple(keys[i][linked] for i in (3, 2, 1, 0)))
                ]  # stable: equal inputs keep row order
                self._sorted = (order, tuple(k[order] for k in keys))
                self.timings["lookup_seconds"] = round(time.monotonic() - started, 4)
            return self._sorted

    def rows_of(self, hashes):
        """External row per input hash (hex), -1 where the set has none; among rows with the same
        input, the first."""
        hashes = list(hashes)
        if not hashes or not self.rows:
            return np.full(len(hashes), -1, dtype=np.int64)
        raw = np.frombuffer(b"".join(bytes.fromhex(h) for h in hashes), dtype=np.uint8)
        return self.rows_of_bytes(raw.reshape(len(hashes), HASH_BYTES))

    def rows_of_bytes(self, raw):
        """`rows_of` for hashes given as (n, 32) bytes (a columnar catalog's column)."""
        out = np.full(len(raw), -1, dtype=np.int64)
        if not len(raw) or not self.rows:
            return out
        query = _keys(raw)
        order, keys = self._lookup()
        lo = np.searchsorted(keys[0], query[0], side="left")
        hi = np.searchsorted(keys[0], query[0], side="right")
        one = np.flatnonzero(hi - lo == 1)
        if one.size:
            pos = lo[one]
            same = (
                (keys[1][pos] == query[1][one])
                & (keys[2][pos] == query[2][one])
                & (keys[3][pos] == query[3][one])
            )
            out[one[same]] = order[pos[same]]
        for i in np.flatnonzero(hi - lo > 1).tolist():  # equal inputs or a shared prefix
            for pos in range(int(lo[i]), int(hi[i])):
                if all(keys[c][pos] == query[c][i] for c in (1, 2, 3)):
                    out[i] = order[pos]
                    break
        return out

    # --- rows ----------------------------------------------------------------------------------
    def block_rows(self, j, buffer32, buffer64=None):
        """Block `j` raised exactly from float16: float32 into `buffer32`, then float64 into
        `buffer64` if given. Returns (first global row, the filled buffer)."""
        s, lo, hi, start = self.blocks[j]
        n = hi - lo
        np.copyto(buffer32[:n], self.shards[s][lo:hi])
        if buffer64 is None:
            return start, buffer32[:n]
        np.copyto(buffer64[:n], buffer32[:n])
        return start, buffer64[:n]

    def gather(self, rows):
        """float64 rows (exact upcast) of the given global rows, in that order."""
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((rows.size, self.dims), dtype=np.float64)
        shard = np.searchsorted(self.starts, rows, side="right") - 1
        for s in np.unique(shard).tolist():
            pick = np.flatnonzero(shard == s)
            out[pick] = self.shards[s][rows[pick] - self.starts[s]].astype(np.float32)
        return out

    def vector(self, row):
        return self.gather([row])[0]

    def norms(self, threads=1):
        """float64 norm of every row (computed once, block by block) and the rows whose norm is
        zero or not finite (a malformed stored vector)."""
        with self._lock:
            if self._norms is not None:
                return self._norms, self._bad
        started = time.monotonic()
        norms = np.empty(self.rows, dtype=np.float64)

        def work(blocks):
            b32 = np.empty((BLOCK_ROWS, self.dims), dtype=np.float32)
            b64 = np.empty((BLOCK_ROWS, self.dims), dtype=np.float64)
            for j in blocks:
                start, block = self.block_rows(j, b32, b64)
                norms[start : start + len(block)] = np.linalg.norm(block, axis=1)

        _parallel(work, range(len(self.blocks)), threads)
        bad = ~np.isfinite(norms) | (norms == 0)
        with self._lock:
            if self._norms is None:
                self._norms, self._bad = norms, bad
                self.timings["norm_seconds"] = round(time.monotonic() - started, 4)
            return self._norms, self._bad


def _parallel(work, items, threads, pool=None):
    """Run `work(part)` over contiguous parts of `items` (one part per thread)."""
    items = list(items)
    if threads <= 1 or len(items) < 2:
        work(items)
        return
    size = -(-len(items) // threads)
    parts = [items[i : i + size] for i in range(0, len(items), size)]
    if pool is None:
        with ThreadPoolExecutor(max_workers=len(parts)) as own:
            list(own.map(work, parts))
    else:
        list(pool.map(work, parts))


def overlay_scores(matrix, rows, q, dims=None):
    """Canonical scores of resident matrix rows (gathered as `rank_rows` gathers them); with
    `dims`, of their first `dims` coordinates normalised again (query projection)."""
    out = np.empty(rows.size, dtype=np.float64)
    for start in range(0, rows.size, BLOCK_ROWS):
        part = rows[start : start + BLOCK_ROWS]
        block = matrix[part]
        if dims is not None:
            block = block[:, :dims]
            block = block / np.linalg.norm(block, axis=1, keepdims=True)
        out[start : start + part.size] = canonical_scores(block, q)
    return out


def projection_space(vectors, projection):
    """(space fingerprint, query dimensions) a query projection accepts, or None."""
    if projection is None:
        return None
    method, dims = projection.get("method"), int(projection.get("query_dimensions", 0))
    if method not in PROJECTIONS:
        raise ValueError("unknown_query_projection")
    if dims <= vectors.dims:
        raise ValueError("query_projection_needs_more_dimensions")
    return vectors.space_with(dims), dims


@dataclass
class ExternalView:
    has_vector: np.ndarray
    invalid: np.ndarray
    generation: dict
    _external: np.ndarray = field(repr=False)  # external row per catalog row, -1 if none
    _overlay: np.ndarray = field(repr=False)  # resident matrix row where _external is -1
    _snapshot: object = field(repr=False)
    _index: object = field(repr=False)
    _cut: int | None = field(default=None, repr=False)  # projected: the set's dimensions

    def rank(self, mask, query, top_k):
        return self._index.rank(self, mask, query, top_k)


class ExternalFloat16Index:
    """`VectorIndexPort` over an `ExternalVectorSet` and, for inputs the set lacks, the state
    store's canonical vectors (a resident local exact matrix)."""

    def __init__(self, vectors, state, *, score_dtype="float64", threads=1, projection=None):
        if score_dtype not in SCORE_DTYPES:
            raise ValueError("unknown_score_dtype")
        if int(threads) < 1:
            raise ValueError("threads_must_be_positive")
        self.vectors, self.state = vectors, state
        self.projection = None if projection is None else dict(projection)
        self.projected = projection_space(vectors, projection)
        self.blobs = ExternalBlobs(state, vectors, projection=projection)
        self.overlay = LocalExactIndex(state, matrix_dir=None)
        self.score_dtype, self.threads = score_dtype, int(threads)
        self._pool = ThreadPoolExecutor(max_workers=self.threads) if self.threads > 1 else None
        self._links, self._lock = {}, threading.Lock()
        self.stats = {"links": 0, "queries": 0, "blocks_scored": 0}

    def capabilities(self):
        return {
            "index": "external_float16",
            "search": EXACT_NATIVE,
            "stored_dtype": "float16",
            "score_dtype": self.score_dtype,
            "metric": "cosine",
            "normalization": "query_l2_float64_row_norm_float64",
            "block_partition": f"shard_aligned_{BLOCK_ROWS}",
            "tie_break": "canonical_key_order",
            "filter": "eligible_mask_before_top_k",
            "vectors": "external_read_only",
            "external": self.vectors.describe(),
            "query_projection": None
            if self.projected is None
            else {
                "method": self.projection["method"],
                "query_dimensions": self.projected[1],
                "stored_dimensions": self.vectors.dims,
                "release_watermark": "unavailable",
            },
            "overlay": "local_exact_resident",
            "snapshot": "authority_generations",
            "vector_history": "not_retained",
            "multi_process": "read_only_shared_files",
            "arm": None,
        }

    async def view(self, tenant, space, dims, catalog, *, run=None):
        if space == self.vectors.space and dims == self.vectors.dims:
            cut = None
        elif self.projected is not None and (space, dims) == self.projected:
            cut = self.vectors.dims
        else:
            raise ContractError("external_vectors_space_mismatch")
        snapshot = await self.overlay.snapshot(tenant, space, dims, run=run)
        key = (tenant, space)
        stamp = (catalog.generation, id(catalog), snapshot.stamp)
        with self._lock:
            cached = self._links.get(key)
        if cached is not None and cached[0] == stamp:
            external, overlay, invalid = cached[1:4]
        else:
            work = functools.partial(self._link, catalog, snapshot)
            external, overlay, invalid = await run(work) if run is not None else work()
            with self._lock:
                self._links[key] = (stamp, external, overlay, invalid, catalog)
                self.stats["links"] += 1
        return ExternalView(
            has_vector=(external >= 0) | (overlay >= 0),
            invalid=invalid,
            generation={
                "vector": list(snapshot.stamp),
                "layout": None,
                "external": self.vectors.fingerprint,
            },
            _external=external,
            _overlay=overlay,
            _snapshot=snapshot,
            _index=self,
            _cut=cut,
        )

    def _link(self, catalog, snapshot):
        hashes = catalog.input_hashes
        raw = getattr(catalog, "input_hash_bytes", None)  # N1F-6 columnar: no hex round trip
        external = self.vectors.rows_of(hashes) if raw is None else self.vectors.rows_of_bytes(raw)
        _, bad = self.vectors.norms(self.threads)
        linked = external >= 0
        invalid = np.zeros(len(hashes), dtype=bool)
        invalid[linked] = bad[external[linked]]
        external[invalid] = -1
        overlay = np.full(len(hashes), -1, dtype=np.int64)
        for i in np.flatnonzero(~linked).tolist():
            h = hashes[i]
            overlay[i] = snapshot.row_of_hash.get(h, -1)
            invalid[i] = h in snapshot.invalid
        return external, overlay, invalid

    # --- scoring -------------------------------------------------------------------------------
    def external_scores(self, rows, q):
        """Scan scores of external rows (any order, repeats allowed) for a normalised float64
        query: every block holding one of them is scored whole, in `score_dtype`."""
        vectors = self.vectors
        norms, _ = vectors.norms(self.threads)
        if 2 * rows.size >= vectors.rows:
            blocks = range(len(vectors.blocks))
        else:
            blocks = np.unique(np.searchsorted(vectors.block_start, rows, side="right") - 1)
        full = np.empty(vectors.rows, dtype=np.float64)
        wide = self.score_dtype == "float64"
        q32 = q.astype(np.float32)

        def work(part):
            b32 = np.empty((BLOCK_ROWS, vectors.dims), dtype=np.float32)
            b64 = np.empty((BLOCK_ROWS, vectors.dims), dtype=np.float64) if wide else None
            for j in part:
                start, block = vectors.block_rows(j, b32, b64)
                n = len(block)
                raw = block @ q if wide else (block @ q32).astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):  # invalid rows: never ranked
                    full[start : start + n] = raw / norms[start : start + n]

        blocks = list(blocks)
        _parallel(work, blocks, self.threads, self._pool)
        with self._lock:
            self.stats["blocks_scored"] += len(blocks)
        return full[rows]

    def gather_scores(self, rows, q):
        """Canonical float64 scores of a few external rows, gathered (rescoring)."""
        if rows.size == 0:
            return np.empty(0)
        norms, _ = self.vectors.norms(self.threads)
        return canonical_scores(self.vectors.gather(rows), q) / norms[rows]

    @staticmethod
    def query(view, query):
        """The normalised float64 query of a view (its first coordinates when projected)."""
        q = np.asarray(query, dtype=float)
        if view._cut is not None:
            q = q[: view._cut]
        return unit(q)

    def rank(self, view, mask, query, top_k):
        external, overlay = view._external, view._overlay
        ready = np.flatnonzero(mask & ((external >= 0) | (overlay >= 0)))
        with self._lock:
            self.stats["queries"] += 1
        if ready.size == 0:
            return ready, np.empty(0)
        q = self.query(view, query)
        scan = np.empty(ready.size, dtype=np.float64)
        rows = external[ready]
        outside = rows < 0
        if (~outside).any():
            scan[~outside] = self.external_scores(rows[~outside], q)
        if outside.any():  # few rows: scored canonically at once
            scan[outside] = overlay_scores(
                view._snapshot.matrix, overlay[ready[outside]], q, view._cut
            )

        def rescore(at):
            scores = scan[at]
            inside = rows[at] >= 0
            scores[inside] = self.gather_scores(rows[at][inside], q)
            return scores

        dtype = np.float32 if self.score_dtype == "float32" else np.float64
        order, scores = certified_top(scan, top_k, scan_bound(q.size, dtype), rescore)
        return ready[order], scores

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)
        self.overlay.close()  # 0.1.0 (parent PUB-4): the resident overlay's generations
        with self._lock:
            self._links.clear()


class ExternalBlobs:
    """The state store's vector blobs with the external set beside them. Reads answer from the
    set for the inputs it has (its space, or the projected space); writes go to the state store.
    The set's files are never written."""

    def __init__(self, state, vectors, *, projection=None):
        self.state, self.vectors = state, vectors
        projected = projection_space(vectors, projection)
        self.projected_space = None if projected is None else projected[0]

    def _external(self, space, hashes):
        if space not in (self.vectors.space, self.projected_space) or not hashes:
            return {}
        rows = self.vectors.rows_of(hashes)
        return {h: int(r) for h, r in zip(hashes, rows.tolist(), strict=True) if r >= 0}

    def put_vector(self, tenant, space, text, vector):
        return self.state.put_vector(tenant, space, text, vector)

    def vector(self, tenant, space, text):
        if space == self.projected_space:
            return None  # no watermark across dimensions: the release check stays unknown
        found = self._external(space, [input_hash(text)])
        if found:
            return self.vectors.vector(next(iter(found.values())))
        return self.state.vector(tenant, space, text)

    def vector_mask(self, tenant, space, catalog):
        """Per catalog row, whether a vector is held for its input (`rvg.storage.catalog.
        vector_mask`): the set's rows by bytes, the state store for the rest."""
        raw = getattr(catalog, "input_hash_bytes", None)
        if space in (self.vectors.space, self.projected_space):
            if raw is not None:
                mask = self.vectors.rows_of_bytes(raw) >= 0
            else:
                mask = self.vectors.rows_of(catalog.input_hashes) >= 0
        else:
            mask = np.zeros(len(catalog), dtype=bool)
        rest = np.flatnonzero(~mask)
        if rest.size:
            hashes = [catalog.input_hashes[i] for i in rest]
            have = self.state.has_vectors(tenant, space, set(hashes))
            mask[rest] = [h in have for h in hashes]
        return mask

    def has_vectors(self, tenant, space, hashes):
        hashes = list(dict.fromkeys(hashes))
        found = set(self._external(space, hashes))
        rest = [h for h in hashes if h not in found]
        return found | (set(self.state.has_vectors(tenant, space, rest)) if rest else set())

    def vectors(self, tenant, space, hashes):
        hashes = list(dict.fromkeys(hashes))
        found = {} if space == self.projected_space else self._external(space, hashes)
        out = {h: self.vectors.vector(r) for h, r in found.items()}
        rest = [h for h in hashes if h not in found]
        if rest:
            out.update(self.state.vectors(tenant, space, rest))
        return out

    def vector_seq(self, tenant, space):
        return self.state.vector_seq(tenant, space)

    def scan_vectors(self, tenant, space, *, source="binary"):
        return self.state.scan_vectors(tenant, space, source=source)

    def scan_changes(self, tenant, space, since):
        return self.state.scan_changes(tenant, space, since)

    def __getattr__(self, name):  # other state reads (vector_meta, ...)
        return getattr(self.state, name)


# --- the candidate stage ------------------------------------------------------------------------
@dataclass(eq=False)
class ExternalCodes:
    packed: np.ndarray  # uint8 (rows, width), one row per external row
    fingerprint: str  # the quantizer's
    vectors: str  # the set's
    sha256: str
    file: str | None = None


@dataclass
class ExternalCandidateView:
    has_vector: np.ndarray
    invalid: np.ndarray
    generation: dict
    _inner: ExternalView = field(repr=False)
    _index: object = field(repr=False)

    def rank(self, mask, query, top_k):
        return self._index.rank(self._inner, mask, query, top_k)


class ExternalCandidateIndex:
    """`VectorIndexPort`: the external index with N1F-4's TurboQuant candidate stage in front.
    Codes are built only by an explicit `build` (in memory, or into a file that `load` maps)."""

    def __init__(self, exact, *, ladder=None):
        from .quantized import LADDER

        if not isinstance(exact, ExternalFloat16Index):
            raise TypeError("external_candidate_stage_requires_external_index")
        self.exact, self.ladder = exact, tuple(ladder or LADDER)
        self.blobs = exact.blobs
        self.codes = None
        self._lock = threading.Lock()
        self.stats = {"queries": 0, "exact_fallback": 0, "exact_small": 0, "overlay_rows": 0}

    def capabilities(self):
        from .quantized import BITS, METHOD, SEED

        return self.exact.capabilities() | {
            "search": APPROXIMATE,
            "candidate_stage": {
                "method": METHOD,
                "bits": BITS,
                "seed": SEED,
                "ladder": list(self.ladder),
                "margin": "2D+10*sqrt(D/d)",
                "rescoring": "external_float64",
                "fallback": "exact_scan",
                "codes": "explicit_build",
            },
        }

    def _count(self, event, n=1):
        with self._lock:
            self.stats[event] = self.stats.get(event, 0) + n
            if event.startswith(("accepted_", "exact_")):
                self.stats["queries"] += 1

    def build(self, path=None):
        """Encode every external row (normalised in float64) and keep the codes; with `path`,
        write them to that file (`.npy`) and a sidecar `<path>.json`, and map the file."""
        from .quantized import quantizer

        vectors = self.exact.vectors
        quant = quantizer(vectors.dims)
        norms, _ = vectors.norms(self.exact.threads)
        started = time.monotonic()
        shape = (vectors.rows, quant.width)
        if path is None:
            packed = np.empty(shape, dtype=np.uint8)
        else:
            packed = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)

        def work(part):
            b32 = np.empty((BLOCK_ROWS, vectors.dims), dtype=np.float32)
            b64 = np.empty((BLOCK_ROWS, vectors.dims), dtype=np.float64)
            for j in part:
                start, block = vectors.block_rows(j, b32, b64)
                n = len(block)
                unit = block / norms[start : start + n, None]
                packed[start : start + n] = quant.encode(unit)

        _parallel(work, range(len(vectors.blocks)), self.exact.threads, self.exact._pool)
        encoded = time.monotonic() - started
        digest = hashlib.sha256()
        for start in range(0, vectors.rows, 1 << 16):
            digest.update(np.ascontiguousarray(packed[start : start + (1 << 16)]).data)
        codes = ExternalCodes(packed, quant.fingerprint, vectors.fingerprint, digest.hexdigest())
        if path is not None:
            packed.flush()
            del packed
            Path(f"{path}.json").write_text(
                json.dumps(
                    {
                        "rows": vectors.rows,
                        "width": quant.width,
                        "fingerprint": quant.fingerprint,
                        "vectors": vectors.fingerprint,
                        "sha256": codes.sha256,
                    }
                )
            )
            codes = self.load(path, verify=False)
        self.codes = codes
        return {
            "rows": vectors.rows,
            "bytes": int(vectors.rows * quant.width),
            "sha256": codes.sha256,
            "encode_seconds": round(encoded, 4),
            "seconds": round(time.monotonic() - started, 4),
            "file": None if path is None else str(path),
        }

    def load(self, path, *, verify=True):
        from .quantized import quantizer

        vectors = self.exact.vectors
        try:
            meta = json.loads(Path(f"{path}.json").read_text())
            packed = np.load(path, mmap_mode="r")
        except (OSError, ValueError):
            raise ContractError("quantized_codes_missing") from None
        quant = quantizer(vectors.dims)
        if (
            meta.get("fingerprint") != quant.fingerprint
            or meta.get("vectors") != vectors.fingerprint
            or packed.shape != (vectors.rows, quant.width)
        ):
            raise ContractError("quantized_codes_mismatch")
        if verify:
            digest = hashlib.sha256()
            for start in range(0, vectors.rows, 1 << 16):
                digest.update(np.ascontiguousarray(packed[start : start + (1 << 16)]).data)
            if digest.hexdigest() != meta["sha256"]:
                raise ContractError("quantized_codes_mismatch")
        self.codes = ExternalCodes(
            packed, meta["fingerprint"], meta["vectors"], meta["sha256"], file=str(path)
        )
        return self.codes

    async def view(self, tenant, space, dims, catalog, *, run=None):
        if self.codes is None:
            raise ContractError("quantized_codes_missing")
        inner = await self.exact.view(tenant, space, dims, catalog, run=run)
        return ExternalCandidateView(
            has_vector=inner.has_vector,
            invalid=inner.invalid,
            generation=inner.generation | {"codes": self.codes.sha256},
            _inner=inner,
            _index=self,
        )

    def rank(self, inner, mask, query, top_k):
        from .quantized import accept, best, quantizer

        exact = self.exact
        external, overlay = inner._external, inner._overlay
        ready = np.flatnonzero(mask & ((external >= 0) | (overlay >= 0)))
        steps = [s for s in self.ladder if s > top_k]
        covered = ready[external[ready] >= 0]
        if ready.size == 0 or not steps or covered.size <= steps[0]:
            self._count("exact_small")
            return exact.rank(inner, mask, query, top_k)
        q = exact.query(inner, query)
        quant = quantizer(exact.vectors.dims)
        wanted = external[covered]
        prepared = quant.prepare(q)
        codes = self.codes.packed
        if 2 * wanted.size >= len(codes):
            approx = self.scores(quant, codes, prepared)[wanted]
        else:
            approx = self.scores(quant, np.asarray(codes[wanted]), prepared)
        order = best(approx, min(steps[-1], covered.size))
        rest = ready[external[ready] < 0]
        fixed = (rest, overlay_scores(inner._snapshot.matrix, overlay[rest], q, inner._cut))
        self._count("overlay_rows", int(rest.size))
        found = accept(
            covered[order],
            approx[order],
            covered.size,
            top_k,
            lambda rows: exact.gather_scores(external[rows], q),
            quant.margin,
            ladder=self.ladder,
            fixed=fixed,
        )
        if found is None:
            self._count("exact_fallback")
            return exact.rank(inner, mask, query, top_k)
        rows, scores, step = found
        self._count(f"accepted_{step}")
        return rows, scores

    def scores(self, quant, codes, prepared):
        """`TurboQuant.scores` with its blocks spread over the index's threads (N1F-6): the same
        blocks from row 0 and the same operations, so the same bits."""
        threads = self.exact.threads
        if threads <= 1 or len(codes) <= 2 * BLOCK_ROWS:
            return quant.scores(codes, prepared)
        low, high = prepared
        out = np.empty(len(codes), dtype=np.float32)

        def work(part):
            for start in part:
                block = np.asarray(codes[start : start + BLOCK_ROWS])
                out[start : start + len(block)] = (
                    quant.levels[block & 15] @ low + quant.levels[block >> 4] @ high
                )

        _parallel(work, range(0, len(codes), BLOCK_ROWS), threads, self.exact._pool)
        return out / np.float32(quant.dims)

    def close(self):
        self.exact.close()


# --- writing a set ------------------------------------------------------------------------------
SHARD_ROWS = 8 * BLOCK_ROWS  # rows per shard file `write_external_set` writes (65,536)
MANIFEST_FILE, HASHES_FILE = "manifest.json", "hashes.bin"
_HEX = frozenset("0123456789abcdef")


def _space_description(space):
    """An `EmbeddingSpace`, or its description as a dict, as the description a manifest keeps."""
    description = asdict(space) if is_dataclass(space) else dict(space)
    dims = description.get("dimensions")
    if not isinstance(dims, int) or isinstance(dims, bool) or dims < 1:
        raise ValueError("external_vectors_space_dimensions_required")
    return description


def _row_input(row, index):
    """(32-byte input hash, vector) of one row: `(title, text, vector)` hashes the RVG embedding
    input of that title and text (`EMBEDDING_INPUT`); `(input hash, vector)` gives the hash as 64
    lowercase hex characters, or None for a row that is the embedding of no RVG input."""
    row = tuple(row)
    if len(row) == 3:
        title, text, vector = row
        if not isinstance(title, str) or not isinstance(text, str):
            raise ContractError("external_vectors_row_input_not_text", detail={"row": index})
        key = input_hash(embedding_input(SimpleNamespace(title=title, text=text)))
        return bytes.fromhex(key), vector
    if len(row) == 2:
        key, vector = row
        if key is None:
            return bytes(HASH_BYTES), vector
        if isinstance(key, str) and len(key) == 2 * HASH_BYTES and set(key) <= _HEX:
            return bytes.fromhex(key), vector
        raise ContractError("external_vectors_row_input_hash_invalid", detail={"row": index})
    raise ContractError("external_vectors_row_shape", detail={"row": index})


def _float16(vector, dims, index):
    """The row as float16, refused when it has the wrong length or no usable float16 value (a
    value that is not finite, overflows float16 or rounds to the zero vector)."""
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (dims,):
        raise ContractError(
            "external_vectors_row_dimensions", detail={"row": index, "dims": int(value.size)}
        )
    with np.errstate(over="ignore", invalid="ignore"):  # refused just below
        half = value.astype(np.float16)
    if not np.isfinite(half).all() or not half.any():
        raise ContractError("external_vectors_row_not_representable", detail={"row": index})
    return half


def write_external_set(out_dir, rows, space, *, shard_rows=SHARD_ROWS):
    """Write an external float16 vector set (`FORMAT`, docs/external-vectors.md) into the new or
    empty directory `out_dir` and return its manifest's path.

    `rows` yields `(title, text, vector)` or `(input_hash, vector)` (`input_hash` 64 lowercase
    hex characters, or None for a row that no catalog row reaches); the row order is the set's
    order, and among rows of one input a deployment uses the first. `space` is the
    `EmbeddingSpace` the vectors were computed in (or its description as a dict); every vector
    has its `dimensions` and is stored as float16, rounded to nearest. A shard holds `shard_rows`
    rows (the last one fewer) and is assembled in memory (`shard_rows` x dims x 2 bytes); a
    multiple of `BLOCK_ROWS` keeps every scan block full.

    The files are written in one pass, the manifest last. A refused row stops the write and
    removes the files written so far; the finished set is read back with `ExternalVectorSet`."""
    shard_rows = int(shard_rows)
    if shard_rows < 1:
        raise ValueError("shard_rows_must_be_positive")
    description = _space_description(space)
    dims = description["dimensions"]
    folder = Path(out_dir)
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise FileExistsError(f"external vector set directory is not empty: {folder}")
    created = not folder.exists()
    folder.mkdir(parents=True, exist_ok=True)
    buffer = np.empty((shard_rows, dims), dtype=np.float16)
    shards, written, filled, total = [], [], 0, 0
    digest = hashlib.sha256()

    def flush():
        nonlocal filled
        if filled:
            path = folder / f"shard-{len(shards):05d}.npy"
            written.append(path)
            np.save(path, buffer[:filled])
            shards.append({"file": path.name, "rows": filled, "sha256": _file_sha256(path)})
            filled = 0

    try:
        written.append(folder / HASHES_FILE)
        with open(folder / HASHES_FILE, "wb") as hashes:
            for index, row in enumerate(rows):
                raw, vector = _row_input(row, index)
                buffer[filled] = _float16(vector, dims, index)
                hashes.write(raw)
                digest.update(raw)
                filled += 1
                total += 1
                if filled == shard_rows:
                    flush()
            flush()
        if not total:
            raise ContractError("external_vectors_empty")
        manifest = {
            "format": FORMAT,
            "dtype": "float16",
            "dims": dims,
            "input_spec": EMBEDDING_INPUT,
            "space": description,
            "space_fingerprint": space_fingerprint(description),
            "shards": shards,
            "hashes": {"file": HASHES_FILE, "sha256": digest.hexdigest()},
        }
        path = folder / MANIFEST_FILE
        written.append(path)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ExternalVectorSet(path)  # the reader's own checks on what was written
        return path
    except BaseException:
        for item in written:
            item.unlink(missing_ok=True)
        if created:
            folder.rmdir()
        raise


def export_external_set(state, tenant, space, out_dir, *, shard_rows=SHARD_ROWS):
    """Write the vectors a state store holds for `tenant` in `space` (an `EmbeddingSpace` or its
    description) as an external set, in input-hash order (`write_external_set`). The set serves
    every tenant of the deployment that opens it: a catalog row finds its vector by input hash."""
    description = _space_description(space)
    with state.scan_vectors(tenant, space_fingerprint(description)) as scan:
        rows = ((key, _parse(payload, scan.source)) for key, _, payload in scan.rows)
        return write_external_set(out_dir, rows, description, shard_rows=shard_rows)
