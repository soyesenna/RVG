"""Exact dense search over canonical float64 vectors (`EXACT_CANONICAL`) behind the
`VectorIndexPort`: the N1 matrix search with the N1F-1 changes.

A matrix holds every vector of one tenant and embedding space, row-normalised float64, rows in
input-hash order, labelled with the space's `vector_seq` — the only generation that moves it.
Documents reach their row through the catalog's input hash (a *link*, rebuilt only when the
catalog or the matrix changes), so an insert without a vector, a revocation or an ACL change
never rewrites the matrix. A query scores the eligible rows exactly, keeps the catalog-order tie
rule of the stable sort, and returns catalog rows: the N1 semantics (filters before top-k, every
row tied with the k-th score kept, the same errors). The eligible rows are scanned with BLAS and
the rows that can reach the top-k are rescored in a fixed summation order (`canonical_scores`), so
a ranking and its scores are the same bits on every platform.

With a matrix directory the matrix is a file shared by every process of the host: one immutable
file per generation, published by atomically replacing a manifest. A builder holds a lease with a
fencing token; a builder whose lease was taken over (it stalled or died) cannot publish. Readers
pin the file they map with a shared lock and old files are removed only when unpinned; a file
removed under a reader stays readable (POSIX), so pins only bound disk use. Without a directory
the matrix stays in memory (single process). Nothing here depends on a knowledge base or a
language."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import functools
import hashlib
import json
import os
import socket
import struct
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import ContractError
from ..retrieval.data import canonical_hash
from .ports import EXACT_CANONICAL

BLOCK_ROWS = 8192  # rows normalised or scored per block: 8192 x 1536 float64 = 96 MiB at most
CANONICAL_ROWS = 512  # rows multiplied at once by `canonical_scores` (12 MiB at 3,072 dims)
LEASE_SECONDS = 120.0
MAGIC = b"RVGM\x02\x00\x00\x00"
PAGE = 4096


class BuildFault(Exception):
    """Raised by an injected fault (tests only)."""


@dataclass(eq=False)
class MatrixSnapshot:
    stamp: tuple
    hashes: tuple  # input hash per matrix row
    row_of_hash: dict
    invalid: frozenset  # hashes whose stored vector is malformed (not in the matrix)
    matrix: np.ndarray  # float64 (rows, dims), each row divided by its own norm
    token: int | None = None  # the publishing lease's fencing token (file mode)
    file: str | None = None

    @property
    def nbytes(self):
        return int(self.matrix.nbytes)


@dataclass
class LocalView:
    has_vector: np.ndarray
    invalid: np.ndarray
    generation: dict
    _rows: np.ndarray = field(repr=False)
    _snapshot: MatrixSnapshot = field(repr=False)

    def rank(self, mask, query, top_k):
        return rank_rows(self._rows, self._snapshot.matrix, mask, query, top_k)


def unit(q):
    """`q` divided by its float64 norm, the norm summed in a fixed order (numpy's pairwise sum,
    no BLAS) — the same bits on every platform, and the bits a matrix row of the same vector gets
    (`np.linalg.norm(..., axis=1)` sums the same way)."""
    q = np.asarray(q, dtype=np.float64)
    return q / np.sqrt(np.add.reduce(q * q))


def canonical_scores(block, q):
    """The canonical float64 scores of rows: every product rounded on its own, then numpy's
    pairwise sum along the row — numpy's own loops, no BLAS. The bits depend neither on the
    platform's BLAS (Accelerate, OpenBLAS, MKL each sum a matrix-vector product in their own
    order, and Accelerate's order also depends on the CPU and the block) nor on the other rows
    scored with the row. A ranking returns these scores. About ten times slower than a BLAS
    product, so a scan uses `scan_scores` and rescores only the rows that can reach the top-k
    (`certified_top`)."""
    block = np.asarray(block, dtype=np.float64)
    out = np.empty(block.shape[0], dtype=np.float64)
    for start in range(0, block.shape[0], CANONICAL_ROWS):
        part = block[start : start + CANONICAL_ROWS]
        out[start : start + part.shape[0]] = np.add.reduce(part * q, axis=1)
    return out


def scan_scores(block, q):
    """Scan scores: one BLAS matrix-vector product (fast; its bits are the platform's). They only
    choose which rows `certified_top` rescores, so they never reach a ranking."""
    return block @ q


def scan_bound(dims, dtype=np.float64):
    """A bound on |scan score - canonical score| of a row and a query of unit norm, whatever the
    summation order of either: a dot product of n terms is within n u / (1 - n u) of the sum of
    |x_i y_i| <= |x| |y| of the exact value (Higham, Accuracy and Stability of Numerical
    Algorithms, 2nd ed., §3.1). (dims + 2) units of roundoff of the scan's dtype (its terms, the
    rounded query, the result) plus as many of float64, times 4."""
    rounding = float(np.finfo(dtype).eps + np.finfo(np.float64).eps) / 2
    return 4.0 * (dims + 2) * rounding


def certified_top(scan, top_k, bound, rescore):
    """(positions, canonical scores) of the top-k positions by canonical score, best first, ties
    in position order. Only the positions whose scan score is within 2 `bound` of the k-th best
    scan score are rescored (`rescore(positions)`, ascending): any other position scores below
    the k positions of the k best scan scores whatever the summation orders, since each of the
    two scores is within `bound` of the other."""
    if top_k < scan.size:
        kth = np.partition(-scan, top_k - 1)[top_k - 1]
        reach = np.flatnonzero(-scan <= kth + 2 * bound)  # ascending: position order among ties
    else:
        reach = np.arange(scan.size)
    scores = rescore(reach)
    order = np.argsort(-scores, kind="stable")[:top_k]
    return reach[order], scores[order]


def rank_rows(matrix_row, matrix, mask, query, top_k):
    """(catalog rows of the top-k eligible rows with a vector, their canonical scores): eligible
    rows only, in catalog order, then the stable descending order (ties keep catalog order).
    Eligible rows are scanned with BLAS and the rows that can reach the top-k are rescored
    canonically (`certified_top`), so the ranking and its scores are the same bits on every
    platform."""
    ready = np.flatnonzero(mask & (matrix_row >= 0))
    if ready.size == 0:
        return ready, np.empty(0)
    q = unit(query)
    rows = matrix_row[ready]
    total = matrix.shape[0]
    if 2 * rows.size >= total:
        # Most rows are eligible: score every row in contiguous blocks (no copies) and pick the
        # eligible ones; the few extra dot products cost less than gathering scattered rows.
        full = np.empty(total, dtype=np.float64)
        for start in range(0, total, BLOCK_ROWS):
            full[start : start + BLOCK_ROWS] = scan_scores(matrix[start : start + BLOCK_ROWS], q)
        scan = full[rows]
    else:
        scan = np.empty(ready.size, dtype=np.float64)
        for start in range(0, ready.size, BLOCK_ROWS):
            part = rows[start : start + BLOCK_ROWS]
            # Matrix rows are in input-hash order, not catalog order: a run is contiguous only
            # if every step is +1 (checking the ends alone would accept a permutation).
            if part.size == 1 or (np.diff(part) == 1).all():  # contiguous ascending: no copy
                block = matrix[part[0] : part[-1] + 1]
            else:
                block = matrix[part]
            scan[start : start + part.size] = scan_scores(block, q)
    order, scores = certified_top(
        scan,
        top_k,
        scan_bound(q.size),
        lambda at: canonical_scores(matrix[rows[at]], q),
    )
    return ready[order], scores


def _parse(payload, source):
    if source == "binary":
        return np.frombuffer(payload, dtype="<f8").astype(np.float64, copy=False)
    return np.array(json.loads(payload), dtype=float)


def _alive(lease):
    if lease.get("host") != socket.gethostname():
        return True  # another host: trust the expiry time only
    try:
        os.kill(int(lease["pid"]), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class MatrixDirectory:
    """The files of one (tenant, space): `manifest.json` (the published generation), `lease.json`
    (the builder and the last issued fencing token), `lock` (short critical sections) and one
    `.rvgm` file per generation."""

    def __init__(self, root, tenant, space, version="v2"):
        # FIX-MATRIX-GEN: the rebuild layout keeps `v2/`; an incremental layout its own
        # `v3/<layout>/`, so a version that knows only the rebuild layout never reads its manifest
        self.dir = Path(root) / version / canonical_hash([tenant, space])[:24]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        self.lease_path = self.dir / "lease.json"

    def locked(self):
        return _Flock(self.dir / "lock")

    def read_manifest(self):
        try:
            return json.loads(self.manifest_path.read_text())
        except FileNotFoundError:
            return None

    def read_lease(self):
        try:
            return json.loads(self.lease_path.read_text())
        except FileNotFoundError:
            return {"token": None, "next": 0}

    def _write_atomic(self, path, value):
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        with open(tmp, "w") as handle:
            handle.write(json.dumps(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(self.dir)

    def acquire(self, stamp, *, now, ttl):
        """("done", manifest) | ("own", token, took_over) | ("wait", lease). A published
        generation at or past `stamp` is done — a query needs vectors at least as new as when it
        started, not exactly those. A live builder is waited for whatever it builds (builds of
        one tenant and space go one after another; a newer one continues from it); only an
        expired or dead builder is taken over, and its token is then fenced."""
        with self.locked():
            manifest = self.read_manifest()
            if manifest is not None and tuple(manifest["stamp"]) >= stamp:
                return ("done", manifest)
            lease = self.read_lease()
            active = lease.get("token") is not None and lease["expires"] > now and _alive(lease)
            if active:
                return ("wait", lease)
            token = int(lease.get("next", 0)) + 1
            self._write_atomic(
                self.lease_path,
                {
                    "token": token,
                    "next": token,
                    "stamp": list(stamp),
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "expires": now + ttl,
                },
            )
            return ("own", token, lease.get("token") is not None)

    def renew(self, token, *, now, ttl):
        with self.locked():
            lease = self.read_lease()
            if lease.get("token") != token:
                return False
            lease["expires"] = now + ttl
            self._write_atomic(self.lease_path, lease)
            return True

    def publish(self, token, stamp, tmp, final_name, meta, fault):
        """Rename the complete file into place and replace the manifest, only while this
        builder still holds the lease. Returns the manifest, or None if fenced."""
        with self.locked():
            lease = self.read_lease()
            if lease.get("token") != token:
                return None
            final = self.dir / final_name
            os.replace(tmp, final)
            _fsync_dir(self.dir)
            fault("after_rename")
            manifest = {"format": 2, "file": final_name, "stamp": list(stamp), "token": token}
            manifest.update(meta)
            self._write_atomic(self.manifest_path, manifest)
            fault("after_manifest")
            self._write_atomic(self.lease_path, {"token": None, "next": lease["next"]})
            return manifest

    def release(self, token):
        with self.locked():
            lease = self.read_lease()
            if lease.get("token") == token:
                self._write_atomic(self.lease_path, {"token": None, "next": lease["next"]})

    def discard(self, token):
        for path in self.dir.glob(f"*-{token}.rvgm.tmp"):
            path.unlink(missing_ok=True)

    def gc(self):
        """Remove generation files other than the published one unless a reader pins them, and
        temporary files that no live lease owns (a crashed or fenced builder's). Returns
        (removed, pinned)."""
        removed = pinned = 0
        with self.locked():
            manifest = self.read_manifest() or {}
            keep = set(manifest.get("files") or [manifest.get("file")])
            lease = self.read_lease()
            live = lease.get("token") if lease.get("token") is not None else None
            for path in self.dir.iterdir():
                name = path.name
                if name.endswith(".rvgm") and live is not None and f"-t{live}." in name:
                    continue  # a part the current lease owner is still writing (N1F-2 layouts)
                if name.endswith(".rvgm") and name not in keep:
                    try:
                        fd = os.open(path, os.O_RDONLY)
                    except FileNotFoundError:
                        continue
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pinned += 1
                        os.close(fd)
                        continue
                    path.unlink(missing_ok=True)
                    os.close(fd)
                    removed += 1
                elif name.endswith(".rvgm.tmp"):
                    token = int(name.split("-")[-1].split(".")[0])
                    if token != live:
                        path.unlink(missing_ok=True)
                        removed += 1
        return removed, pinned


class _Flock:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _no_fault(point):
    return None


class LocalExactIndex:
    """`VectorIndexPort` over a `VectorBlobStore`. `split_generations`, `source` and the
    catalog-cache arms exist for the N1F-1 comparison; the product uses the defaults."""

    def __init__(
        self,
        blobs,
        *,
        matrix_dir=None,
        state=None,
        split_generations=True,
        source="binary",
        lease_seconds=LEASE_SECONDS,
        poll_seconds=0.01,
        fault=None,
        clock=time.time,
    ):
        if source not in ("binary", "json"):
            raise ValueError("unknown_vector_source")
        if not split_generations and state is None:
            raise ValueError("state_required_without_split_generations")
        self.blobs, self.state = blobs, state
        self.matrix_dir = None if matrix_dir is None else Path(matrix_dir)
        self.split_generations, self.source = split_generations, source
        self.lease_seconds, self.poll_seconds = lease_seconds, poll_seconds
        self.fault, self.clock = fault or _no_fault, clock
        self._entries, self._links, self._building = {}, {}, {}
        self._lock = threading.Lock()
        self.stats = {
            "builds": 0,
            "adopted": 0,
            "waits": 0,
            "fenced": 0,
            "takeovers": 0,
            "gc_removed": 0,
            "gc_pinned": 0,
            "links": 0,
            "vector_bytes_read": 0,
            "matrix_bytes_written": 0,
        }
        self.build_seconds = []
        self.last_matrix_bytes = None

    def close(self):
        """Let go of the generations and links this index holds (0.1.0, parent PUB-4). A
        generation's pinned file and its memory map then close as soon as no view in flight still
        holds them, not when the garbage collector reaches the cycle the index is part of (a store
        and its storage refer to each other). Idempotent; a later query opens the published
        generation again."""
        with self._lock:
            self._entries.clear()
            self._links.clear()

    def capabilities(self):
        return {
            "index": "local_exact",
            "search": EXACT_CANONICAL,
            "stored_dtype": "float64",
            "metric": "cosine",
            "normalization": "row_l2_float64",
            "tie_break": "canonical_key_order",
            "filter": "eligible_mask_before_top_k",
            "snapshot": "authority_generations",
            "vector_history": "not_retained",
            "multi_process": "lease_fencing_manifest"
            if self.matrix_dir is not None
            else "single_process_resident",
            "arm": None
            if (self.split_generations and self.source == "binary")
            else {"split_generations": self.split_generations, "source": self.source},
        }

    def stamp(self, tenant, space):
        vectors = self.blobs.vector_seq(tenant, space)
        if self.split_generations:
            return (vectors,)
        return (self.state.snapshot(tenant), vectors)

    async def view(self, tenant, space, dims, catalog, *, run=None):
        snapshot = await self.snapshot(tenant, space, dims, run=run)
        key = (tenant, space)
        with self._lock:
            cached = self._links.get(key)
        if cached is not None and cached[0] == (catalog.generation, id(catalog), snapshot.stamp):
            rows, invalid = cached[1], cached[2]
        else:
            rows = np.fromiter(
                (snapshot.row_of_hash.get(h, -1) for h in catalog.input_hashes),
                dtype=np.int64,
                count=len(catalog),
            )
            invalid = np.fromiter(
                (h in snapshot.invalid for h in catalog.input_hashes),
                dtype=bool,
                count=len(catalog),
            )
            with self._lock:
                self._links[key] = (
                    (catalog.generation, id(catalog), snapshot.stamp),
                    rows,
                    invalid,
                    catalog,  # keeps id(catalog) unique while cached
                )
                self.stats["links"] += 1
        return LocalView(
            has_vector=rows >= 0,
            invalid=invalid,
            generation={"vector": list(snapshot.stamp), "layout": snapshot.token},
            _rows=rows,
            _snapshot=snapshot,
        )

    async def snapshot(self, tenant, space, dims, *, run=None):
        for _ in range(8):
            stamp = self.stamp(tenant, space)
            key = (tenant, space)
            with self._lock:
                entry = self._entries.get(key)
            if entry is not None and entry.stamp >= stamp:
                return entry
            flight = (tenant, space, stamp)
            pending = self._building.get(flight)
            if pending is not None and pending.get_loop() is asyncio.get_running_loop():
                result = await asyncio.shield(pending)
            else:
                future = self._building[flight] = asyncio.get_running_loop().create_future()
                try:
                    obtain = functools.partial(self._obtain, tenant, space, dims, stamp)
                    result = await run(obtain) if run is not None else obtain()
                except BaseException as exc:
                    future.set_exception(exc)
                    future.exception()
                    raise
                finally:
                    if self._building.get(flight) is future:
                        del self._building[flight]
                future.set_result(result)
            if result is not None:
                with self._lock:
                    current = self._entries.get(key)
                    if current is None or current.stamp <= result.stamp:
                        self._entries[key] = result
                current = entry = None  # the replaced snapshot's pin goes with its last reference
                if self.matrix_dir is not None:
                    self.collect(tenant, space)
                return result
            # A fenced build: read the stamp again.
        raise ContractError("dense_generation_unstable")

    def _directory(self, tenant, space):
        """The matrix directory of (tenant, space) in this index's layout."""
        return MatrixDirectory(self.matrix_dir, tenant, space)

    def collect(self, tenant, space):
        """Remove superseded, unpinned generation files and dead builders' temporary files."""
        removed, pinned = self._directory(tenant, space).gc()
        with self._lock:
            self.stats["gc_removed"] += removed
            self.stats["gc_pinned"] += pinned
        return removed, pinned

    # --- building ---------------------------------------------------------------------------
    def _obtain(self, tenant, space, dims, stamp):
        if self.matrix_dir is None:
            return self._build(tenant, space, dims, stamp, None, None)
        directory = self._directory(tenant, space)
        deadline = self.clock() + 4 * self.lease_seconds
        delay = self.poll_seconds
        while self.clock() < deadline:
            outcome = directory.acquire(stamp, now=self.clock(), ttl=self.lease_seconds)
            if outcome[0] == "done":
                opened = self._open(directory, outcome[1], dims)
                if opened is not None:
                    with self._lock:
                        self.stats["adopted"] += 1
                    return opened
                # The published file is unreadable: rebuild the same generation.
                outcome = ("own", *self._force(directory, stamp))
            if outcome[0] == "wait":
                with self._lock:
                    self.stats["waits"] += 1
                time.sleep(delay)
                delay = min(delay * 2, 0.2)
                continue
            _, token, took_over = outcome
            if took_over:
                with self._lock:
                    self.stats["takeovers"] += 1
            try:
                built = self._build(tenant, space, dims, stamp, directory, token)
            except OSError as exc:  # disk full, permissions, a vanished directory: ERROR
                directory.release(token)
                directory.discard(token)
                if directory.read_lease().get("token") not in (None, token):
                    with self._lock:
                        self.stats["fenced"] += 1
                    continue  # taken over (and collected) while stalled: not an error
                raise ContractError("dense_build_io", detail={"errno": exc.errno}) from None
            except BaseException:
                directory.release(token)
                directory.discard(token)
                raise
            if built is None:  # fenced: another builder owns this generation now
                with self._lock:
                    self.stats["fenced"] += 1
                continue
            return built
        raise ContractError("dense_build_wait_exceeded")

    def _force(self, directory, stamp):
        with directory.locked():
            lease = directory.read_lease()
            token = int(lease.get("next", 0)) + 1
            directory._write_atomic(
                directory.lease_path,
                {
                    "token": token,
                    "next": token,
                    "stamp": list(stamp),
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "expires": self.clock() + self.lease_seconds,
                },
            )
        return token, True

    def _build(self, tenant, space, dims, stamp, directory, token):
        started = time.monotonic()
        with self.blobs.scan_vectors(tenant, space, source=self.source) as scan:
            if self.split_generations:
                stamp = (scan.vector_seq,)  # label with what was actually read
            else:
                stamp = (stamp[0], scan.vector_seq)
            capacity = scan.count
            header_room = _header_room(capacity)
            name = f"m-{'.'.join(map(str, stamp))}-{token}.rvgm"
            if directory is None:
                matrix = np.empty((capacity, dims), dtype=np.float64)
                tmp = None
            else:
                tmp = directory.dir / (name + ".tmp")
                size = header_room + capacity * dims * 8
                with open(tmp, "wb") as handle:
                    handle.truncate(size)
                matrix = (
                    np.memmap(
                        tmp, dtype=np.float64, mode="r+", offset=header_room, shape=(capacity, dims)
                    )
                    if capacity
                    else None
                )
            hashes, invalid, count, read = [], set(), 0, 0
            buffer = np.empty((min(BLOCK_ROWS, max(capacity, 1)), dims), dtype=np.float64)
            filled = 0
            renewed = self.clock()

            def flush():
                nonlocal count, filled
                if filled:
                    self.fault("during_write")
                    block = buffer[:filled]
                    matrix[count : count + filled] = block / np.linalg.norm(
                        block, axis=1, keepdims=True
                    )
                    count += filled
                    filled = 0

            for input_hash_, _, payload in scan.rows:
                read += len(payload)
                value = _parse(payload, scan.source)
                if (
                    value.shape != (dims,)
                    or not np.isfinite(value).all()
                    or np.linalg.norm(value) == 0
                ):
                    invalid.add(input_hash_)
                    continue
                buffer[filled] = value
                filled += 1
                hashes.append(input_hash_)
                if filled == len(buffer):
                    flush()
                    if directory is not None and self.clock() - renewed > self.lease_seconds / 3:
                        if not directory.renew(token, now=self.clock(), ttl=self.lease_seconds):
                            break  # fenced mid-build: stop reading, publication will refuse
                        renewed = self.clock()
            flush()
        if directory is None:
            snapshot = MatrixSnapshot(
                stamp=stamp,
                hashes=tuple(hashes),
                row_of_hash={h: i for i, h in enumerate(hashes)},
                invalid=frozenset(invalid),
                matrix=matrix[:count],
            )
            self._built(started, read, 0, snapshot)
            return snapshot
        digest = hashlib.sha256()
        if matrix is not None:
            matrix.flush()
            for start in range(0, count, BLOCK_ROWS):
                digest.update(
                    np.ascontiguousarray(matrix[start : min(count, start + BLOCK_ROWS)]).data
                )
            del matrix
        header = json.dumps(
            {
                "stamp": list(stamp),
                "rows": count,
                "dims": dims,
                "hashes": hashes,
                "invalid": sorted(invalid),
                "sha256": digest.hexdigest(),
                "token": token,
            }
        ).encode()
        if len(header) + 16 > header_room:
            raise ContractError("dense_header_overflow")
        with open(tmp, "r+b") as handle:
            handle.write(MAGIC + struct.pack("<Q", len(header)) + header)
            handle.truncate(header_room + count * dims * 8)
            handle.flush()
            os.fsync(handle.fileno())
        self.fault("after_data_written")
        manifest = directory.publish(
            token,
            stamp,
            tmp,
            name,
            {
                "rows": count,
                "dims": dims,
                "sha256": digest.hexdigest(),
                "header_bytes": header_room,
            },
            self.fault,
        )
        if manifest is None:
            tmp.unlink(missing_ok=True)
            return None
        opened = self._open(directory, manifest, dims)
        if opened is None:
            raise ContractError("dense_published_file_unreadable")
        self._built(started, read, header_room + count * dims * 8, opened)
        return opened

    def _built(self, started, read, written, snapshot):
        with self._lock:
            self.stats["builds"] += 1
            self.stats["vector_bytes_read"] += read
            self.stats["matrix_bytes_written"] += written
            self.build_seconds.append(round(time.monotonic() - started, 4))
            self.last_matrix_bytes = snapshot.nbytes

    def _open(self, directory, manifest, dims):
        path = directory.dir / manifest["file"]
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)  # pin: GC leaves a pinned generation alone
            head = os.pread(fd, 16, 0)
            if len(head) != 16 or head[:8] != MAGIC:
                raise OSError(errno.EINVAL, "bad_matrix_file")
            length = struct.unpack("<Q", head[8:])[0]
            header = json.loads(os.pread(fd, length, 16))
            rows, offset = header["rows"], manifest["header_bytes"]
            if (
                header["dims"] != dims
                or header["token"] != manifest["token"]
                or os.fstat(fd).st_size != offset + rows * dims * 8
            ):
                raise OSError(errno.EINVAL, "matrix_file_mismatch")
            if rows:
                with open(fd, "rb", closefd=False) as handle:  # the map keeps its own handle
                    matrix = np.memmap(
                        handle, dtype=np.float64, mode="r", offset=offset, shape=(rows, dims)
                    )
            else:
                matrix = np.empty((0, dims), dtype=np.float64)
        except (OSError, ValueError, KeyError):
            os.close(fd)
            return None
        snapshot = MatrixSnapshot(
            stamp=tuple(header["stamp"]),
            hashes=tuple(header["hashes"]),
            row_of_hash={h: i for i, h in enumerate(header["hashes"])},
            invalid=frozenset(header["invalid"]),
            matrix=matrix,
            token=manifest["token"],
            file=manifest["file"],
        )
        weakref.finalize(snapshot, os.close, fd)  # the pin lasts as long as the snapshot
        return snapshot


def _header_room(capacity):
    need = 256 + capacity * 70
    return (need // PAGE + 1) * PAGE
