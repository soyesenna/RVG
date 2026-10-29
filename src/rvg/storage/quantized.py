"""An opt-in candidate stage in front of the local exact index (N1F-4): TurboQuant 4-bit codes
propose candidates, the canonical float64 matrix ranks them.

TurboQuant (Zandieh, Daliri, Hadian, Mirrokni, arXiv:2504.19874) needs no training data: a fixed
random rotation (the QR of a seeded Gaussian matrix) makes every coordinate of a unit vector,
times sqrt(d), close to N(0, 1), and each coordinate is quantized with the Lloyd-Max codebook of
N(0, 1) — 16 levels, two coordinates per byte. One seed, one codebook and one margin serve every
deployment; nothing is fitted to a knowledge base or a language.

A query scores the eligible rows that have codes (float32), takes the best 400 and rescores them
from the same matrix rows with the exact path's `canonical_scores`; the top-k is accepted only if
the 400th approximate score plus a margin is below the k-th canonical score, else 1,600, else the
eligible set is ranked by the exact path itself (`rank_rows`). The margin is 2D + 10 sqrt(D/d),
with D the codebook's distortion: a unit row's approximate score is about (1 - D) s minus noise of
standard deviation at most sqrt(D/d), so a true top-k row left out of an accepted ladder step needs
noise past ten standard deviations. That is a probability model, not a proof: the stage declares
`APPROXIMATE`.

Codes are written only by an explicit `build` (never by a query): one file per tenant and space
beside the matrix files, labelled with the vector generation it encoded and the quantizer's
fingerprint. A query without codes is `quantized_codes_missing`; rows whose vectors changed since
the codes, or that the codes do not have, are scored exactly every time (overlay), so a stale code
never decides anything."""

from __future__ import annotations

import errno
import functools
import hashlib
import json
import math
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import ConfigurationError, ContractError
from ..retrieval.data import canonical_hash
from .local_exact import BLOCK_ROWS, LocalExactIndex, canonical_scores, rank_rows, unit
from .ports import APPROXIMATE

METHOD = "turboquant_mse"
STAGES = ("none", "turboquant_mse4")
BITS = 4
SEED = 20261017
LADDER = (400, 1600)
Z = 10.0
MAGIC = b"RVGQ\x01\x00\x00\x00"
PAGE = 4096


# --- the quantizer ------------------------------------------------------------------------------
_erf = np.vectorize(math.erf)


def _Phi(x):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def _phi(x):
    out = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    out[~np.isfinite(x)] = 0.0
    return out


@functools.lru_cache(maxsize=8)
def lloyd_max_gaussian(bits, iters=2000, tol=1e-13):
    """The Lloyd-Max codebook of N(0, 1) with 2^bits levels and its decision boundaries."""
    c = np.linspace(-2.5, 2.5, 2**bits)
    for _ in range(iters):
        b = (c[:-1] + c[1:]) / 2.0
        edges = np.concatenate(([-np.inf], b, [np.inf]))
        mass = _Phi(edges[1:]) - _Phi(edges[:-1])
        c_new = (_phi(edges[:-1]) - _phi(edges[1:])) / mass
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            break
        c = c_new
    return c.astype(np.float64), ((c[:-1] + c[1:]) / 2.0).astype(np.float64)


def distortion(codebook, boundaries):
    """E[(Y - c(Y))^2] for Y ~ N(0, 1), computed from the cells: the codebook's own error."""
    edges = np.concatenate(([-np.inf], boundaries, [np.inf]))
    cdf, pdf = _Phi(edges), _phi(edges)
    mass = cdf[1:] - cdf[:-1]
    first = pdf[:-1] - pdf[1:]  # E[Y; cell]
    edge_pdf = np.where(np.isfinite(edges), np.nan_to_num(edges) * pdf, 0.0)
    second = mass - (edge_pdf[1:] - edge_pdf[:-1])  # E[Y^2; cell]
    return float(np.sum(second - 2.0 * codebook * first + codebook**2 * mass))


@functools.lru_cache(maxsize=4)
def rotation(dims, seed=SEED):
    """The Haar rotation (d x d, float32) of a seed; a row vector x is rotated as x @ R."""
    g = np.random.default_rng(seed).standard_normal((dims, dims))
    q, r = np.linalg.qr(g)
    q = np.ascontiguousarray(q * np.sign(np.diag(r))[None, :], dtype=np.float32)
    q.flags.writeable = False
    return q


class TurboQuant:
    """The 4-bit MSE quantizer of one dimension."""

    def __init__(self, dims, *, bits=BITS, seed=SEED):
        if bits != 4:
            raise ValueError("unsupported_quantizer_bits")
        self.dims, self.bits, self.seed = int(dims), bits, seed
        self.codebook, self.boundaries = lloyd_max_gaussian(bits)
        self.levels = self.codebook.astype(np.float32)
        self.rotation = rotation(self.dims, seed)
        self.distortion = distortion(self.codebook, self.boundaries)
        self.margin = 2.0 * self.distortion + Z * math.sqrt(self.distortion / self.dims)
        self.width = (self.dims + 1) // 2  # bytes per row: two 4-bit codes per byte
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "method": METHOD,
                    "bits": bits,
                    "seed": seed,
                    "dims": self.dims,
                    "codebook": self.codebook.tolist(),
                }
            ).encode()
        )
        digest.update(self.rotation.tobytes())
        self.fingerprint = digest.hexdigest()

    def describe(self):
        return {
            "method": METHOD,
            "bits": self.bits,
            "seed": self.seed,
            "dims": self.dims,
            "distortion": self.distortion,
            "margin": self.margin,
            "fingerprint": self.fingerprint,
        }

    def encode(self, rows):
        """Packed codes (n, width) of unit rows: byte j holds coordinate 2j in its low four bits
        and coordinate 2j + 1 in its high four bits."""
        y = math.sqrt(self.dims) * (np.asarray(rows, dtype=np.float32) @ self.rotation)
        code = np.searchsorted(self.boundaries, y, side="left").astype(np.uint8)
        if self.dims % 2:
            code = np.pad(code, ((0, 0), (0, 1)))
        return np.ascontiguousarray(code[:, 0::2] | (code[:, 1::2] << 4))

    def prepare(self, query):
        """The rotated unit query, split into the coordinates of the low and the high codes."""
        q = np.asarray(query, dtype=float)
        qy = math.sqrt(self.dims) * ((q / np.linalg.norm(q)).astype(np.float32) @ self.rotation)
        if self.dims % 2:
            qy = np.append(qy, np.float32(0.0))
        return np.ascontiguousarray(qy[0::2]), np.ascontiguousarray(qy[1::2])

    def scores(self, codes, prepared):
        """Approximate cosine scores (float32) of packed rows for one prepared query."""
        low, high = prepared
        out = np.empty(len(codes), dtype=np.float32)
        for start in range(0, len(codes), BLOCK_ROWS):
            block = np.asarray(codes[start : start + BLOCK_ROWS])
            out[start : start + len(block)] = (
                self.levels[block & 15] @ low + self.levels[block >> 4] @ high
            )
        return out / np.float32(self.dims)

    def scores_many(self, codes, queries):
        """Approximate scores (n, q) of packed rows for many queries at once (one matrix product
        per block; the evaluation's batch entry — its bits may differ from `scores`)."""
        prepared = [self.prepare(q) for q in queries]
        low = np.stack([p[0] for p in prepared], axis=1)
        high = np.stack([p[1] for p in prepared], axis=1)
        block = np.asarray(codes)
        return (self.levels[block & 15] @ low + self.levels[block >> 4] @ high) / np.float32(
            self.dims
        )


@functools.lru_cache(maxsize=8)
def quantizer(dims):
    return TurboQuant(dims)


def encode_matrix(quant, matrix):
    """Packed codes of a canonical matrix, block by block from the first row (deterministic)."""
    rows = matrix.shape[0]
    packed = np.empty((rows, quant.width), dtype=np.uint8)
    for start in range(0, rows, BLOCK_ROWS):
        packed[start : start + BLOCK_ROWS] = quant.encode(matrix[start : start + BLOCK_ROWS])
    return packed


# --- ranking ------------------------------------------------------------------------------------
def best(approx, count):
    """Positions of the `count` best approximate scores, best first; ties keep position order."""
    if count < approx.size:
        kth = np.partition(-approx, count - 1)[count - 1]
        found = np.flatnonzero(-approx <= kth)
    else:
        found = np.arange(approx.size)
    return found[np.argsort(-approx[found], kind="stable")][:count]


def accept(
    order_rows, order_approx, covered, top_k, canonical, margin, *, ladder=LADDER, fixed=None
):
    """The ladder over approximate candidates. `order_rows` are the covered eligible rows in
    approximate order (best first, at least min(ladder[-1], covered) of them) with their scores
    `order_approx`; `canonical(rows)` gives canonical scores; `fixed` = (rows, canonical scores)
    always in the pool (overlay). Returns (rows, scores, step) — the pool's top-k by (score desc,
    row asc) at the first step whose last approximate score plus the margin is below the k-th
    canonical score — or None when no step qualifies (the caller ranks exactly)."""
    if fixed is None:
        fixed = (np.empty(0, dtype=np.int64), np.empty(0))
    for step in (s for s in ladder if s > top_k):
        if step >= covered:
            return None
        rows = order_rows[:step]
        pool = np.concatenate([rows, fixed[0]])
        scores = np.concatenate([canonical(rows), fixed[1]])
        kth = -np.partition(-scores, top_k - 1)[top_k - 1]
        if float(order_approx[step - 1]) + margin < kth:
            pick = np.lexsort((pool, -scores))[:top_k]
            return pool[pick], scores[pick], step
    return None


def _canonical(matrix, matrix_rows, q):
    if len(matrix_rows) == 0:
        return np.empty(0)
    return canonical_scores(np.asarray(matrix[matrix_rows], dtype=np.float64), q)


def rank_quantized(
    matrix_row, matrix, code_row, codes, quant, mask, query, top_k, *, ladder=LADDER, count=None
):
    """(catalog rows of the top-k eligible rows with a vector, their scores), as `rank_rows`
    returns them, through the candidate stage. `code_row` maps a catalog row to its packed code
    row, -1 for a row ranked exactly every time (overlay); `count(event, n)` records what
    happened."""
    count = count or (lambda event, n=1: None)
    ready = np.flatnonzero(mask & (matrix_row >= 0))
    steps = [s for s in ladder if s > top_k]
    covered = ready[code_row[ready] >= 0]
    if ready.size == 0 or not steps or covered.size <= steps[0]:
        count("exact_small")
        return rank_rows(matrix_row, matrix, mask, query, top_k)
    q = unit(query)
    overlay = ready[code_row[ready] < 0]
    wanted = code_row[covered]
    prepared = quant.prepare(q)
    if 2 * wanted.size >= len(codes):
        approx = quant.scores(codes, prepared)[wanted]  # most codes needed: score all, no gather
    else:
        approx = quant.scores(np.asarray(codes[wanted]), prepared)
    order = best(approx, min(steps[-1], covered.size))
    fixed = (overlay, _canonical(matrix, matrix_row[overlay], q))
    count("overlay_rows", int(overlay.size))
    found = accept(
        covered[order],
        approx[order],
        covered.size,
        top_k,
        lambda rows: _canonical(matrix, matrix_row[rows], q),
        quant.margin,
        ladder=ladder,
        fixed=fixed,
    )
    if found is None:
        count("exact_fallback")
        return rank_rows(matrix_row, matrix, mask, query, top_k)
    rows, scores, step = found
    count(f"accepted_{step}")
    count("candidates_rescored", step)
    return rows, scores


# --- codes --------------------------------------------------------------------------------------
@dataclass(eq=False)
class Codes:
    frontier: int  # the vector_seq the codes encode
    hashes: tuple  # input hash per code row
    row_of_hash: dict
    packed: np.ndarray  # uint8 (rows, width)
    fingerprint: str
    sha256: str
    file: str | None = None
    signature: tuple | None = None  # (inode, size, mtime_ns) of the file read


@dataclass
class QuantizedView:
    has_vector: np.ndarray
    invalid: np.ndarray
    generation: dict
    _rows: np.ndarray = field(repr=False)
    _code_rows: np.ndarray = field(repr=False)
    _snapshot: object = field(repr=False)
    _codes: Codes = field(repr=False)
    _index: object = field(repr=False)
    _dims: int = field(repr=False)

    def rank(self, mask, query, top_k):
        index = self._index
        return rank_quantized(
            self._rows,
            self._snapshot.matrix,
            self._code_rows,
            self._codes.packed,
            quantizer(self._dims),
            mask,
            query,
            top_k,
            ladder=index.ladder,
            count=index._count,
        )


class QuantizedCandidateIndex:
    """`VectorIndexPort`: the local exact index with the TurboQuant candidate stage in front.
    `ladder` is a test hook; deployments use `LADDER`."""

    def __init__(self, exact, *, ladder=LADDER):
        if (
            type(exact) is not LocalExactIndex
            or not exact.split_generations
            or exact.source != "binary"
        ):
            raise ValueError("candidate_stage_requires_local_exact")
        self.exact, self.ladder = exact, tuple(ladder)
        self.blobs = exact.blobs
        self._codes, self._links = {}, {}
        self._lock = threading.Lock()
        self.stats = {
            "queries": 0,
            "accepted_" + str(self.ladder[0]): 0,
            "accepted_" + str(self.ladder[-1]): 0,
            "exact_fallback": 0,
            "exact_small": 0,
            "overlay_rows": 0,
            "candidates_rescored": 0,
            "codes_built": 0,
            "codes_loaded": 0,
        }

    def close(self):
        """The exact index under the stage, and the links this stage holds to its snapshots. The
        codes stay: they are read into memory, hold no file, and an in-memory build has no file
        to read them from again."""
        self.exact.close()
        with self._lock:
            self._links.clear()

    def capabilities(self):
        return self.exact.capabilities() | {
            "search": APPROXIMATE,
            "candidate_stage": {
                "method": METHOD,
                "bits": BITS,
                "seed": SEED,
                "ladder": list(self.ladder),
                "margin": "2D+10*sqrt(D/d)",
                "rescoring": "canonical_float64",
                "fallback": "exact_scan",
                "codes": "explicit_build",
            },
        }

    def _count(self, event, n=1):
        with self._lock:
            self.stats[event] = self.stats.get(event, 0) + n
            if event.startswith(("accepted_", "exact_")):
                self.stats["queries"] += 1

    def _path(self, tenant, space):
        folder = Path(self.exact.matrix_dir) / "quantized-v1" / canonical_hash([tenant, space])[:24]
        return folder / "codes.rvgq"

    # --- building -------------------------------------------------------------------------------
    async def build(self, tenant, space, dims, *, run=None):
        """Encode the current matrix of (tenant, space) and publish its codes (explicit only)."""
        snapshot = await self.exact.snapshot(tenant, space, dims, run=run)
        work = functools.partial(self._encode, tenant, space, dims, snapshot)
        return await run(work) if run is not None else work()

    def _encode(self, tenant, space, dims, snapshot):
        started = time.monotonic()
        quant = quantizer(dims)
        packed = encode_matrix(quant, snapshot.matrix)
        digest = hashlib.sha256(packed.tobytes()).hexdigest()
        codes = Codes(
            frontier=int(snapshot.stamp[-1]),
            hashes=tuple(snapshot.hashes),
            row_of_hash={h: i for i, h in enumerate(snapshot.hashes)},
            packed=packed,
            fingerprint=quant.fingerprint,
            sha256=digest,
        )
        if self.exact.matrix_dir is not None:
            codes = self._publish(tenant, space, dims, codes)
        with self._lock:
            self._codes[(tenant, space)] = codes
            self.stats["codes_built"] += 1
        return {
            "rows": len(codes.hashes),
            "bytes": int(packed.nbytes),
            "frontier": codes.frontier,
            "sha256": digest,
            "fingerprint": quant.fingerprint,
            "seconds": round(time.monotonic() - started, 4),
            "file": codes.file,
        }

    def _publish(self, tenant, space, dims, codes):
        path = self._path(tenant, space)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps(
            {
                "format": 1,
                "method": METHOD,
                "bits": BITS,
                "seed": SEED,
                "dims": dims,
                "rows": len(codes.hashes),
                "width": codes.packed.shape[1],
                "frontier": codes.frontier,
                "hashes": list(codes.hashes),
                "fingerprint": codes.fingerprint,
                "sha256": codes.sha256,
            }
        ).encode()
        room = ((16 + len(header)) // PAGE + 1) * PAGE
        tmp = path.with_name(path.name + f".tmp{os.getpid()}.{threading.get_ident()}")
        with open(tmp, "wb") as handle:
            handle.write(MAGIC + struct.pack("<Q", len(header)) + header)
            handle.write(b"\0" * (room - 16 - len(header)))
            handle.write(codes.packed.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return self._read(path, dims)

    # --- reading --------------------------------------------------------------------------------
    def _read(self, path, dims):
        try:
            with open(path, "rb") as handle:
                stat = os.fstat(handle.fileno())
                head = handle.read(16)
                if len(head) != 16 or head[:8] != MAGIC:
                    raise OSError(errno.EINVAL, "bad_codes_file")
                length = struct.unpack("<Q", head[8:])[0]
                header = json.loads(handle.read(length))
        except FileNotFoundError:
            raise ContractError("quantized_codes_missing") from None
        except (OSError, ValueError):
            raise ContractError("quantized_codes_mismatch") from None
        quant = quantizer(dims)
        room = ((16 + length) // PAGE + 1) * PAGE
        rows, width = header.get("rows"), header.get("width")
        if (
            header.get("method") != METHOD
            or header.get("bits") != BITS
            or header.get("seed") != SEED
            or header.get("dims") != dims
            or header.get("fingerprint") != quant.fingerprint
            or width != quant.width
            or stat.st_size != room + rows * width
        ):
            raise ContractError("quantized_codes_mismatch")
        packed = np.fromfile(path, dtype=np.uint8, offset=room).reshape(rows, width)
        if hashlib.sha256(packed.tobytes()).hexdigest() != header["sha256"]:
            raise ContractError("quantized_codes_mismatch")
        with self._lock:
            self.stats["codes_loaded"] += 1
        return Codes(
            frontier=int(header["frontier"]),
            hashes=tuple(header["hashes"]),
            row_of_hash={h: i for i, h in enumerate(header["hashes"])},
            packed=packed,
            fingerprint=header["fingerprint"],
            sha256=header["sha256"],
            file=path.name,
            signature=(stat.st_ino, stat.st_size, stat.st_mtime_ns),
        )

    def _current(self, tenant, space, dims):
        """The codes of (tenant, space): this process's, or the file's (re-read when replaced)."""
        with self._lock:
            codes = self._codes.get((tenant, space))
        if self.exact.matrix_dir is None:
            if codes is None:
                raise ContractError("quantized_codes_missing")
            return codes
        path = self._path(tenant, space)
        try:
            stat = path.stat()
        except FileNotFoundError:
            raise ContractError("quantized_codes_missing") from None
        if codes is not None and codes.signature == (stat.st_ino, stat.st_size, stat.st_mtime_ns):
            return codes
        codes = self._read(path, dims)
        with self._lock:
            self._codes[(tenant, space)] = codes
        return codes

    def _link(self, tenant, space, snapshot, codes):
        """Code row per matrix row, -1 where the codes do not hold the snapshot's bytes."""
        key = (tenant, space)
        with self._lock:
            cached = self._links.get(key)
        if cached is not None and cached[0] is snapshot and cached[1] is codes:
            return cached[2]
        changed = set()
        current = int(snapshot.stamp[-1])
        if current != codes.frontier:
            with self.blobs.scan_changes(tenant, space, min(current, codes.frontier)) as scan:
                changed = {row[0] for row in scan.rows}
        link = np.fromiter(
            (-1 if h in changed else codes.row_of_hash.get(h, -1) for h in snapshot.hashes),
            dtype=np.int64,
            count=len(snapshot.hashes),
        )
        with self._lock:
            self._links[key] = (snapshot, codes, link)
        return link

    async def view(self, tenant, space, dims, catalog, *, run=None):
        inner = await self.exact.view(tenant, space, dims, catalog, run=run)
        snapshot = inner._snapshot

        def attach():
            codes = self._current(tenant, space, dims)
            link = self._link(tenant, space, snapshot, codes)
            rows = inner._rows
            code_rows = np.full(rows.size, -1, dtype=np.int64)
            has = rows >= 0
            code_rows[has] = link[rows[has]]
            return codes, code_rows

        codes, code_rows = await run(attach) if run is not None else attach()
        return QuantizedView(
            has_vector=inner.has_vector,
            invalid=inner.invalid,
            generation=inner.generation | {"codes": codes.frontier},
            _rows=inner._rows,
            _code_rows=code_rows,
            _snapshot=snapshot,
            _codes=codes,
            _index=self,
            _dims=dims,
        )


def candidate_index(stage, exact):
    """The vector index of a deployment's `candidate_stage`."""
    if stage == "none":
        return exact
    if stage != "turboquant_mse4":
        raise ValueError("unknown_candidate_stage")
    return QuantizedCandidateIndex(exact)


async def build_codes(storage, tenant, space, dims, *, run=None):
    """Write the codes of (tenant, space) for a deployment whose candidate stage is on."""
    index = getattr(getattr(storage, "storage", storage), "index", None)
    if not isinstance(index, QuantizedCandidateIndex):
        raise ContractError("candidate_stage_off")
    return await index.build(tenant, space, dims, run=run)


async def build_candidate_codes(storage, tenant, space, dims=None, *, run=None):
    """`rvg.build_candidate_codes` (0.1.0, public): `build_codes` for a deployment whose
    candidate stage is on (`vectors.candidate_stage: turboquant_mse4`), after its vectors are in.
    `space` is the embedding space — an `EmbeddingSpace` (its dimensions are read from it) or
    its fingerprint, then with `dims`. Returns what was written."""
    fingerprint = getattr(space, "fingerprint", space)
    dims = getattr(space, "dimensions", None) if dims is None else dims
    if not isinstance(fingerprint, str) or not fingerprint or dims is None:
        raise ConfigurationError(
            "build_candidate_codes needs an EmbeddingSpace, or its fingerprint and dims"
        )
    return await build_codes(storage, tenant, fingerprint, int(dims), run=run)
