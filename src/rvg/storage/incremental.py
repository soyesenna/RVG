"""Exact incremental matrix layouts (N1F-2 arms) behind the same vector index port.

E0H (`local_exact.LocalExactIndex`) rewrites the whole matrix whenever a vector changes. The
layouts here keep what did not change and write only what did, and all three stay exact
(`EXACT_CANONICAL`): rows are the canonical float64 vectors, each divided by its own norm when
written, every eligible row is scored, ties keep catalog order.

- `base_delta` (E1): an immutable base file and one delta file. A generation rewrites the delta
  (its live rows + the new ones); a rewritten vector's old base row becomes a tombstone. When the
  delta holds more than `theta` of the live base rows, base and delta are merged into a new base.
- `segments` (E2): every generation adds one immutable segment with its new rows; old rows of
  rewritten vectors become tombstones. When a size tier holds `fanout` segments, those alone are
  merged (tombstones dropped) — local compaction.
- `paged` (E3): fixed pages of `page_rows` rows, one immutable file per page version. New rows
  fill the tail page (a new version of it) and then new pages; a rewrite writes a new version of
  the page holding that row. A row keeps its page and slot for life, so the row directory and the
  catalog link change only where vectors changed.

A generation is a manifest listing its part files; publication, leases, fencing tokens, reader
pins and garbage collection are `local_exact.MatrixDirectory`'s. A builder catches up from the
published generation through the blob store's change feed (`scan_changes`) instead of reading
every vector again. Nothing here depends on a knowledge base or a language."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import struct
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import ContractError
from .local_exact import (
    BLOCK_ROWS,
    MAGIC,
    LocalExactIndex,
    MatrixDirectory,
    _header_room,
    _parse,
    canonical_scores,
    certified_top,
    scan_bound,
    scan_scores,
    unit,
)
from .ports import EXACT_CANONICAL

LAYOUTS = ("base_delta", "segments", "paged")
DEFAULT_RECIPES = {
    "base_delta": {"theta": 0.10},
    "segments": {"base_rows": 4096, "fanout": 4},
    "paged": {"page_rows": 1024},
}


# --- part files --------------------------------------------------------------------------------
def write_part(path, matrix, hashes, meta):
    """An immutable part file (the `.rvgm` layout: magic, header JSON with the row hashes, page
    padding, float64 rows), written and fsynced under its final name. Only a manifest makes it
    visible. Returns (sha256 of the rows, bytes written)."""
    rows, dims = matrix.shape
    room = _header_room(rows)
    digest = hashlib.sha256()
    for start in range(0, rows, BLOCK_ROWS):
        digest.update(np.ascontiguousarray(matrix[start : start + BLOCK_ROWS]).data)
    header = json.dumps(
        {"rows": rows, "dims": dims, "hashes": list(hashes), "sha256": digest.hexdigest(), **meta}
    ).encode()
    if len(header) + 16 > room:
        raise ContractError("dense_header_overflow")
    with open(path, "wb") as handle:
        handle.write(MAGIC + struct.pack("<Q", len(header)) + header)
        handle.write(b"\0" * (room - 16 - len(header)))
        for start in range(0, rows, BLOCK_ROWS):
            handle.write(np.ascontiguousarray(matrix[start : start + BLOCK_ROWS]).tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest(), room + rows * dims * 8


class PartWriter:
    """A part file written progressively (rows in blocks, never all in memory): space for
    `capacity` rows is reserved, the header — with the row hashes, which are known only at the
    end — is written last, the file is cut to the rows written and fsynced."""

    def __init__(self, path, capacity, dims):
        self.path, self.dims = Path(path), dims
        self.room = _header_room(capacity)
        with open(self.path, "wb") as handle:
            handle.truncate(self.room + capacity * dims * 8)
        self.matrix = (
            np.memmap(
                self.path, dtype=np.float64, mode="r+", offset=self.room, shape=(capacity, dims)
            )
            if capacity
            else None
        )
        self.count, self.hashes = 0, []

    def add(self, rows, hashes):
        n = len(hashes)
        if n:
            self.matrix[self.count : self.count + n] = rows
            self.count += n
            self.hashes += list(hashes)

    def close(self, meta):
        digest = hashlib.sha256()
        if self.matrix is not None:
            self.matrix.flush()
            for start in range(0, self.count, BLOCK_ROWS):
                end = min(self.count, start + BLOCK_ROWS)
                digest.update(np.ascontiguousarray(self.matrix[start:end]).data)
            self.matrix = None
        header = json.dumps(
            {
                "rows": self.count,
                "dims": self.dims,
                "hashes": self.hashes,
                "offset": self.room,
                "sha256": digest.hexdigest(),
                **meta,
            }
        ).encode()
        if len(header) + 16 > self.room:
            raise ContractError("dense_header_overflow")
        size = self.room + self.count * self.dims * 8
        with open(self.path, "r+b") as handle:
            handle.write(MAGIC + struct.pack("<Q", len(header)) + header)
            handle.truncate(size)
            handle.flush()
            os.fsync(handle.fileno())
        return digest.hexdigest(), size


@dataclass(eq=False)
class Part:
    file: str
    matrix: np.ndarray
    hashes: tuple


def open_part(path, dims):
    """Map a part read-only and pin it (shared lock) for as long as the Part lives."""
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        head = os.pread(fd, 16, 0)
        if len(head) != 16 or head[:8] != MAGIC:
            raise OSError(errno.EINVAL, "bad_part_file")
        length = struct.unpack("<Q", head[8:])[0]
        header = json.loads(os.pread(fd, length, 16))
        rows = header["rows"]
        room = header.get("offset", _header_room(rows))
        if header["dims"] != dims or os.fstat(fd).st_size != room + rows * dims * 8:
            raise OSError(errno.EINVAL, "part_file_mismatch")
        if rows:
            with open(fd, "rb", closefd=False) as handle:
                matrix = np.memmap(
                    handle, dtype=np.float64, mode="r", offset=room, shape=(rows, dims)
                )
        else:
            matrix = np.empty((0, dims), dtype=np.float64)
    except BaseException:
        os.close(fd)
        raise
    part = Part(Path(path).name, matrix, tuple(header["hashes"]))
    weakref.finalize(part, os.close, fd)
    return part


def normalised(values, dims):
    if not len(values):
        return np.empty((0, dims), dtype=np.float64)
    block = np.asarray(values, dtype=np.float64)
    return block / np.linalg.norm(block, axis=1, keepdims=True)


# --- a generation ------------------------------------------------------------------------------
@dataclass(eq=False)
class Generation:
    stamp: tuple
    token: int
    entries: list  # the manifest's parts: file, offset, rows, dead, and tier/page
    parts: list  # [(offset, Part)] aligned with entries
    row_of_hash: dict
    invalid: frozenset
    total: int
    state: dict
    changed: frozenset | None = None  # hashes whose row moved, appeared or vanished
    parent: int | None = None  # token of the generation `changed` is relative to
    file: str | None = None

    @property
    def nbytes(self):
        return sum(p.matrix.nbytes for _, p in self.parts)


def rank_parts(matrix_row, parts, mask, query, top_k):
    """`rank_rows` over several parts: the eligible rows' scan scores in catalog order, each from
    a matrix-vector product over its own part, then the rows that can reach the top-k rescored
    canonically (`certified_top`): the stable descending order of canonical scores."""
    ready = np.flatnonzero(mask & (matrix_row >= 0))
    if ready.size == 0:
        return ready, np.empty(0)
    q = unit(query)
    rows = matrix_row[ready]
    scan = np.empty(rows.size, dtype=np.float64)
    for offset, part in parts:
        n = part.matrix.shape[0]
        if n == 0:
            continue
        inside = np.flatnonzero((rows >= offset) & (rows < offset + n))
        if inside.size == 0:
            continue
        local = rows[inside] - offset
        if 2 * inside.size >= n:
            full = np.empty(n, dtype=np.float64)
            for start in range(0, n, BLOCK_ROWS):
                full[start : start + BLOCK_ROWS] = scan_scores(
                    part.matrix[start : start + BLOCK_ROWS], q
                )
            scan[inside] = full[local]
            continue
        for start in range(0, local.size, BLOCK_ROWS):
            chunk = local[start : start + BLOCK_ROWS]
            where = inside[start : start + BLOCK_ROWS]
            if chunk.size == 1 or (np.diff(chunk) == 1).all():
                block = part.matrix[chunk[0] : chunk[-1] + 1]
            else:
                block = part.matrix[chunk]
            scan[where] = scan_scores(block, q)

    def rescore(at):
        scores = np.empty(at.size, dtype=np.float64)
        for offset, part in parts:
            n = part.matrix.shape[0]
            inside = np.flatnonzero((rows[at] >= offset) & (rows[at] < offset + n))
            if inside.size:
                scores[inside] = canonical_scores(part.matrix[rows[at][inside] - offset], q)
        return scores

    order, scores = certified_top(scan, top_k, scan_bound(q.size), rescore)
    return ready[order], scores


@dataclass
class PartsView:
    has_vector: np.ndarray
    invalid: np.ndarray
    generation: dict
    _rows: np.ndarray = field(repr=False)
    _snapshot: Generation = field(repr=False)

    def rank(self, mask, query, top_k):
        return rank_parts(self._rows, self._snapshot.parts, mask, query, top_k)


# --- the index ---------------------------------------------------------------------------------
class IncrementalExactIndex(LocalExactIndex):
    """`VectorIndexPort` with an incremental layout (`base_delta`, `segments`, `paged`)."""

    def __init__(self, blobs, *, matrix_dir, layout, recipe=None, **kw):
        if layout not in LAYOUTS:
            raise ValueError("unknown_matrix_layout")
        if matrix_dir is None:
            raise ValueError("incremental_layouts_need_a_matrix_directory")
        super().__init__(blobs, matrix_dir=matrix_dir, **kw)
        self.layout = layout
        self.recipe = {**DEFAULT_RECIPES[layout], **(recipe or {})}
        self._parts, self._parts_lock = {}, threading.Lock()
        self.stats.update(
            {
                "catch_ups": 0,
                "full_builds": 0,
                "merges": 0,
                "rows_written": 0,
                "rows_copied": 0,
                "incremental_links": 0,
                "incremental_directories": 0,
            }
        )

    def close(self):
        """`LocalExactIndex.close`, and the parts opened once per process (each a pinned file
        and a memory map)."""
        super().close()
        with self._parts_lock:
            self._parts.clear()

    def capabilities(self):
        return super().capabilities() | {
            "layout": self.layout,
            "recipe": dict(self.recipe),
            "search": EXACT_CANONICAL,
        }

    # --- views ---------------------------------------------------------------------------------
    async def view(self, tenant, space, dims, catalog, *, run=None):
        g = await self.snapshot(tenant, space, dims, run=run)
        key = (tenant, space)
        with self._lock:
            cached = self._links.get(key)
        same_catalog = cached is not None and cached[0][:2] == (catalog.generation, id(catalog))
        if same_catalog and cached[0][2] == g.stamp and cached[4] == g.token:
            rows, invalid = cached[1], cached[2]
        elif same_catalog and g.changed is not None and cached[4] == g.parent:
            # The matrix moved by one generation under the same catalog: only rows whose
            # vector moved, appeared or vanished change their link.
            rows, invalid = cached[1].copy(), cached[2].copy()
            for h in g.changed:
                where = catalog.rows_of_hash.get(h)
                if where is not None:
                    rows[where] = g.row_of_hash.get(h, -1)
                    invalid[where] = h in g.invalid
            with self._lock:
                self.stats["incremental_links"] += 1
        else:
            rows = np.fromiter(
                (g.row_of_hash.get(h, -1) for h in catalog.input_hashes),
                dtype=np.int64,
                count=len(catalog),
            )
            invalid = np.fromiter(
                (h in g.invalid for h in catalog.input_hashes), dtype=bool, count=len(catalog)
            )
            with self._lock:
                self.stats["links"] += 1
        with self._lock:
            self._links[key] = (
                (catalog.generation, id(catalog), g.stamp),
                rows,
                invalid,
                catalog,
                g.token,
            )
        return PartsView(
            has_vector=rows >= 0,
            invalid=invalid,
            generation={"vector": list(g.stamp), "layout": g.token, "kind": self.layout},
            _rows=rows,
            _snapshot=g,
        )

    # --- obtaining a generation ----------------------------------------------------------------
    def _directory(self, tenant, space):
        """FIX-MATRIX-GEN: `v3/<layout>/`, apart from the rebuild layout's `v2/` (a version that
        knows only the rebuild layout never reads a format-3 manifest)."""
        return MatrixDirectory(self.matrix_dir, tenant, space, version=f"v3/{self.layout}")

    def _obtain(self, tenant, space, dims, stamp):
        directory = self._directory(tenant, space)
        deadline = self.clock() + 4 * self.lease_seconds
        delay = self.poll_seconds
        while self.clock() < deadline:
            outcome = directory.acquire(stamp, now=self.clock(), ttl=self.lease_seconds)
            if outcome[0] == "done":
                opened = self._open_generation(
                    directory, outcome[1], dims, self._held(tenant, space)
                )
                if opened is not None:
                    with self._lock:
                        self.stats["adopted"] += 1
                    return opened
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
                built = self._advance(directory, tenant, space, dims, token)
            except OSError as exc:
                directory.release(token)
                self._discard(directory, token)
                if directory.read_lease().get("token") not in (None, token):
                    with self._lock:
                        self.stats["fenced"] += 1
                    continue  # taken over (and collected) while stalled: not an error
                raise ContractError("dense_build_io", detail={"errno": exc.errno}) from None
            except BaseException:
                directory.release(token)
                self._discard(directory, token)
                raise
            if built is None:
                with self._lock:
                    self.stats["fenced"] += 1
                continue
            return built
        raise ContractError("dense_build_wait_exceeded")

    def _held(self, tenant, space):
        with self._lock:
            return self._entries.get((tenant, space))

    def _discard(self, directory, token):
        for path in directory.dir.glob(f"*-t{token}.rvgm"):
            path.unlink(missing_ok=True)

    def _part(self, directory, name, dims):
        # FIX-MATRIX-GEN: parts are cached by directory and name — two tenants' generations may
        # carry the same file name (the stamp and fencing token count per directory)
        key = (str(directory.dir), name)
        with self._parts_lock:
            part = self._parts.get(key)
        if part is None:
            part = open_part(directory.dir / name, dims)
            with self._parts_lock:
                self._parts[key] = part
        return part

    def _open_generation(self, directory, manifest, dims, held=None):
        """A published generation: its parts (each opened once per process), the invalid set
        and the row directory — updated from the held generation when the manifest says it is
        that generation's child and no row of it moved, rebuilt otherwise."""
        if manifest.get("format") != 3 or manifest.get("layout") != self.layout:
            return None
        try:
            parts = [
                (e["offset"], self._part(directory, e["file"], dims)) for e in manifest["parts"]
            ]
        except (OSError, ValueError, KeyError):
            return None
        dead = {(e["file"], d) for e in manifest["parts"] for d in e["dead"]}
        changed = manifest.get("changed")
        incremental = (
            held is not None
            and changed is not None
            and manifest.get("parent") == held.token
            and not manifest.get("moved")
        )
        if incremental:
            changed = frozenset(changed)
            row_of_hash = dict(held.row_of_hash)
            for h in changed:
                row_of_hash.pop(h, None)
            fresh = {e["file"] for e in manifest["parts"]} - {e["file"] for e in held.entries}
            for offset, part in parts:
                if part.file in fresh:
                    for local, h in enumerate(part.hashes):
                        if h in changed and (part.file, local) not in dead:
                            row_of_hash[h] = offset + local
            with self._lock:
                self.stats["incremental_directories"] += 1
        else:
            row_of_hash = {
                h: offset + local
                for offset, part in parts
                for local, h in enumerate(part.hashes)
                if (part.file, local) not in dead
            }
            changed = None
        with self._parts_lock:
            live = {(str(directory.dir), e["file"]) for e in manifest["parts"]}
            for key in [k for k in self._parts if k[0] == str(directory.dir) and k not in live]:
                del self._parts[key]
        return Generation(
            stamp=tuple(manifest["stamp"]),
            token=manifest["token"],
            entries=[dict(e) for e in manifest["parts"]],
            parts=parts,
            row_of_hash=row_of_hash,
            invalid=frozenset(manifest.get("invalid", ())),
            total=manifest["total"],
            state=manifest.get("state", {}),
            changed=changed,
            parent=manifest.get("parent") if incremental else None,
        )

    def _advance(self, directory, tenant, space, dims, token):
        """The generation for the current vectors: a catch-up from the published one through
        the change feed, or a full build when none is published in this layout."""
        started = time.monotonic()
        published = directory.read_manifest()
        held = self._held(tenant, space)
        parent = None
        if published is not None and published.get("layout") == self.layout:
            parent = (
                held
                if held is not None and held.token == published.get("token")
                else self._open_generation(directory, published, dims, held)
            )
        if parent is not None and not parent.entries:
            parent = None  # nothing published yet but an empty generation: build in full
        if parent is None:
            with self.blobs.scan_vectors(tenant, space) as scan:
                target = (scan.vector_seq,)
                plan = self._initial(directory, scan, dims, token, target)
            with self._lock:
                self.stats["full_builds"] += 1
        else:
            with self.blobs.scan_changes(tenant, space, parent.stamp[0]) as scan:
                target = (scan.vector_seq,)
                plan = self._catch_up(directory, parent, list(scan.rows), dims, token, target)
            with self._lock:
                self.stats["catch_ups"] += 1
        self.fault("after_data_written")
        body = {
            "format": 3,
            "layout": self.layout,
            "parts": plan["parts"],
            "files": [e["file"] for e in plan["parts"]],
            "total": plan["total"],
            "invalid": sorted(plan["invalid"]),
            "state": plan["state"],
            "moved": plan["moved"],
            "parent": parent.token if parent is not None else None,
            "changed": sorted(plan["changed"]) if plan["changed"] is not None else None,
        }
        manifest = self._publish(directory, token, target, body)
        if manifest is None:
            self._discard(directory, token)
            return None
        g = self._open_generation(directory, manifest, dims, parent)
        if g is None:
            raise ContractError("dense_published_generation_unreadable")
        with self._lock:
            self.stats["builds"] += 1
            self.stats["vector_bytes_read"] += plan["read"]
            self.stats["matrix_bytes_written"] += plan["written"]
            self.build_seconds.append(round(time.monotonic() - started, 4))
            self.last_matrix_bytes = g.nbytes
        return g

    def _publish(self, directory, token, stamp, body):
        with directory.locked():
            lease = directory.read_lease()
            if lease.get("token") != token:
                return None
            self.fault("after_rename")
            manifest = {**body, "stamp": list(stamp), "token": token}
            directory._write_atomic(directory.manifest_path, manifest)
            self.fault("after_manifest")
            directory._write_atomic(directory.lease_path, {"token": None, "next": lease["next"]})
            return manifest

    # --- layouts -------------------------------------------------------------------------------
    def _parse(self, changes, dims):
        """(valid {hash: vector}, invalid hashes, bytes read) of changed vectors."""
        valid, invalid, read = {}, set(), 0
        for row in changes:
            h, payload = row[0], row[2]
            read += len(payload)
            value = _parse(payload, "binary")
            if value.shape != (dims,) or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
                invalid.add(h)
                valid.pop(h, None)
            else:
                valid[h] = value
                invalid.discard(h)
        return valid, invalid, read

    def _tier(self, rows):
        tier = 0
        while rows > self.recipe["base_rows"] * self.recipe["fanout"] ** tier:
            tier += 1
        return tier

    def _emit(self, directory, name, sources, dims, token, *, copied=0):
        """Write one part from `sources` — (matrix, local rows, hashes) each — block by block, so
        no source is ever read into memory whole. Returns bytes written."""
        capacity = sum(len(rows) for _, rows, _ in sources)
        writer = PartWriter(directory.dir / name, capacity, dims)
        for matrix, rows, hashes in sources:
            rows = list(rows)
            for start in range(0, len(rows), BLOCK_ROWS):
                chunk = rows[start : start + BLOCK_ROWS]
                writer.add(np.asarray(matrix[chunk]), hashes[start : start + BLOCK_ROWS])
        _, size = writer.close({"token": token})
        with self._lock:
            self.stats["rows_written"] += writer.count
            self.stats["rows_copied"] += copied
        return size

    def _initial(self, directory, scan, dims, token, stamp):
        """A full build streamed from the vector scan (input-hash order): one part (E1 base, E2
        segment) or pages of `page_rows` (E3), written block by block."""
        paged = self.layout == "paged"
        size = self.recipe["page_rows"] if paged else max(scan.count, 1)
        kind = {"base_delta": "base", "segments": "seg", "paged": "page"}[self.layout]
        parts, invalid, read, written = [], set(), 0, 0
        state = {"writer": None, "name": None, "page": 0}
        block, hashes = [], []

        def open_part_writer():
            k = state["page"]
            name = (
                f"page-{k}-{stamp[0]}-t{token}.rvgm"
                if paged
                else (f"{kind}-{stamp[0]}-0-t{token}.rvgm")
            )
            state["writer"], state["name"] = PartWriter(directory.dir / name, size, dims), name

        def flush():
            if block:
                state["writer"].add(normalised(block, dims), hashes)
                block.clear()
                hashes.clear()

        def close_part_writer():
            nonlocal written
            flush()
            writer = state["writer"]
            written += writer.close({"token": token})[1]
            entry = {
                "file": state["name"],
                "offset": state["page"] * size if paged else 0,
                "rows": writer.count,
                "dead": [],
            }
            if paged:
                entry["page"] = state["page"]
            if self.layout == "segments":
                entry["tier"] = self._tier(writer.count)
            parts.append(entry)
            with self._lock:
                self.stats["rows_written"] += writer.count
            state["writer"] = None
            state["page"] += 1

        for row in scan.rows:
            h, payload = row[0], row[2]
            read += len(payload)
            value = _parse(payload, "binary")
            if value.shape != (dims,) or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
                invalid.add(h)
                continue
            if state["writer"] is None:
                open_part_writer()
            block.append(value)
            hashes.append(h)
            if len(block) == BLOCK_ROWS or (paged and state["writer"].count + len(block) == size):
                flush()
            if paged and state["writer"].count == size:
                close_part_writer()
        if state["writer"] is not None:
            close_part_writer()
        total = parts[-1]["offset"] + parts[-1]["rows"] if parts else 0
        return {
            "parts": parts,
            "total": total,
            "invalid": invalid,
            "written": written,
            "state": {"page_rows": size} if paged else {},
            "moved": True,
            "changed": None,
            "read": read,
        }

    def _catch_up(self, directory, parent, changes, dims, token, stamp):
        valid, bad, read = self._parse(changes, dims)
        changed = {row[0] for row in changes}
        invalid = (set(parent.invalid) - set(valid)) | bad
        entries = [dict(e, dead=set(e["dead"])) for e in parent.entries]
        by_file = {part.file: part for _, part in parent.parts}
        live = {}  # changed hash -> (entry index, local row) of its live row in the parent
        for i, e in enumerate(entries):
            for local, h in enumerate(by_file[e["file"]].hashes):
                if h in changed and local not in e["dead"]:
                    live[h] = (i, local)
        layout = {
            "base_delta": self._base_delta,
            "segments": self._segments,
            "paged": self._paged,
        }[self.layout]
        plan = layout(directory, entries, by_file, live, valid, dims, token, stamp)
        for e in plan["parts"]:
            e["dead"] = sorted(e["dead"])
        total = plan["parts"][-1]["offset"] + plan["parts"][-1]["rows"] if plan["parts"] else 0
        plan.update(total=total, invalid=invalid, changed=changed, read=read)
        return plan

    def _base_delta(self, directory, entries, by_file, live, valid, dims, token, stamp):
        base = entries[0]
        delta = entries[1] if len(entries) > 1 else None
        for i, local in live.values():  # rewritten or now invalid: the old row is dead
            entries[i]["dead"].add(local)
        base_part = by_file[base["file"]]
        keep, keep_hashes = [], []
        if delta is not None:
            part = by_file[delta["file"]]
            keep = [r for r in range(delta["rows"]) if r not in delta["dead"]]
            keep_hashes = [part.hashes[r] for r in keep]
        new_hashes = sorted(valid)
        fresh = normalised([valid[h] for h in new_hashes], dims)
        live_base = base["rows"] - len(base["dead"])
        if len(keep) + len(new_hashes) > self.recipe["theta"] * max(live_base, 1):
            base_rows = [r for r in range(base["rows"]) if r not in base["dead"]]
            sources = [(base_part.matrix, base_rows, [base_part.hashes[r] for r in base_rows])]
            if delta is not None:
                sources.append((by_file[delta["file"]].matrix, keep, keep_hashes))
            sources.append((fresh, range(len(new_hashes)), new_hashes))
            name = f"base-{stamp[0]}-0-t{token}.rvgm"
            written = self._emit(
                directory, name, sources, dims, token, copied=len(base_rows) + len(keep)
            )
            rows = len(base_rows) + len(keep) + len(new_hashes)
            with self._lock:
                self.stats["merges"] += 1
            return {
                "parts": [{"file": name, "offset": 0, "rows": rows, "dead": set()}],
                "state": {},
                "moved": True,
                "written": written,
            }
        parts, written = [dict(base)], 0
        rows = len(keep) + len(new_hashes)
        if rows:
            name = f"delta-{stamp[0]}-0-t{token}.rvgm"
            sources = [(fresh, range(len(new_hashes)), new_hashes)]
            if delta is not None:
                sources.insert(0, (by_file[delta["file"]].matrix, keep, keep_hashes))
            written = self._emit(directory, name, sources, dims, token, copied=len(keep))
            parts.append({"file": name, "offset": base["rows"], "rows": rows, "dead": set()})
        moved = delta is not None and len(keep) != delta["rows"]
        return {"parts": parts, "state": {}, "moved": moved, "written": written}

    def _segments(self, directory, entries, by_file, live, valid, dims, token, stamp):
        for i, local in live.values():
            entries[i]["dead"].add(local)
        segments = [dict(e) for e in entries]
        fresh = {}  # file -> (matrix, hashes) of segments made in this generation, not yet written
        new_hashes = sorted(valid)
        if new_hashes:
            name = f"seg-{stamp[0]}-0-t{token}.rvgm"
            fresh[name] = (normalised([valid[h] for h in new_hashes], dims), new_hashes)
            segments.append(
                {
                    "file": name,
                    "rows": len(new_hashes),
                    "dead": set(),
                    "tier": self._tier(len(new_hashes)),
                }
            )
        moved, k, written = False, 1, 0
        while True:
            tiers = {}
            for seg in segments:
                tiers.setdefault(seg["tier"], []).append(seg)
            full = sorted(t for t, group in tiers.items() if len(group) >= self.recipe["fanout"])
            if not full:
                break
            group = tiers[full[0]][: self.recipe["fanout"]]
            sources, copied, hashes = [], 0, 0
            for seg in group:
                keep = [r for r in range(seg["rows"]) if r not in seg["dead"]]
                if seg["file"] in fresh:
                    matrix, hs = fresh.pop(seg["file"])
                else:
                    part = by_file[seg["file"]]
                    matrix, hs = part.matrix, part.hashes
                    copied += len(keep)
                sources.append((matrix, keep, [hs[r] for r in keep]))
                hashes += len(keep)
            name = f"seg-{stamp[0]}-{k}-t{token}.rvgm"
            k += 1
            written += self._emit(directory, name, sources, dims, token, copied=copied)
            position = segments.index(group[0])
            segments = [seg for seg in segments if seg not in group]
            segments.insert(
                position,
                {
                    "file": name,
                    "rows": hashes,
                    "dead": set(),
                    "tier": max(full[0] + 1, self._tier(hashes)),
                },
            )
            # A merge can cascade into the next tier within this generation: read it back.
            by_file[name] = open_part(directory.dir / name, dims)
            moved = True
            with self._lock:
                self.stats["merges"] += 1
        for name, (matrix, hs) in fresh.items():
            written += self._emit(directory, name, [(matrix, range(len(hs)), hs)], dims, token)
        offset = 0
        for seg in segments:
            seg["offset"] = offset
            offset += seg["rows"]
        return {"parts": segments, "state": {}, "moved": moved, "written": written}

    def _paged(self, directory, entries, by_file, live, valid, dims, token, stamp):
        size = self.recipe["page_rows"]
        touched = {}  # page index -> [matrix copy, hashes, appended rows]

        def page(k):
            if k not in touched:
                if k < len(entries):
                    part = by_file[entries[k]["file"]]
                    rows = entries[k]["rows"]
                    with self._lock:
                        self.stats["rows_copied"] += rows
                    touched[k] = [np.array(part.matrix[:rows]), list(part.hashes), []]
                else:
                    touched[k] = [np.empty((0, dims)), [], []]
            return touched[k]

        for h, (k, local) in live.items():
            if h in valid:  # rewritten in place: same page, same slot
                page(k)[0][local] = normalised([valid[h]], dims)[0]
            else:  # now invalid: the slot stays, unreferenced
                entries[k]["dead"].add(local)
        appended = sorted(h for h in valid if h not in live)
        tail = len(entries) - 1 if entries and entries[-1]["rows"] < size else len(entries)
        for h in appended:
            current = page(tail)
            if len(current[1]) >= size:
                tail += 1
                current = page(tail)
            current[1].append(h)
            current[2].append(valid[h])
        parts, written = [], 0
        for k in range(max(len(entries), max(touched, default=-1) + 1)):
            if k in touched:
                matrix, hashes, extra = touched[k]
                if extra:
                    matrix = np.vstack([matrix, normalised(extra, dims)])
                name = f"page-{k}-{stamp[0]}-t{token}.rvgm"
                written += self._emit(
                    directory, name, [(matrix, range(len(hashes)), hashes)], dims, token
                )
                dead = entries[k]["dead"] if k < len(entries) else set()
                parts.append(
                    {"file": name, "offset": k * size, "rows": len(hashes), "dead": dead, "page": k}
                )
            else:
                parts.append(dict(entries[k]))
        return {"parts": parts, "state": {"page_rows": size}, "moved": False, "written": written}
