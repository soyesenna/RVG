"""A tenant's document catalog: every retrieval segment in canonical key order `(id, revision,
segment)`, with the columns a query filters on, built once per catalog generation (N1F-1). A
revision that fits the embedding input is one row, the revision itself (K8-0: today's data); a
longer one is a row per segment (K8-1, `storage/segments.py`); one that cannot be segmented
without cutting is not a row and is listed in `blocked` with the reason.

The catalog moves only when documents are inserted (`document_seq`). Revocations, ACL changes and
the question's `known_at` / snapshot are masks computed per query over these columns, so they
never rebuild it. Revisions are immutable, so a parsed revision is reused by every later catalog
of its tenant; a new catalog parses only the revisions it has not seen. Canonical order is
Python's string order (code points), the same as SQLite's BINARY `ORDER BY id, revision` and
independent of any database collation.

N1F-6 (opt-in, `catalog: columnar`): `ColumnarCatalog` gives the same rows and the same answers
from fixed-width columns only — codes into one sorted table of names, revision, observed and
inserted numbers and the 32-byte input hashes — and reads a row's document from the state store
when a query needs it (revisions are immutable, so the same bytes whenever read). It is built by
streaming the bodies in key order, so a knowledge base of millions of revisions fits in a few
hundred megabytes; it is rebuilt for each document generation (no parsed reuse)."""

from __future__ import annotations

import asyncio
import threading
import time
from array import array
from collections import OrderedDict
from datetime import UTC, datetime, timedelta

import numpy as np

from ..errors import ContractError
from .identity import UnitKey, input_hash
from .segments import catalog_entries

LAYOUTS = ("resident", "columnar")
# The layout a CatalogCache gets when none is named; deployments name theirs in storage.json
# (`catalog`). Tests may switch this constant (N1F-6); nothing reads the environment.
DEFAULT_LAYOUT = "resident"
BODY_BATCH = 5000  # revisions streamed per page when a columnar catalog is built
FETCH_BATCH = 256  # keys per `bodies` call (above 256 a SQLite store reads the whole tenant)
LRU_REVISIONS = 4096  # parsed revisions a columnar catalog keeps for its queries
NAME_WIDTH = 32  # names up to this length are kept as fixed-width unicode
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_MICRO = timedelta(microseconds=1)
_EMPTY = np.empty(0, dtype=np.int64)


def micros(value: datetime) -> int:
    return (value - EPOCH) // _ONE_MICRO


def _groups(values):
    out = {}
    for row, value in enumerate(values):
        out.setdefault(value, []).append(row)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


class Catalog:
    """Immutable. `generation` is the cache key it was read for; `inserted` is -1 for revisions
    written before event numbers (always visible, their meaning before N3)."""

    def __init__(
        self, tenant, generation, documents, inserted, hashes, *, segments=None, blocked=None
    ):
        """`segments`: per row (row key, segment id, segmentation); None = every row is its
        whole revision."""
        self.tenant, self.generation = tenant, generation
        self.documents = tuple(documents)
        self.ids = np.array([d.id for d in self.documents], dtype=object)
        self.families = np.array([d.family for d in self.documents], dtype=object)
        self.observed = np.array([micros(d.observed_at) for d in self.documents], dtype=np.int64)
        self.inserted = np.asarray(inserted, dtype=np.int64)
        self.input_hashes = tuple(hashes)
        self.rows_of_id = _groups(self.ids)
        self.rows_of_family = _groups(self.families)
        segments = segments or [(d.key, "s0", None) for d in self.documents]
        self.row_keys = tuple(k for k, _, _ in segments)
        self.unit_keys = tuple(
            UnitKey(tenant, d.id, d.revision, sid, *(() if spec is None else (spec,)))
            for d, (_, sid, spec) in zip(self.documents, segments, strict=True)
        )
        self.row_of_key = {k: i for i, k in enumerate(self.row_keys)}
        self.rows_of_revision = _groups([d.key for d in self.documents])
        self.blocked = dict(blocked or {})
        self._rows_of_hash = None

    layout = "resident"

    def __len__(self):
        return len(self.documents)

    def docs(self, rows):
        """The documents of `rows`, in that order (N1F-6: the access the columnar layout batches)."""
        return [self.documents[i] for i in rows]

    def revision_starts(self):
        """The first row of each revision (rows are in key order)."""
        keys = [d.key for d in self.documents]
        return np.asarray(
            [i for i in range(len(keys)) if i == 0 or keys[i] != keys[i - 1]], dtype=np.int64
        )

    @property
    def rows_of_hash(self):
        """Input hash -> catalog rows (built on first use; incremental index links)."""
        if self._rows_of_hash is None:
            self._rows_of_hash = _groups(self.input_hashes)
        return self._rows_of_hash

    def _rows(self, identifiers):
        parts = [self.rows_of_id[i] for i in identifiers if i in self.rows_of_id]
        return np.concatenate(parts) if parts else _EMPTY

    def eligible(self, *, known_at, at=None, revoked=(), hidden=(), allowed=None, excluded=()):
        """Rows the product may rank for this question: known by `known_at`, inserted by
        knowledge snapshot `at` (if given), not revoked (the caller passes the revocations of
        that snapshot), visible to the principal, within `allowed` (which only narrows) and not
        in `excluded` (N1F-6: a deny list, e.g. an evaluation's hidden documents)."""
        mask = self.observed <= micros(known_at)
        if at is not None:
            mask &= (self.inserted < 0) | (self.inserted <= at)
        for removed in (hidden, revoked, excluded):
            if removed:
                mask[self._rows(removed)] = False
        if allowed is not None:
            keep = np.zeros(len(mask), dtype=bool)
            keep[self._rows(allowed)] = True
            mask &= keep
        return mask

    def family_rows(self, families):
        parts = [self.rows_of_family[f] for f in families if f in self.rows_of_family]
        return np.sort(np.concatenate(parts)) if parts else _EMPTY


def parse(document):
    """A revision's catalog rows `(row, row key, segment id, segmentation or None for the K8-0
    whole revision, input hash)`, or the reason it cannot be segmented."""
    try:
        entries = catalog_entries(document)
    except ContractError as exc:
        return exc.code
    return [
        (row, key, sid, None if len(entries) == 1 else spec, input_hash(text))
        for row, key, sid, spec, text in entries
    ]


# --- N1F-6: the columnar layout -----------------------------------------------------------------
class _Names:
    """One sorted table of distinct names (ids and families) in Python's string order; a name's
    code is its position. Fixed-width unicode when every name is short and has no NUL (numpy
    compares it in code-point order, as Python does), else an object array."""

    def __init__(self, names):
        unique = sorted(set(names))
        width = max((len(n) for n in unique), default=1)
        self.fixed = width <= NAME_WIDTH and not any("\0" in n for n in unique)
        self.table = (
            np.array(unique, dtype=f"<U{max(1, width)}")
            if self.fixed
            else np.array(unique + [None], dtype=object)[:-1]  # 1-d even for equal lengths
        )
        self.width = width

    def __len__(self):
        return len(self.table)

    def name(self, code):
        return str(self.table[code])

    def codes(self, names):
        """Codes of names that are in the table (a name that cannot be in it is skipped), as
        (positions in `names`, codes)."""
        names = list(names)
        keep = [
            i
            for i, n in enumerate(names)
            if isinstance(n, str) and (not self.fixed or (len(n) <= self.width and "\0" not in n))
        ]
        if not keep or not len(self.table):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        picked = [names[i] for i in keep]
        if self.fixed:
            values = np.array(picked, dtype=self.table.dtype)
        else:
            values = np.array([*picked, None], dtype=object)[:-1]
        pos = np.searchsorted(self.table, values)
        inside = pos < len(self.table)
        found = np.zeros(len(keep), dtype=bool)
        found[inside] = self.table[pos[inside]] == values[inside]
        return np.asarray(keep, dtype=np.int64)[found], pos[found].astype(np.int64)


class _Column:
    """A per-row string column read through name codes."""

    def __init__(self, names, codes):
        self._names, self._codes = names, codes

    def __len__(self):
        return len(self._codes)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            return self._names.name(self._codes[i])
        return np.asarray([self._names.name(c) for c in self._codes[i]], dtype=object)

    def __iter__(self):
        for start in range(0, len(self._codes), 65536):
            yield from self._names.table[self._codes[start : start + 65536]].tolist()

    def __array__(self, dtype=None, copy=None):
        return np.asarray(list(self), dtype=object if dtype is None else dtype)


class _HexColumn:
    """The input hashes as the resident catalog gives them (hex strings), from 32-byte rows."""

    def __init__(self, raw):
        self._raw = raw

    def __len__(self):
        return len(self._raw)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            return self._raw[i].tobytes().hex()
        return tuple(r.tobytes().hex() for r in self._raw[i])

    def __iter__(self):
        for row in self._raw:
            yield row.tobytes().hex()

    def __eq__(self, other):  # equal to the resident catalog's tuple of the same hashes
        if isinstance(other, (tuple, list, _HexColumn)):
            return tuple(self) == tuple(other)
        return NotImplemented

    __hash__ = None


def _hash_keys(raw):
    words = np.ascontiguousarray(raw, dtype=np.uint8).view(">u8").reshape(-1, 4)
    return tuple(words[:, i].astype(np.uint64) for i in range(4))


class _RowsOfHash:
    """input hash (hex) -> catalog rows in row order, like the resident dict's `get`."""

    def __init__(self, raw):
        keys = _hash_keys(raw)
        self._order = np.lexsort(keys[::-1])  # stable: equal hashes keep row order
        self._keys = tuple(k[self._order] for k in keys)

    def get(self, digest, default=None):
        if not isinstance(digest, str) or len(digest) != 64:
            return default
        try:
            q = _hash_keys(np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).reshape(1, 32))
        except ValueError:
            return default
        lo = np.searchsorted(self._keys[0], q[0][0], side="left")
        hi = np.searchsorted(self._keys[0], q[0][0], side="right")
        rows = [
            int(self._order[p])
            for p in range(int(lo), int(hi))
            if all(self._keys[c][p] == q[c][0] for c in (1, 2, 3))
        ]
        return np.asarray(rows, dtype=np.int64) if rows else default

    def __getitem__(self, digest):
        rows = self.get(digest)
        if rows is None:
            raise KeyError(digest)
        return rows

    def __contains__(self, digest):
        return self.get(digest) is not None


class _Lookup:
    """A read-only mapping computed per key (row keys, revisions, ids, families)."""

    def __init__(self, find):
        self._find = find

    def get(self, key, default=None):
        found = self._find(key)
        return default if found is None else found

    def __getitem__(self, key):
        found = self._find(key)
        if found is None:
            raise KeyError(key)
        return found

    def __contains__(self, key):
        return self._find(key) is not None


class _Rows:
    """A per-row sequence computed on access (row keys, unit keys)."""

    def __init__(self, n, item):
        self._n, self._item = n, item

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return tuple(self._item(j) for j in range(*i.indices(self._n)))
        i = int(i)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(i)
        return self._item(i)

    def __iter__(self):
        for i in range(self._n):
            yield self._item(i)

    def __eq__(self, other):  # equal to the resident catalog's tuple of the same items
        if isinstance(other, (tuple, list, _Rows)):
            return tuple(self) == tuple(other)
        return NotImplemented

    __hash__ = None


class _Documents:
    def __init__(self, catalog):
        self._catalog = catalog

    def __len__(self):
        return len(self._catalog)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return tuple(self._catalog.docs(range(*i.indices(len(self._catalog)))))
        i = int(i)
        if i < 0:
            i += len(self._catalog)
        if not 0 <= i < len(self._catalog):
            raise IndexError(i)
        return self._catalog.docs([i])[0]

    def __iter__(self):
        for start in range(0, len(self._catalog), FETCH_BATCH):
            yield from self._catalog.docs(
                range(start, min(start + FETCH_BATCH, len(self._catalog)))
            )


class ColumnarCatalog:
    """Immutable, the same rows and answers as `Catalog` from fixed-width columns; documents are
    read from the state store when asked for (N1F-6)."""

    layout = "columnar"

    def __init__(self, tenant, generation, state, columns, *, blocked=None):
        (names, id_code, family_code, revision, observed, inserted, raw, segments) = columns
        self.tenant, self.generation, self.state = tenant, generation, state
        self._names = names
        self._id = id_code
        self._family = family_code
        self.revision = revision
        self.observed, self.inserted = observed, inserted
        self.input_hash_bytes = raw
        self._segments = segments  # row -> (row key, segment id, segmentation) for K8-1 rows
        self.blocked = dict(blocked or {})
        self.ids = _Column(names, id_code)
        self.families = _Column(names, family_code)
        self.input_hashes = _HexColumn(raw)
        n = len(id_code)
        self.row_keys = _Rows(n, self._row_key)
        self.unit_keys = _Rows(n, self._unit_key)
        self.row_of_key = _Lookup(self._row_of_key)
        self.rows_of_revision = _Lookup(self._rows_of_revision)
        self.rows_of_id = _Lookup(lambda i: self._id_rows(i))
        self.rows_of_family = _Lookup(lambda f: self._family_rows_of(f))
        self.documents = _Documents(self)
        self._lock = threading.Lock()
        self._family_index = self._hash_index = None
        self._cache = OrderedDict()  # (id, revision) -> TemporalDocument or parsed rows
        self.fetched = 0  # revisions read from the state store (a measurement, not a result)

    def __len__(self):
        return len(self._id)

    @property
    def rows_of_hash(self):
        with self._lock:
            if self._hash_index is None:
                self._hash_index = _RowsOfHash(self.input_hash_bytes)
            return self._hash_index

    # --- keys -----------------------------------------------------------------------------------
    def _doc_key(self, i):
        return f"{self._names.name(self._id[i])}@{int(self.revision[i])}"

    def _row_key(self, i):
        seg = self._segments.get(i)
        return seg[0] if seg is not None else self._doc_key(i)

    def _unit_key(self, i):
        seg = self._segments.get(i)
        name, rev = self._names.name(self._id[i]), int(self.revision[i])
        if seg is None:
            return UnitKey(self.tenant, name, rev, "s0")
        return UnitKey(self.tenant, name, rev, seg[1], seg[2])

    def _id_range(self, code):
        lo = int(np.searchsorted(self._id, code, side="left"))
        hi = int(np.searchsorted(self._id, code, side="right"))
        return lo, hi

    def _id_rows(self, identifier):
        _, codes = self._names.codes([identifier])
        if not len(codes):
            return None
        lo, hi = self._id_range(codes[0])
        return np.arange(lo, hi, dtype=np.int64) if hi > lo else None

    def _rows_of_revision(self, key):
        if not isinstance(key, str) or "@" not in key:
            return None
        identifier, _, rev = key.rpartition("@")
        if not rev.isdigit():
            return None
        rows = self._id_rows(identifier)
        if rows is None:
            return None
        rows = rows[self.revision[rows] == int(rev)]
        return rows if rows.size else None

    def _row_of_key(self, key):
        rows = self._rows_of_revision(key.split("~", 1)[0]) if isinstance(key, str) else None
        if rows is None:
            return None
        for r in rows.tolist():
            if self._row_key(r) == key:
                return r
        return None

    def revision_starts(self):
        n = len(self)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        change = (self._id[1:] != self._id[:-1]) | (self.revision[1:] != self.revision[:-1])
        return np.concatenate([[0], np.flatnonzero(change) + 1]).astype(np.int64)

    # --- filters --------------------------------------------------------------------------------
    def _ranges(self, identifiers):
        """(first rows, end rows) of the ids present (an id's rows are contiguous)."""
        _, codes = self._names.codes(identifiers)
        return (
            np.searchsorted(self._id, codes, side="left"),
            np.searchsorted(self._id, codes, side="right"),
        )

    def _set(self, mask, identifiers, value):
        """Set the rows of `identifiers` in `mask` to `value`: range by range for a few ids, by
        a running count of range starts and ends for many."""
        lo, hi = self._ranges(identifiers)
        if len(lo) <= 4096:
            for a, b in zip(lo.tolist(), hi.tolist(), strict=True):
                mask[a:b] = value
            return mask
        marks = np.zeros(len(self) + 1, dtype=np.int64)
        np.add.at(marks, lo, 1)
        np.add.at(marks, hi, -1)
        inside = np.cumsum(marks[:-1]) > 0
        mask[inside] = value
        return mask

    def _mask_of_ids(self, identifiers):
        return self._set(np.zeros(len(self), dtype=bool), identifiers, True)

    def _rows(self, identifiers):
        return np.flatnonzero(self._mask_of_ids(identifiers))

    def eligible(self, *, known_at, at=None, revoked=(), hidden=(), allowed=None, excluded=()):
        """`Catalog.eligible`, the same mask."""
        mask = self.observed <= micros(known_at)
        if at is not None:
            mask &= (self.inserted < 0) | (self.inserted <= at)
        for removed in (hidden, revoked, excluded):
            if removed:
                self._set(mask, removed, False)
        if allowed is not None:
            mask &= self._mask_of_ids(allowed)
        return mask

    def _families(self):
        with self._lock:
            if self._family_index is None:
                order = np.argsort(self._family, kind="stable")
                self._family_index = (order, self._family[order])
            return self._family_index

    def _family_rows_of(self, family):
        rows = self.family_rows([family])
        return rows if rows.size else None

    def family_rows(self, families):
        _, codes = self._names.codes(families)
        if not len(codes):
            return _EMPTY
        order, sorted_codes = self._families()
        parts = [
            order[
                np.searchsorted(sorted_codes, c, "left") : np.searchsorted(sorted_codes, c, "right")
            ]
            for c in np.unique(codes)
        ]
        rows = np.concatenate(parts) if parts else _EMPTY
        return np.sort(rows) if rows.size else _EMPTY

    # --- documents ------------------------------------------------------------------------------
    def docs(self, rows):
        """The documents of `rows`, in that order: whole revisions as stored, K8-1 rows as their
        segment views; read from the state store in batches when not cached."""
        rows = [int(r) for r in rows]
        wanted = []
        with self._lock:
            for r in rows:
                key = (self._names.name(self._id[r]), int(self.revision[r]))
                if key not in self._cache and key not in wanted:
                    wanted.append(key)
        fetched = {}
        for start in range(0, len(wanted), FETCH_BATCH):
            part = wanted[start : start + FETCH_BATCH]
            got = self.state.bodies(self.tenant, part)
            missing = [k for k in part if k not in got]
            if missing:
                raise ContractError("catalog_revision_missing")
            fetched.update(got)
        out = []
        with self._lock:
            for key, doc in fetched.items():
                self._cache[key] = doc
                self.fetched += 1
            for r in rows:
                key = (self._names.name(self._id[r]), int(self.revision[r]))
                doc = self._cache.get(key)
                if doc is None:  # evicted by another thread's batch: read it again below
                    doc = self.state.bodies(self.tenant, [key])[key]
                    self._cache[key] = doc
                self._cache.move_to_end(key)
                seg = self._segments.get(r)
                if seg is None:
                    out.append(doc)
                else:
                    entries = parse(doc)
                    out.append(next(e[0] for e in entries if e[1] == seg[0]))
            while len(self._cache) > LRU_REVISIONS:
                self._cache.popitem(last=False)
        return out


def _ordered_bodies(state, tenant, rows):
    """(row, document) for every catalog row `(id, revision, inserted)`, in the given key order:
    one keyset stream when the store has `iter_bodies` (a revision inserted after the catalog
    read is skipped), else `bodies` in batches of `FETCH_BATCH`."""
    stream = getattr(state, "iter_bodies", None)
    if stream is not None:
        it = iter(rows)
        row = next(it, None)
        want = None if row is None else (row[0], row[1])
        for key, doc in stream(tenant, batch=BODY_BATCH):
            if want is None:
                return
            if key < want:
                continue  # not in the catalog read (inserted after it)
            if key > want:
                raise ContractError("catalog_revision_missing")
            yield row, doc
            row = next(it, None)
            want = None if row is None else (row[0], row[1])
        if want is not None:
            raise ContractError("catalog_revision_missing")
        return
    for start in range(0, len(rows), FETCH_BATCH):
        part = rows[start : start + FETCH_BATCH]
        got = state.bodies(tenant, [(r[0], r[1]) for r in part])
        for row in part:
            doc = got.get((row[0], row[1]))
            if doc is None:
                raise ContractError("catalog_revision_missing")
            yield row, doc


def build_columnar(state, tenant, generation, rows):
    """A ColumnarCatalog of `rows` ((id, revision, inserted or None) in key order)."""
    ids, families = [], []
    revision, observed, inserted = array("q"), array("q"), array("q")
    raw = bytearray()
    segments, blocked = {}, {}
    for (identifier, rev, seq), doc in _ordered_bodies(state, tenant, rows):
        entry = parse(doc)
        if isinstance(entry, str):
            blocked[f"{identifier}@{rev}"] = entry
            continue
        family = identifier if doc.family == identifier else doc.family
        seen = micros(doc.observed_at)
        for _row, row_key, sid, spec, digest in entry:
            if spec is not None:
                segments[len(ids)] = (row_key, sid, spec)
            ids.append(identifier)
            families.append(family)
            revision.append(rev)
            observed.append(seen)
            inserted.append(-1 if seq is None else seq)
            raw += bytes.fromhex(digest)
    names = _Names(ids + families)
    code_type = np.int32 if len(names) < 2**31 else np.int64

    def codes(values):
        out = np.empty(len(values), dtype=code_type)
        for start in range(0, len(values), 1 << 20):
            part = values[start : start + (1 << 20)]
            _, found = names.codes(part)
            if len(found) != len(part):
                raise ContractError("catalog_name_table_incomplete")
            out[start : start + len(found)] = found
        return out

    columns = (
        names,
        codes(ids),
        codes(families),
        np.frombuffer(revision, dtype=np.int64).copy(),
        np.frombuffer(observed, dtype=np.int64).copy(),
        np.frombuffer(inserted, dtype=np.int64).copy(),
        np.frombuffer(raw, dtype=np.uint8).reshape(-1, 32).copy(),
        segments,
    )
    return ColumnarCatalog(tenant, generation, state, columns, blocked=blocked)


def coverage(catalog, has_vector, *, revoked=()):
    """Per revision (every row of it has a vector): {revisions, revisions_complete, rows,
    rows_without_vector, blocked}; revisions of `revoked` ids are not counted (N1F-6)."""
    has_vector = np.asarray(has_vector, dtype=bool)
    starts = catalog.revision_starts()
    if len(catalog) == 0:
        return {
            "revisions": 0,
            "revisions_complete": 0,
            "rows": 0,
            "rows_without_vector": 0,
            "blocked": len(catalog.blocked),
        }
    keep = np.ones(len(catalog), dtype=bool)
    if revoked:
        keep[catalog._rows(revoked)] = False
    complete = np.logical_and.reduceat(has_vector | ~keep, starts)
    counted = np.logical_or.reduceat(keep, starts)
    return {
        "revisions": int(counted.sum()),
        "revisions_complete": int((complete & counted).sum()),
        "rows": int(keep.sum()),
        "rows_without_vector": int((~has_vector & keep).sum()),
        "blocked": len(catalog.blocked),
    }


def vector_mask(storage, tenant, space, catalog):
    """Per catalog row, whether the blob store holds a vector for its input (synchronous; the
    external set's rows are matched by bytes, no hex set of the whole tenant is made)."""
    blobs = storage.blobs
    fast = getattr(blobs, "vector_mask", None)
    if fast is not None:
        return fast(tenant, space, catalog)
    hashes = list(catalog.input_hashes)
    have = blobs.has_vectors(tenant, space, set(hashes))
    return np.fromiter((h in have for h in hashes), dtype=bool, count=len(hashes))


class CatalogCache:
    """Catalogs of one state store, keyed by tenant; one build per (tenant, generation) however
    many queries ask at once. `key` and `reuse_parsed` exist for the N1F-1 comparison arms only
    (`knowledge_seq` and no reuse reproduce the N1 snapshot's rebuild behaviour). `layout`:
    "resident" (`Catalog`) or "columnar" (`ColumnarCatalog`, N1F-6); None takes
    `DEFAULT_LAYOUT`."""

    def __init__(self, state, *, key="document_seq", reuse_parsed=True, layout=None):
        if key not in ("document_seq", "knowledge_seq"):
            raise ValueError("unknown_catalog_key")
        layout = DEFAULT_LAYOUT if layout is None else layout
        if layout not in LAYOUTS:
            raise ValueError("unknown_catalog_layout")
        self.state, self.key, self.reuse_parsed = state, key, reuse_parsed
        self.layout = layout
        self._entries, self._parsed, self._building = {}, {}, {}
        self._lock = threading.Lock()
        self.builds, self.parsed, self.last_seconds = 0, 0, None

    def generation(self, tenant):
        return getattr(self.state.generations(tenant), self.key)

    async def get(self, tenant, *, run=None):
        generation = self.generation(tenant)
        with self._lock:
            entry = self._entries.get(tenant)
        if entry is not None and entry.generation == generation:
            return entry
        flight = (tenant, generation)
        pending = self._building.get(flight)
        if pending is not None and pending.get_loop() is asyncio.get_running_loop():
            return await asyncio.shield(pending)
        future = self._building[flight] = asyncio.get_running_loop().create_future()
        try:
            build = lambda: self.build(tenant)
            catalog = await run(build) if run is not None else build()
        except BaseException as exc:
            future.set_exception(exc)
            future.exception()  # retrieved: a failure with no waiter is not a warning
            raise
        finally:
            if self._building.get(flight) is future:
                del self._building[flight]
        future.set_result(catalog)
        return catalog

    def build(self, tenant):
        started = time.monotonic()
        read = self.state.catalog_rows(tenant)
        rows = sorted(read.rows, key=lambda r: (r[0], r[1]))
        if self.layout == "columnar":
            catalog = build_columnar(self.state, tenant, getattr(read.generations, self.key), rows)
            return self._keep(tenant, catalog, started, len(rows))
        with self._lock:
            known = self._parsed.setdefault(tenant, {}) if self.reuse_parsed else {}
            missing = [(i, r) for i, r, _ in rows if (i, r) not in known]
        if missing:
            fresh = self.state.bodies(tenant, missing)
            parsed = {k: parse(d) for k, d in fresh.items()}
            with self._lock:
                known.update(parsed)
                self.parsed += len(parsed)
        documents, inserted, hashes, segments, blocked = [], [], [], [], {}
        for i, r, e in rows:
            entry = known[(i, r)]
            if isinstance(entry, str):
                blocked[f"{i}@{r}"] = entry
                continue
            for doc, key, sid, spec, h in entry:
                documents.append(doc)
                inserted.append(-1 if e is None else e)
                hashes.append(h)
                segments.append((key, sid, spec))
        catalog = Catalog(
            tenant,
            getattr(read.generations, self.key),
            documents,
            inserted,
            hashes,
            segments=segments,
            blocked=blocked,
        )
        return self._keep(tenant, catalog, started, 0)

    def _keep(self, tenant, catalog, started, parsed):
        with self._lock:
            self.parsed += parsed
            current = self._entries.get(tenant)
            if current is None or current.generation <= catalog.generation:
                self._entries[tenant] = catalog
            self.builds += 1
            self.last_seconds = round(time.monotonic() - started, 4)
        return catalog
