"""SQLite adapter of the knowledge state and vector blob ports (the pre-PORT store, same tables).

Documents, revocations, ACL and the event ledger keep the N3 schema and meaning: the event number
is the DB-wide AUTOINCREMENT, which is commit order because SQLite serialises writers; a tenant's
knowledge snapshot is its largest event number. Added (PORT-0 / N1F-1), all by migration on open
and additive only: the event kind and target, an outbox row written in the same transaction as
each state change, indexes that make the generation counters O(log n), and canonical binary
vectors (float64 little-endian + sha256) with a per-(tenant, space) `vector_seq`. The JSON vector
table is kept and still written, so the code before this change reads the same store.

Reads that must be consistent (the release check, a catalog read) run in one read transaction
(a WAL snapshot), so writers of other processes cannot interleave with them."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..errors import ContractError
from ..grounded.store import PassageStore
from ..temporal.access import Release, hidden_ids
from ..temporal.models import TemporalDocument
from .digest import knowledge_digest, vector_json
from .identity import embedding_input, input_hash
from .ports import CatalogRows, Generations, VectorScan

MIGRATION_VECTORS = "vectors-json-to-binary/1"


class _LegacyPassageStore(PassageStore):
    """The shared SQLite file's older layer. Vector writes go through the state store, so a
    script that writes `store.base.put_vector` keeps both formats and the counters in step."""

    def __init__(self, path, owner):
        super().__init__(path)
        self._owner = owner

    def put_vector(self, tenant, space, text, vector):
        return self._owner.put_vector(tenant, space, text, vector)


def _checked_entries(entries):
    wanted = dict(entries)
    if any(e not in ("allow", "deny") for e in wanted.values()) or not all(
        isinstance(p, str) and p.strip() for p in wanted
    ):
        raise ValueError("invalid_acl_entries")
    return wanted


def canonical_vector(vector):
    value = np.asarray(vector, dtype=float)
    if value.ndim != 1 or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
        raise ContractError("invalid_passage_vector")
    return value


class SQLiteState:
    backend = "sqlite"

    def __init__(self, path):
        self.path = str(path)
        self.base = _LegacyPassageStore(path, self)
        self.conn = self.base.conn
        self.lock = self.base.lock
        self.conn.executescript("""
          CREATE TABLE IF NOT EXISTS temporal_documents(
            tenant TEXT,id TEXT,revision INTEGER,body TEXT,
            PRIMARY KEY(tenant,id,revision));
          CREATE TABLE IF NOT EXISTS temporal_revocations(
            tenant TEXT,id TEXT,PRIMARY KEY(tenant,id));
          CREATE TABLE IF NOT EXISTS temporal_events(
            seq INTEGER PRIMARY KEY AUTOINCREMENT,tenant TEXT);
          CREATE TABLE IF NOT EXISTS temporal_acl(
            tenant TEXT,id TEXT,principal TEXT,effect TEXT CHECK(effect IN ('allow','deny')),
            PRIMARY KEY(tenant,id,principal));
          CREATE TABLE IF NOT EXISTS temporal_acl_epochs(
            tenant TEXT PRIMARY KEY,epoch INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS vector_blobs(
            tenant TEXT,space TEXT,input_hash TEXT,dims INTEGER NOT NULL,dtype TEXT NOT NULL,
            data BLOB NOT NULL,sha256 TEXT NOT NULL,seq INTEGER NOT NULL,
            PRIMARY KEY(tenant,space,input_hash));
          CREATE TABLE IF NOT EXISTS storage_outbox(
            id INTEGER PRIMARY KEY AUTOINCREMENT,tenant TEXT NOT NULL,kind TEXT NOT NULL,
            knowledge_seq INTEGER,payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS storage_migrations(
            name TEXT PRIMARY KEY,applied_at TEXT NOT NULL,detail TEXT);
          CREATE TABLE IF NOT EXISTS temporal_supersessions(
            tenant TEXT NOT NULL,target_id TEXT NOT NULL,source_id TEXT NOT NULL,
            source_revision INTEGER NOT NULL,
            PRIMARY KEY(tenant,target_id,source_id,source_revision));
          CREATE TABLE IF NOT EXISTS event_ledger(
            tenant TEXT NOT NULL,issuer TEXT NOT NULL,event_id TEXT NOT NULL,
            digest TEXT NOT NULL,kind TEXT NOT NULL,result TEXT NOT NULL,knowledge_seq INTEGER,
            PRIMARY KEY(tenant,issuer,event_id));
          CREATE TABLE IF NOT EXISTS projection_frontiers(
            tenant TEXT,space TEXT,index_name TEXT,vector_seq INTEGER NOT NULL DEFAULT 0,
            document_seq INTEGER NOT NULL DEFAULT 0,token INTEGER NOT NULL DEFAULT 0,
            holder TEXT,expires REAL NOT NULL DEFAULT 0,
            PRIMARY KEY(tenant,space,index_name));
        """)
        self._migrate()

    # --- schema -------------------------------------------------------------------------------
    def _migrate(self):
        """N3: the event number of each insert and revocation (NULL before: always visible).
        PORT-0: the event kind and target (NULL for events recorded before: kind unknown, not
        guessed) and indexes. N1F-1: binary copies of JSON vectors, checked value by value."""
        with self.lock, self.conn:
            for table in ("temporal_documents", "temporal_revocations"):
                columns = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
                if "event_seq" not in columns:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN event_seq INTEGER")
            columns = {r[1] for r in self.conn.execute("PRAGMA table_info(temporal_events)")}
            for column, kind in (
                ("kind", "TEXT"),
                ("document_id", "TEXT"),
                ("revision", "INTEGER"),
            ):
                if column not in columns:
                    self.conn.execute(f"ALTER TABLE temporal_events ADD COLUMN {column} {kind}")
            self.conn.executescript("""
              CREATE INDEX IF NOT EXISTS temporal_events_tenant ON temporal_events(tenant,seq);
              CREATE INDEX IF NOT EXISTS temporal_documents_event
                ON temporal_documents(tenant,event_seq);
              CREATE INDEX IF NOT EXISTS temporal_revocations_event
                ON temporal_revocations(tenant,event_seq);
              CREATE INDEX IF NOT EXISTS vector_blobs_seq ON vector_blobs(tenant,space,seq);
            """)
        self._migrate_vectors()
        self._index_supersessions()

    def _index_supersessions(self):
        """RVI-02: the replacement declarations of every stored revision, one row per (target,
        declaring revision). Filled on every open with what is missing, so revisions written by a
        version before this one (which does not write the rows) are indexed too. Additive: a
        version before this one ignores the table."""
        with self.lock, self.conn:
            rows = self.conn.execute(
                "SELECT tenant,id,revision,body FROM temporal_documents "
                "WHERE body LIKE '%\"supersedes\":[{%'"
            ).fetchall()
            for tenant, identifier, revision, body in rows:
                for replacement in json.loads(body).get("supersedes") or []:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO temporal_supersessions VALUES(?,?,?,?)",
                        (tenant, replacement["document_id"], identifier, revision),
                    )

    def _migrate_vectors(self):
        with self.lock, self.conn:
            rows = self.conn.execute(
                "SELECT p.tenant,p.space,p.input_hash,p.vector FROM passage_vectors p "
                "WHERE NOT EXISTS (SELECT 1 FROM vector_blobs b WHERE b.tenant=p.tenant "
                "AND b.space=p.space AND b.input_hash=p.input_hash)"
            ).fetchall()
            if not rows:
                return
            seqs, mismatched, unreadable = {}, 0, 0
            for tenant, space, key, raw in rows:
                if (tenant, space) not in seqs:
                    seqs[tenant, space] = self._next_vector_seq(tenant, space)
                try:
                    value = np.array(json.loads(raw), dtype=float)
                    data = value.astype("<f8").tobytes()
                    dims = int(value.shape[0]) if value.ndim == 1 else 0
                    if json.dumps(value.tolist()) != raw:
                        mismatched += 1
                except (ValueError, TypeError):
                    data, dims = b"", 0  # kept as malformed: it fails only if it is ranked
                    unreadable += 1
                self.conn.execute(
                    "INSERT INTO vector_blobs VALUES(?,?,?,?,?,?,?,?)",
                    (
                        tenant,
                        space,
                        key,
                        dims,
                        "<f8",
                        data,
                        hashlib.sha256(data).hexdigest(),
                        seqs[tenant, space],
                    ),
                )
            self.conn.execute(
                "INSERT OR REPLACE INTO storage_migrations VALUES(?,?,?)",
                (
                    MIGRATION_VECTORS,
                    datetime.now(UTC).isoformat(),
                    json.dumps(
                        {
                            "rows": len(rows),
                            "json_roundtrip_mismatch": mismatched,
                            "unreadable": unreadable,
                        }
                    ),
                ),
            )

    def migrations(self):
        with self.lock:
            rows = self.conn.execute("SELECT * FROM storage_migrations").fetchall()
        return {r[0]: {"applied_at": r[1], **json.loads(r[2] or "{}")} for r in rows}

    def describe(self):
        return {
            "backend": self.backend,
            "engine_version": sqlite3.sqlite_version,
            "storage": "memory" if self.path == ":memory:" else "file",
            "journal": "wal",
            "event_order": "db_serialised_autoincrement",
            "release_read": "read_transaction",
            "multi_host": False,
        }

    def close(self):
        self.base.close()

    @contextmanager
    def _read(self):
        """One consistent read: a read transaction (a WAL snapshot) unless this connection is
        already inside a transaction."""
        with self.lock:
            if self.conn.in_transaction:
                yield
                return
            self.conn.execute("BEGIN")
            try:
                yield
            finally:
                self.conn.execute("COMMIT")

    def _outbox(self, tenant, kind, seq, payload):
        self.conn.execute(
            "INSERT INTO storage_outbox(tenant,kind,knowledge_seq,payload) VALUES(?,?,?,?)",
            (tenant, kind, seq, json.dumps(payload, sort_keys=True)),
        )

    def _event(self, tenant, kind, document_id, revision=None, number=None):
        """A knowledge event. Its number is the file's next one, or `number` when a catch-up
        replays another deployment's event (PORT-3): later than every event of the tenant and
        not used by another tenant of the file (event numbers are per file here)."""
        if number is None:
            return self.conn.execute(
                "INSERT INTO temporal_events(tenant,kind,document_id,revision) VALUES(?,?,?,?)",
                (tenant, kind, document_id, revision),
            ).lastrowid
        if number <= self.snapshot(tenant):
            raise ContractError("event_number_not_after_snapshot")
        try:
            self.conn.execute(
                "INSERT INTO temporal_events(seq,tenant,kind,document_id,revision) "
                "VALUES(?,?,?,?,?)",
                (number, tenant, kind, document_id, revision),
            )
        except sqlite3.IntegrityError as exc:
            raise ContractError("event_number_taken") from exc
        return number

    # --- knowledge state ----------------------------------------------------------------------
    def upsert(self, tenant, document: TemporalDocument, *, event_seq=None):
        if not tenant.strip():
            raise ValueError("tenant_required")
        with self.lock, self.conn:
            self._upsert_in(tenant, document, event_seq)
        return self.snapshot(tenant)

    def _upsert_in(self, tenant, document, event_seq=None):
        """Insert a revision inside the caller's transaction: its event number, or None when the
        same bytes are already stored. A changed body raises before anything is written."""
        body = document.model_dump_json()
        old = self.conn.execute(
            "SELECT body FROM temporal_documents WHERE tenant=? AND id=? AND revision=?",
            (tenant, document.id, document.revision),
        ).fetchone()
        if old:
            if old[0] != body:
                raise ValueError("immutable_temporal_revision_changed")
            return None
        event = self._event(tenant, "insert", document.id, document.revision, event_seq)
        self.conn.execute(
            "INSERT INTO temporal_documents(tenant,id,revision,body,event_seq) VALUES(?,?,?,?,?)",
            (tenant, document.id, document.revision, body, event),
        )
        for replacement in document.supersedes:
            self.conn.execute(
                "INSERT OR IGNORE INTO temporal_supersessions VALUES(?,?,?,?)",
                (tenant, replacement.document_id, document.id, document.revision),
            )
        self._outbox(
            tenant,
            "insert",
            event,
            {
                "id": document.id,
                "revision": document.revision,
                "input_hash": input_hash(embedding_input(document)),
            },
        )
        return event

    def revoke(self, tenant, identifier, *, event_seq=None):
        """Revoke every revision of a document. A repeated revocation changes nothing and so
        records no event (N3)."""
        with self.lock, self.conn:
            self._revoke_in(tenant, identifier, event_seq)

    def _revoke_in(self, tenant, identifier, event_seq=None):
        added = self.conn.execute(
            "INSERT OR IGNORE INTO temporal_revocations(tenant,id) VALUES(?,?)",
            (tenant, identifier),
        ).rowcount
        if not added:
            return None
        event = self._event(tenant, "revoke", identifier, None, event_seq)
        self.conn.execute(
            "UPDATE temporal_revocations SET event_seq=? WHERE tenant=? AND id=?",
            (event, tenant, identifier),
        )
        self._outbox(tenant, "revoke", event, {"id": identifier})
        return event

    def set_access(self, tenant, identifier, principal, effect):
        """An ACL entry (`allow`, `deny`, or None to remove it); every change moves the tenant's
        ACL epoch, which is not a knowledge event."""
        if effect not in {"allow", "deny", None}:
            raise ValueError("invalid_acl_effect")
        with self.lock, self.conn:
            if effect is None:
                self.conn.execute(
                    "DELETE FROM temporal_acl WHERE tenant=? AND id=? AND principal=?",
                    (tenant, identifier, principal),
                )
            else:
                self.conn.execute(
                    "INSERT OR REPLACE INTO temporal_acl VALUES(?,?,?,?)",
                    (tenant, identifier, principal, effect),
                )
            self.conn.execute(
                "INSERT INTO temporal_acl_epochs VALUES(?,1) "
                "ON CONFLICT(tenant) DO UPDATE SET epoch=epoch+1",
                (tenant,),
            )
            epoch = self.acl_epoch(tenant)
            self._outbox(
                tenant,
                "acl",
                None,
                {"id": identifier, "principal": principal, "effect": effect, "epoch": epoch},
            )

    def replace_access(self, tenant, identifier, entries):
        """Replace a document's whole ACL with `entries` (`{principal: "allow" | "deny"}`; empty:
        no entries, so the document is visible to its tenant) in one write transaction: no reader
        sees a mix of old and new entries. Each changed entry moves the ACL epoch by one and writes
        the outbox entry `set_access` writes, so a catch-up replays it unchanged; an ACL that is
        already `entries` changes nothing. Returns the number of changed entries and the epoch."""
        wanted = _checked_entries(entries)
        with self.lock, self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            changed = self._replace_access_in(tenant, identifier, wanted)
            epoch = self.acl_epoch(tenant)
        return {"changed": changed, "acl_epoch": epoch}

    def _replace_access_in(self, tenant, identifier, wanted):
        current = {
            r[0]: r[1]
            for r in self.conn.execute(
                "SELECT principal,effect FROM temporal_acl WHERE tenant=? AND id=?",
                (tenant, identifier),
            )
        }
        changes = sorted(
            (p, wanted.get(p))
            for p in set(current) | set(wanted)
            if current.get(p) != wanted.get(p)
        )
        for principal, effect in changes:
            if effect is None:
                self.conn.execute(
                    "DELETE FROM temporal_acl WHERE tenant=? AND id=? AND principal=?",
                    (tenant, identifier, principal),
                )
            else:
                self.conn.execute(
                    "INSERT OR REPLACE INTO temporal_acl VALUES(?,?,?,?)",
                    (tenant, identifier, principal, effect),
                )
            self.conn.execute(
                "INSERT INTO temporal_acl_epochs VALUES(?,1) "
                "ON CONFLICT(tenant) DO UPDATE SET epoch=epoch+1",
                (tenant,),
            )
            self._outbox(
                tenant,
                "acl",
                None,
                {
                    "id": identifier,
                    "principal": principal,
                    "effect": effect,
                    "epoch": self.acl_epoch(tenant),
                },
            )
        return len(changes)

    def commit_event(
        self,
        tenant,
        *,
        document=None,
        access=None,
        revoke=None,
        target=None,
        ledger=None,
        keep_access=True,
    ):
        """One source event published in ONE write transaction (RVI-07A): the event-id ledger
        check, the document's ACL replacement (when `keep_access` is False; `access` is the entry
        map, {} clears it; `target` names the document of an ACL-only event), the new revision or
        the revocation, their outbox rows and the ledger row. A failure anywhere leaves nothing
        written. `ledger` is `{issuer, event_id, digest, kind}`: the same key with the same digest
        returns the recorded result unchanged (`ledger: REPLAYED`); the same key with another
        digest raises `event_id_conflict`."""
        if not tenant.strip():
            raise ValueError("tenant_required")
        wanted = None if keep_access else _checked_entries(access or {})
        with self.lock, self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            if ledger is not None:
                row = self.conn.execute(
                    "SELECT digest,result FROM event_ledger WHERE tenant=? AND issuer=? "
                    "AND event_id=?",
                    (tenant, ledger["issuer"], ledger["event_id"]),
                ).fetchone()
                if row is not None:
                    if row[0] != ledger["digest"]:
                        raise ContractError("event_id_conflict")
                    return json.loads(row[1]) | {"ledger": "REPLAYED"}
            if document is not None:
                old = self.conn.execute(
                    "SELECT body FROM temporal_documents WHERE tenant=? AND id=? AND revision=?",
                    (tenant, document.id, document.revision),
                ).fetchone()
                if old and old[0] != document.model_dump_json():
                    raise ValueError("immutable_temporal_revision_changed")
            target = document.id if document is not None else (revoke or target)
            changed = None if wanted is None else self._replace_access_in(tenant, target, wanted)
            inserted = self._upsert_in(tenant, document) if document is not None else None
            revoked = self._revoke_in(tenant, revoke) if revoke is not None else None
            result = {
                "event_seq": inserted if document is not None else revoked,
                "access": None
                if changed is None
                else {"changed": changed, "acl_epoch": self.acl_epoch(tenant)},
                "changed": bool(inserted or revoked or changed),
                "knowledge_seq": self.snapshot(tenant),
            }
            if ledger is not None:
                self.conn.execute(
                    "INSERT INTO event_ledger VALUES(?,?,?,?,?,?,?)",
                    (
                        tenant,
                        ledger["issuer"],
                        ledger["event_id"],
                        ledger["digest"],
                        ledger["kind"],
                        json.dumps(result, sort_keys=True),
                        result["knowledge_seq"],
                    ),
                )
        return result | {"ledger": "RECORDED" if ledger is not None else None}

    def ledger(self, tenant):
        """The tenant's event-id ledger rows (issuer, event id, digest, kind, knowledge seq)."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT issuer,event_id,digest,kind,knowledge_seq FROM event_ledger WHERE tenant=? "
                "ORDER BY issuer,event_id",
                (tenant,),
            ).fetchall()
        return [
            dict(zip(("issuer", "event_id", "digest", "kind", "knowledge_seq"), r, strict=True))
            for r in rows
        ]

    def acl_epoch(self, tenant):
        with self.lock:
            row = self.conn.execute(
                "SELECT epoch FROM temporal_acl_epochs WHERE tenant=?", (tenant,)
            ).fetchone()
        return row[0] if row else 0

    def hidden(self, tenant, principal):
        with self.lock:
            rows = self.conn.execute(
                "SELECT id,principal,effect FROM temporal_acl WHERE tenant=?", (tenant,)
            ).fetchall()
        return hidden_ids(rows, principal)

    def revoked(self, tenant, *, at=None):
        """Revoked document ids, all of them or those revoked by event `at` (NULL = always)."""
        with self.lock:
            return {
                r[0]
                for r in self.conn.execute(
                    "SELECT id FROM temporal_revocations WHERE tenant=? "
                    "AND (? IS NULL OR event_seq IS NULL OR event_seq<=?)",
                    (tenant, at, at),
                )
            }

    def changed_families(self, tenant, since):
        """Families of documents inserted or revoked after event `since`."""
        with self.lock:
            bodies = self.conn.execute(
                "SELECT body FROM temporal_documents WHERE tenant=? AND event_seq>? "
                "UNION ALL SELECT d.body FROM temporal_documents d JOIN temporal_revocations r "
                "ON r.tenant=d.tenant AND r.id=d.id WHERE d.tenant=? AND r.event_seq>?",
                (tenant, since, tenant, since),
            ).fetchall()
        return {TemporalDocument.model_validate_json(b).family for (b,) in bodies}

    def authorize_release(self, tenant, document_ids, principal):
        """The release check (N3): in one read transaction, none of the answer's documents is
        revoked and the principal may see all of them. What it read is returned with the
        decision (`revocation_seq` is the knowledge snapshot at the fence, its N3 name; `hidden`
        the principal's hidden ids, FIX-ACL-WINDOW)."""
        with self._read():
            revoked = self.revoked(tenant) & set(document_ids)
            seen = frozenset(self.hidden(tenant, principal))
            hidden = seen & set(document_ids)
            seq = self.snapshot(tenant)
            epoch = self.acl_epoch(tenant)
        if revoked:
            return Release(
                False, "revoked_before_release", seq, epoch, tuple(sorted(revoked)), seen
            )
        if hidden:
            return Release(
                False,
                "permission_changed_before_release",
                seq,
                epoch,
                tuple(sorted(hidden)),
                seen,
            )
        return Release(True, None, seq, epoch, hidden=seen)

    def snapshot(self, tenant):
        with self.lock:
            return self.conn.execute(
                "SELECT COALESCE(MAX(seq),0) FROM temporal_events WHERE tenant=?", (tenant,)
            ).fetchone()[0]

    def generations(self, tenant):
        with self.lock:
            row = self._generations(tenant)
        return Generations(*row)

    def _generations(self, tenant):
        return self.conn.execute(
            "SELECT (SELECT COALESCE(MAX(seq),0) FROM temporal_events WHERE tenant=?),"
            "(SELECT COALESCE(MAX(event_seq),0) FROM temporal_documents WHERE tenant=?),"
            "(SELECT COALESCE(MAX(event_seq),0) FROM temporal_revocations WHERE tenant=?),"
            "(SELECT COALESCE(MAX(epoch),0) FROM temporal_acl_epochs WHERE tenant=?)",
            (tenant, tenant, tenant, tenant),
        ).fetchone()

    def catalog_rows(self, tenant):
        with self._read():
            generations = Generations(*self._generations(tenant))
            rows = self.conn.execute(
                "SELECT id,revision,event_seq FROM temporal_documents WHERE tenant=? "
                "ORDER BY id,revision",
                (tenant,),
            ).fetchall()
        return CatalogRows(generations, tuple((r[0], r[1], r[2]) for r in rows))

    def bodies(self, tenant, keys: Iterable[tuple[str, int]]):
        wanted = set(keys)
        with self.lock:
            if len(wanted) > 256:
                rows = self.conn.execute(
                    "SELECT id,revision,body FROM temporal_documents WHERE tenant=?", (tenant,)
                ).fetchall()
                rows = [r for r in rows if (r[0], r[1]) in wanted]
            else:
                rows = [
                    r
                    for i, rev in wanted
                    for r in self.conn.execute(
                        "SELECT id,revision,body FROM temporal_documents "
                        "WHERE tenant=? AND id=? AND revision=?",
                        (tenant, i, rev),
                    ).fetchall()
                ]
        return {(r[0], r[1]): TemporalDocument.model_validate_json(r[2]) for r in rows}

    def iter_bodies(self, tenant, *, batch=5000):
        """((id, revision), document) of every revision of a tenant in key order, read in keyset
        pages of `batch` rows (N1F-6: a columnar catalog streams them instead of holding all)."""
        after = None
        while True:
            with self.lock:
                if after is None:
                    rows = self.conn.execute(
                        "SELECT id,revision,body FROM temporal_documents WHERE tenant=? "
                        "ORDER BY id,revision LIMIT ?",
                        (tenant, batch),
                    ).fetchall()
                else:
                    rows = self.conn.execute(
                        "SELECT id,revision,body FROM temporal_documents WHERE tenant=? "
                        "AND (id>? OR (id=? AND revision>?)) ORDER BY id,revision LIMIT ?",
                        (tenant, after[0], after[0], after[1], batch),
                    ).fetchall()
            if not rows:
                return
            for identifier, revision, body in rows:
                yield (identifier, revision), TemporalDocument.model_validate_json(body)
            after = (rows[-1][0], rows[-1][1])

    def documents(self, tenant, *, known_at=None, allowed_ids=None, at=None):
        """Documents in store order `(id, revision)`, without revoked ones. `at` is a knowledge
        snapshot (event number): only documents inserted by then, only revocations made by then."""
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT d.body FROM temporal_documents d WHERE d.tenant=?
                AND (? IS NULL OR d.event_seq IS NULL OR d.event_seq<=?) AND NOT EXISTS
                (SELECT 1 FROM temporal_revocations r WHERE r.tenant=d.tenant AND r.id=d.id
                 AND (? IS NULL OR r.event_seq IS NULL OR r.event_seq<=?))
                ORDER BY d.id,d.revision
            """,
                (tenant, at, at, at, at),
            ).fetchall()
        docs = [TemporalDocument.model_validate_json(r[0]) for r in rows]
        return [
            d
            for d in docs
            if (known_at is None or d.observed_at <= known_at)
            and (allowed_ids is None or d.id in allowed_ids)
        ]

    def supersession_sources(self, tenant, ids, *, at=None):
        """RVI-02, the validity plane: the revisions that can change whether the revisions of
        documents `ids` are in force — every revision that declares a replacement of one of them
        and every revision of the same ids — whatever their ACL, revocation or a user's document
        selection (`at`: only those inserted by that knowledge event). Their metadata decides
        validity by code; their text is never shown to a model or a user."""
        wanted = sorted(set(ids))
        if not wanted:
            return []
        marks = ",".join("?" * len(wanted))
        with self.lock:
            rows = self.conn.execute(
                f"""
                SELECT d.body FROM temporal_supersessions s JOIN temporal_documents d
                  ON d.tenant=s.tenant AND d.id=s.source_id AND d.revision=s.source_revision
                WHERE s.tenant=? AND s.target_id IN ({marks})
                  AND (? IS NULL OR d.event_seq IS NULL OR d.event_seq<=?)
                UNION
                SELECT body FROM temporal_documents WHERE tenant=? AND id IN ({marks})
                  AND (? IS NULL OR event_seq IS NULL OR event_seq<=?)
                """,
                (tenant, *wanted, at, at, tenant, *wanted, at, at),
            ).fetchall()
        found = {}
        for (body,) in rows:
            document = TemporalDocument.model_validate_json(body)
            found[document.key] = document
        return [found[k] for k in sorted(found)]

    def events(self, tenant, since=0):
        with self.lock:
            rows = self.conn.execute(
                "SELECT seq,kind,document_id,revision FROM temporal_events "
                "WHERE tenant=? AND seq>? ORDER BY seq",
                (tenant, since),
            ).fetchall()
        return [{"seq": r[0], "kind": r[1], "document_id": r[2], "revision": r[3]} for r in rows]

    def outbox(self, tenant, since=0):
        with self.lock:
            rows = self.conn.execute(
                "SELECT id,kind,knowledge_seq,payload FROM storage_outbox "
                "WHERE tenant=? AND id>? ORDER BY id",
                (tenant, since),
            ).fetchall()
        return [
            {"id": r[0], "kind": r[1], "knowledge_seq": r[2], "payload": json.loads(r[3])}
            for r in rows
        ]

    def digest(self, tenant, space=None):
        with self._read():
            documents = self.conn.execute(
                "SELECT id,revision,body FROM temporal_documents WHERE tenant=?", (tenant,)
            ).fetchall()
            revocations = [
                r[0]
                for r in self.conn.execute(
                    "SELECT id FROM temporal_revocations WHERE tenant=?", (tenant,)
                )
            ]
            vectors = (
                None
                if space is None
                else self.conn.execute(
                    "SELECT input_hash,data FROM vector_blobs WHERE tenant=? AND space=?",
                    (tenant, space),
                ).fetchall()
            )
        return knowledge_digest(
            sorted(documents, key=lambda r: (r[0], r[1])),
            sorted(revocations),
            None if vectors is None else sorted((r[0], r[1]) for r in vectors),
        )

    def export_tenant(self, tenant):
        """Everything of one tenant another state store needs to continue it: events with their
        numbers, revision bodies byte for byte, revocations, ACL and epoch, canonical vectors of
        every space (no re-embedding). One read transaction."""
        return self.export_with_mark(tenant)[0]

    def outbox_mark(self, tenant):
        """The last outbox entry of a tenant (PORT-3: where a catch-up continues from)."""
        with self.lock:
            return self.conn.execute(
                "SELECT COALESCE(MAX(id),0) FROM storage_outbox WHERE tenant=?", (tenant,)
            ).fetchone()[0]

    def export_with_mark(self, tenant):
        """`export_tenant` and the outbox mark of the same read: every change after the export is
        an outbox entry after the mark (PORT-3)."""
        with self._read():
            mark = self.conn.execute(
                "SELECT COALESCE(MAX(id),0) FROM storage_outbox WHERE tenant=?", (tenant,)
            ).fetchone()[0]
            events = self.conn.execute(
                "SELECT seq,kind,document_id,revision FROM temporal_events WHERE tenant=? "
                "ORDER BY seq",
                (tenant,),
            ).fetchall()
            documents = self.conn.execute(
                "SELECT id,revision,body,event_seq FROM temporal_documents WHERE tenant=?",
                (tenant,),
            ).fetchall()
            revocations = self.conn.execute(
                "SELECT id,event_seq FROM temporal_revocations WHERE tenant=?", (tenant,)
            ).fetchall()
            acl = self.conn.execute(
                "SELECT id,principal,effect FROM temporal_acl WHERE tenant=?", (tenant,)
            ).fetchall()
            epoch = self.acl_epoch(tenant)
            vectors = self.conn.execute(
                "SELECT space,input_hash,dims,data,sha256,seq FROM vector_blobs WHERE tenant=?",
                (tenant,),
            ).fetchall()
        export = {
            "events": [
                {"seq": r[0], "kind": r[1], "document_id": r[2], "revision": r[3]} for r in events
            ],
            "documents": [
                {
                    "id": r[0],
                    "revision": r[1],
                    "body": r[2],
                    "event_seq": r[3],
                    "family": json.loads(r[2])["family"],
                }
                for r in documents
            ],
            "revocations": [{"id": r[0], "event_seq": r[1]} for r in revocations],
            "acl": [{"id": r[0], "principal": r[1], "effect": r[2]} for r in acl],
            "acl_epoch": epoch,
            "vectors": [
                {
                    "space": r[0],
                    "input_hash": r[1],
                    "dims": r[2],
                    "data": bytes(r[3]),
                    "sha256": r[4],
                    "seq": r[5],
                }
                for r in vectors
            ],
        }
        return export, mark

    def import_tenant(self, tenant, *, events, documents, revocations, acl, acl_epoch, vectors):
        """Load one tenant exported from another state store, keeping its event numbers and
        vector bytes, in one transaction; only into a tenant this file has never held."""
        with self.lock, self.conn:
            for table in (
                "temporal_events",
                "temporal_documents",
                "temporal_revocations",
                "temporal_acl",
                "temporal_acl_epochs",
                "vector_blobs",
            ):
                if self.conn.execute(
                    f"SELECT 1 FROM {table} WHERE tenant=? LIMIT 1", (tenant,)
                ).fetchone():
                    raise ValueError("import_target_tenant_not_empty")
            try:
                self.conn.executemany(
                    "INSERT INTO temporal_events(seq,tenant,kind,document_id,revision) "
                    "VALUES(?,?,?,?,?)",
                    [
                        (e["seq"], tenant, e["kind"], e["document_id"], e["revision"])
                        for e in events
                    ],
                )
            except sqlite3.IntegrityError as exc:
                raise ContractError("import_event_numbers_taken") from exc
            self.conn.executemany(
                "INSERT INTO temporal_documents(tenant,id,revision,body,event_seq) "
                "VALUES(?,?,?,?,?)",
                [(tenant, d["id"], d["revision"], d["body"], d["event_seq"]) for d in documents],
            )
            for d in documents:
                for replacement in json.loads(d["body"]).get("supersedes") or []:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO temporal_supersessions VALUES(?,?,?,?)",
                        (tenant, replacement["document_id"], d["id"], d["revision"]),
                    )
            self.conn.executemany(
                "INSERT INTO temporal_revocations(tenant,id,event_seq) VALUES(?,?,?)",
                [(tenant, r["id"], r["event_seq"]) for r in revocations],
            )
            self.conn.executemany(
                "INSERT INTO temporal_acl VALUES(?,?,?,?)",
                [(tenant, a["id"], a["principal"], a["effect"]) for a in acl],
            )
            if acl_epoch:
                self.conn.execute(
                    "INSERT INTO temporal_acl_epochs VALUES(?,?)", (tenant, acl_epoch)
                )
            for v in vectors:
                if hashlib.sha256(v["data"]).hexdigest() != v["sha256"]:
                    raise ContractError("vector_blob_corrupted")
                self.conn.execute(
                    "INSERT INTO vector_blobs VALUES(?,?,?,?,?,?,?,?)",
                    (
                        tenant,
                        v["space"],
                        v["input_hash"],
                        v["dims"],
                        "<f8",
                        v["data"],
                        v["sha256"],
                        v["seq"],
                    ),
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO passage_vectors VALUES(?,?,?,?)",
                    (tenant, v["space"], v["input_hash"], vector_json(v["data"])),
                )

    # --- vector blobs -------------------------------------------------------------------------
    def _next_vector_seq(self, tenant, space):
        return (
            self.conn.execute(
                "SELECT COALESCE(MAX(seq),0) FROM vector_blobs WHERE tenant=? AND space=?",
                (tenant, space),
            ).fetchone()[0]
            + 1
        )

    def put_vector(self, tenant, space, text, vector):
        """Store a canonical vector. Writing the same bytes again changes nothing (no new
        generation); different bytes replace the value — vectors are a function of (space,
        input), so no history is kept (declared: `vector_history: not_retained`)."""
        return self.put_vector_blob(tenant, space, input_hash(text), vector)

    def put_vector_blob(self, tenant, space, key, vector):
        """`put_vector` by input hash: a catch-up copies vectors without their input text."""
        value = canonical_vector(vector)
        data = value.astype("<f8").tobytes()
        with self.lock, self.conn:
            old = self.conn.execute(
                "SELECT data FROM vector_blobs WHERE tenant=? AND space=? AND input_hash=?",
                (tenant, space, key),
            ).fetchone()
            if old is not None and old[0] == data:
                return False
            seq = self._next_vector_seq(tenant, space)
            self.conn.execute(
                "INSERT OR REPLACE INTO vector_blobs VALUES(?,?,?,?,?,?,?,?)",
                (
                    tenant,
                    space,
                    key,
                    int(value.shape[0]),
                    "<f8",
                    data,
                    hashlib.sha256(data).hexdigest(),
                    seq,
                ),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO passage_vectors VALUES(?,?,?,?)",
                (tenant, space, key, vector_json(data)),
            )
            self._outbox(tenant, "vector", None, {"space": space, "input_hash": key, "seq": seq})
        return True

    def vector(self, tenant, space, text):
        with self.lock:
            row = self.conn.execute(
                "SELECT data,sha256 FROM vector_blobs WHERE tenant=? AND space=? AND input_hash=?",
                (tenant, space, input_hash(text)),
            ).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[0]).hexdigest() != row[1]:
            raise ContractError("vector_blob_corrupted")
        return np.frombuffer(row[0], dtype="<f8").astype(np.float64)

    def has_vectors(self, tenant, space, hashes):
        hashes, out = list(hashes), set()
        for start in range(0, len(hashes), 500):
            part = hashes[start : start + 500]
            with self.lock:
                found = self.conn.execute(
                    "SELECT input_hash FROM vector_blobs WHERE tenant=? AND space=? "
                    f"AND input_hash IN ({','.join('?' * len(part))})",
                    (tenant, space, *part),
                ).fetchall()
            out.update(r[0] for r in found)
        return out

    def vectors(self, tenant, space, hashes):
        """input hash -> canonical float64 vector, for the hashes that have one (checksummed)."""
        hashes, out = list(hashes), {}
        for start in range(0, len(hashes), 500):
            part = hashes[start : start + 500]
            with self.lock:
                rows = self.conn.execute(
                    "SELECT input_hash,data,sha256 FROM vector_blobs WHERE tenant=? AND space=? "
                    f"AND input_hash IN ({','.join('?' * len(part))})",
                    (tenant, space, *part),
                ).fetchall()
            for h, data, digest in rows:
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ContractError("vector_blob_corrupted")
                out[h] = np.frombuffer(data, dtype="<f8").astype(np.float64)
        return out

    def vector_checksums(self, tenant, space):
        """(input hash, sha256 of the bytes) of every vector of a space, sorted — a cheap
        comparison between deployments; each read still checks bytes against the sha256."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT input_hash,sha256 FROM vector_blobs WHERE tenant=? AND space=?",
                (tenant, space),
            ).fetchall()
        return sorted((r[0], r[1]) for r in rows)

    def vector_meta(self, tenant, space):
        """input hash -> (dims, vector_seq of its current bytes), for every vector of a space."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT input_hash,dims,seq FROM vector_blobs WHERE tenant=? AND space=?",
                (tenant, space),
            ).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    # --- projection frontier (PORT-2) ---------------------------------------------------------
    def projection_frontier(self, tenant, space, index):
        with self.lock:
            row = self.conn.execute(
                "SELECT vector_seq,document_seq,token FROM projection_frontiers "
                "WHERE tenant=? AND space=? AND index_name=?",
                (tenant, space, index),
            ).fetchone()
        return (
            {"vector_seq": row[0], "document_seq": row[1], "token": row[2]}
            if row
            else {"vector_seq": 0, "document_seq": 0, "token": 0}
        )

    def projection_lease(self, tenant, space, index, holder, *, now, ttl):
        """A projector's lease: a new fencing token when the lease is free, expired or already
        this holder's; None while another holder's lease is live."""
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO projection_frontiers(tenant,space,index_name) VALUES(?,?,?)",
                (tenant, space, index),
            )
            token, owner, expires = self.conn.execute(
                "SELECT token,holder,expires FROM projection_frontiers "
                "WHERE tenant=? AND space=? AND index_name=?",
                (tenant, space, index),
            ).fetchone()
            if owner not in (None, holder) and expires > now:
                return None
            self.conn.execute(
                "UPDATE projection_frontiers SET token=?,holder=?,expires=? "
                "WHERE tenant=? AND space=? AND index_name=?",
                (token + 1, holder, now + ttl, tenant, space, index),
            )
            return token + 1

    def advance_projection(self, tenant, space, index, token, vector_seq, document_seq):
        """Move the frontier forward, only under the current fencing token and never back."""
        with self.lock, self.conn:
            moved = self.conn.execute(
                "UPDATE projection_frontiers SET vector_seq=MAX(vector_seq,?),"
                "document_seq=MAX(document_seq,?) "
                "WHERE tenant=? AND space=? AND index_name=? AND token=?",
                (vector_seq, document_seq, tenant, space, index, token),
            ).rowcount
        return bool(moved)

    def vector_seq(self, tenant, space):
        with self.lock:
            return self.conn.execute(
                "SELECT COALESCE(MAX(seq),0) FROM vector_blobs WHERE tenant=? AND space=?",
                (tenant, space),
            ).fetchone()[0]

    @contextmanager
    def scan_changes(self, tenant, space, since):
        """Vectors of (tenant, space) written after `vector_seq` `since` — the latest bytes of
        each, in (seq, input hash) order — with the `vector_seq` of the same read (N1F-2)."""
        query = (
            "SELECT input_hash,dims,data,seq FROM vector_blobs WHERE tenant=? AND space=? "
            "AND seq>? ORDER BY seq,input_hash"
        )
        seq_query = "SELECT COALESCE(MAX(seq),0) FROM vector_blobs WHERE tenant=? AND space=?"
        with self._read():
            seq = self.conn.execute(seq_query, (tenant, space)).fetchone()[0]
            rows = self.conn.execute(query, (tenant, space, since)).fetchall()
        yield VectorScan(seq, len(rows), iter([(r[0], r[1], r[2], r[3]) for r in rows]))

    @contextmanager
    def scan_vectors(self, tenant, space, *, source="binary"):
        """Every vector of (tenant, space) in input-hash order with the `vector_seq` of the same
        read. A file store is read through its own read-only connection (a WAL snapshot), so a
        build in a worker thread neither holds the store lock nor sees half a write."""
        if source == "binary":
            query = (
                "SELECT input_hash,dims,data FROM vector_blobs WHERE tenant=? AND space=? "
                "ORDER BY input_hash"
            )
            count_query = "SELECT COUNT(*) FROM vector_blobs WHERE tenant=? AND space=?"
        elif source == "json":
            query = (
                "SELECT input_hash,NULL,vector FROM passage_vectors WHERE tenant=? AND space=? "
                "ORDER BY input_hash"
            )
            count_query = "SELECT COUNT(*) FROM passage_vectors WHERE tenant=? AND space=?"
        else:
            raise ValueError("unknown_vector_source")
        seq_query = "SELECT COALESCE(MAX(seq),0) FROM vector_blobs WHERE tenant=? AND space=?"
        if self.path == ":memory:":
            with self._read():
                seq = self.conn.execute(seq_query, (tenant, space)).fetchone()[0]
                rows = self.conn.execute(query, (tenant, space)).fetchall()
            yield VectorScan(seq, len(rows), iter(rows), source)
            return
        conn = sqlite3.connect(Path(self.path).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            conn.execute("BEGIN")
            seq = conn.execute(seq_query, (tenant, space)).fetchone()[0]
            count = conn.execute(count_query, (tenant, space)).fetchone()[0]
            yield VectorScan(seq, count, conn.execute(query, (tenant, space)), source)
            conn.execute("COMMIT")
        finally:
            conn.close()
