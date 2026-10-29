"""The passage store the SQLite state adapter is built on. The legacy grounded retriever stayed in
the research repository."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import RLock

import numpy as np

from ..errors import ContractError
from ..retrieval.data import canonical_hash
from ..retrieval.store import split_spans


class PassageStore:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = RLock()
        self.conn.executescript("""
          PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS source_versions(seq INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant TEXT, id TEXT, revision INTEGER, title TEXT, text TEXT, sha TEXT, deleted INTEGER,
            UNIQUE(tenant,id,revision));
          CREATE TABLE IF NOT EXISTS source_revocations(tenant TEXT,id TEXT,PRIMARY KEY(tenant,id));
          CREATE TABLE IF NOT EXISTS passage_vectors(tenant TEXT,space TEXT,input_hash TEXT,vector TEXT,
            PRIMARY KEY(tenant,space,input_hash));
          CREATE TABLE IF NOT EXISTS answer_cache(tenant TEXT,key TEXT,body TEXT,lineage TEXT,
            PRIMARY KEY(tenant,key));
        """)

    def close(self):
        self.conn.close()

    def upsert(
        self, tenant: str, identifier: str, revision: int, title: str, text: str, *, deleted=False
    ):
        if (
            not tenant.strip()
            or not identifier.strip()
            or type(revision) is not int
            or revision < 1
            or len(text) > 200_000
            or (not deleted and not text.strip())
        ):
            raise ValueError("invalid_source_document")
        digest = hashlib.sha256(text.encode()).hexdigest()
        with self.lock, self.conn:
            old = self.conn.execute(
                "SELECT * FROM source_versions WHERE tenant=? AND id=? AND revision=?",
                (tenant, identifier, revision),
            ).fetchone()
            if old:
                if (old["sha"], old["title"], bool(old["deleted"])) != (digest, title, deleted):
                    raise ValueError("immutable_revision_changed")
                return old["seq"]
            cursor = self.conn.execute(
                "INSERT INTO source_versions(tenant,id,revision,title,text,sha,deleted) VALUES(?,?,?,?,?,?,?)",
                (tenant, identifier, revision, title, text, digest, int(deleted)),
            )
            self.conn.execute("DELETE FROM answer_cache WHERE tenant=?", (tenant,))
            return cursor.lastrowid

    def revoke(self, tenant, identifier):
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO source_revocations VALUES(?,?)", (tenant, identifier)
            )
            self.conn.execute("DELETE FROM answer_cache WHERE tenant=?", (tenant,))

    def snapshot(self, tenant):
        return self.conn.execute(
            "SELECT COALESCE(MAX(seq),0) FROM source_versions WHERE tenant=?", (tenant,)
        ).fetchone()[0]

    def documents(self, tenant, snapshot=None):
        snapshot = self.snapshot(tenant) if snapshot is None else snapshot
        return [
            dict(r)
            for r in self.conn.execute(
                """
          WITH latest AS (SELECT id,MAX(revision) revision FROM source_versions WHERE tenant=? AND seq<=? GROUP BY id)
          SELECT d.* FROM source_versions d JOIN latest l ON d.id=l.id AND d.revision=l.revision
          WHERE d.tenant=? AND d.deleted=0 AND NOT EXISTS
          (SELECT 1 FROM source_revocations r WHERE r.tenant=d.tenant AND r.id=d.id) ORDER BY d.id
        """,
                (tenant, snapshot, tenant),
            )
        ]

    def passages(self, tenant, snapshot=None):
        rows = []
        for doc in self.documents(tenant, snapshot):
            for start, end in split_spans(doc["text"], 1200, overlap=96):
                identifier = (
                    "p-" + canonical_hash([tenant, doc["id"], doc["revision"], start, end])[:24]
                )
                rows.append(
                    {
                        "id": identifier,
                        "document_id": doc["id"],
                        "revision": doc["revision"],
                        "title": doc["title"],
                        "text": doc["text"][start:end],
                        "start": start,
                        "end": end,
                        "source_sha256": doc["sha"],
                    }
                )
        return rows

    def current(self, tenant, passage):
        rows = self.conn.execute(
            "SELECT revision,sha,title,text,deleted FROM source_versions WHERE tenant=? AND id=? ORDER BY revision DESC LIMIT 1",
            (tenant, passage["document_id"]),
        ).fetchone()
        revoked = self.conn.execute(
            "SELECT 1 FROM source_revocations WHERE tenant=? AND id=?",
            (tenant, passage["document_id"]),
        ).fetchone()
        metadata_matches = bool(
            rows
            and not revoked
            and not rows["deleted"]
            and rows["revision"] == passage["revision"]
            and rows["sha"] == passage["source_sha256"]
        )
        if not metadata_matches:
            return False
        if "title" in passage and passage["title"] != rows["title"]:
            return False
        snippet = passage.get("text", passage.get("quote"))
        start, end = passage.get("start"), passage.get("end")
        return (
            isinstance(snippet, str)
            and type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(rows["text"])
            and rows["text"][start:end] == snippet
            and hashlib.sha256(rows["text"].encode()).hexdigest() == rows["sha"]
        )

    def put_vector(self, tenant, space, text, vector):
        value = np.asarray(vector, dtype=float)
        if value.ndim != 1 or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
            raise ContractError("invalid_passage_vector")
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO passage_vectors VALUES(?,?,?,?)",
                (tenant, space, canonical_hash(text), json.dumps(value.tolist())),
            )

    def vector(self, tenant, space, text):
        row = self.conn.execute(
            "SELECT vector FROM passage_vectors WHERE tenant=? AND space=? AND input_hash=?",
            (tenant, space, canonical_hash(text)),
        ).fetchone()
        return np.array(json.loads(row[0]), dtype=float) if row else None

    def cache_get(self, tenant, key):
        row = self.conn.execute(
            "SELECT body,lineage FROM answer_cache WHERE tenant=? AND key=?", (tenant, key)
        ).fetchone()
        if row and all(self.current(tenant, p) for p in json.loads(row["lineage"])):
            return json.loads(row["body"])
        return None

    def cache_put(self, tenant, key, body, lineage):
        with self.lock, self.conn:
            if (
                body.get("status") != "OK"
                or not lineage
                or not all(self.current(tenant, p) for p in lineage)
            ):
                return False
            self.conn.execute(
                "INSERT OR REPLACE INTO answer_cache VALUES(?,?,?,?)",
                (tenant, key, json.dumps(body, ensure_ascii=False), json.dumps(lineage)),
            )
            return True
