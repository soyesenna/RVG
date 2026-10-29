"""The decision cache's local store (PROD-2, opt-in, parent docs/PROD_2_PROTOCOL.md §2.1): one
SQLite file of decision entries, looked up by tenant, policy hash, arm and first-stage evidence
window. What an entry holds and how it is reused is `rvg.universal.cache`."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading


def window_key(ids):
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


def arm_key(stamp):
    return json.dumps(stamp, sort_keys=True, ensure_ascii=False) if stamp is not None else ""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class DecisionCache:
    """An opt-in local store of decision entries (SQLite; ":memory:" for tests)."""

    def __init__(self, path=":memory:", *, readonly=False):
        self.path, self.readonly = str(path), readonly
        target = f"file:{self.path}?mode=ro" if readonly else self.path
        self.conn = sqlite3.connect(target, uri=readonly, check_same_thread=False)
        self.lock = threading.Lock()
        if not readonly:
            with self.lock, self.conn:
                self.conn.execute(
                    "CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, tenant TEXT NOT NULL,"
                    " policy TEXT NOT NULL, arm TEXT NOT NULL, first_key TEXT NOT NULL,"
                    " body TEXT NOT NULL)"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS entries_lookup ON entries"
                    " (tenant, policy, arm, first_key)"
                )

    def put(self, entry):
        """Store an entry; an entry already there (same id) is kept. True when stored."""
        with self.lock, self.conn:
            cursor = self.conn.execute(
                "INSERT OR IGNORE INTO entries VALUES (?,?,?,?,?,?)",
                (
                    entry["id"],
                    entry["tenant"],
                    entry["policy"],
                    arm_key(entry["arm"]),
                    window_key(entry["first_ids"]),
                    canonical(entry),
                ),
            )
            return cursor.rowcount == 1

    def by_first(self, tenant, policy, arm, first_ids):
        with self.lock:
            rows = self.conn.execute(
                "SELECT body FROM entries WHERE tenant=? AND policy=? AND arm=? AND first_key=?"
                " ORDER BY id",
                (tenant, policy, arm_key(arm), window_key(first_ids)),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def scope(self, tenant, policy, arm):
        with self.lock:
            rows = self.conn.execute(
                "SELECT body FROM entries WHERE tenant=? AND policy=? AND arm=? ORDER BY id",
                (tenant, policy, arm_key(arm)),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def get(self, entry_id):
        with self.lock:
            row = self.conn.execute("SELECT body FROM entries WHERE id=?", (entry_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def count(self):
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def close(self):
        self.conn.close()
