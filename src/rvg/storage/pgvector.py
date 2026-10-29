"""pgvector projection backend (PORT-2): points in a PostgreSQL table with a `vector(d)` column
(float32), ranked by cosine distance over a sequential scan — no ANN index, so the ranking is
exact over the stored float32 vectors (`EXACT_NATIVE`). The plan of the search query is recorded
in `describe()` so a run that silently used an index is visible.

A point's write never replaces a newer one: `ON CONFLICT ... WHERE blob_seq <= EXCLUDED.blob_seq`,
so a late or repeated projection cannot bring back older bytes."""

from __future__ import annotations

import re

import numpy as np

from ..errors import ConfigurationError
from ..retrieval.data import canonical_hash

_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _literal(vector):
    return "[" + ",".join(repr(float(x)) for x in np.asarray(vector, dtype=np.float64)) + "]"


class PgvectorBackend:
    name = "pgvector"

    def __init__(self, dsn, *, schema="rvg_index"):
        import psycopg

        if not _NAME.match(schema):
            raise ConfigurationError("invalid_postgres_schema_name")
        self._psycopg, self.schema = psycopg, schema
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        # The `vector` type lives in the extension's own schema, wherever it was installed.
        home = self.conn.execute(
            "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid "
            "WHERE e.extname = 'vector'"
        ).fetchone()[0]
        self.conn.execute(
            "SELECT set_config('search_path', %s, false)",
            (",".join(dict.fromkeys((schema, home))),),
        )
        self.server = self.conn.execute("SHOW server_version").fetchone()[0]
        self.extension = self.conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname='vector'"
        ).fetchone()[0]
        self._dims, self.plan = {}, None

    def table(self, space):
        return "points_" + canonical_hash(space)[:12]

    def describe(self):
        return {
            "backend": "pgvector",
            "server_version": self.server,
            "extension_version": self.extension,
            "column": "vector(d) float32",
            "operator": "<=> cosine distance, score = 1 - distance",
            "index": "none (sequential scan, exact)",
            "conditional_write": "ON CONFLICT WHERE blob_seq <= new",
            "search_plan": self.plan,
        }

    def ensure(self, space, dims):
        if self._dims.get(space) == dims:
            return
        t = self.table(space)
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {t}(point_id uuid PRIMARY KEY, tenant text NOT NULL, "
            "document_id text NOT NULL, revision integer NOT NULL, input_hash text NOT NULL, "
            "observed_us bigint NOT NULL, inserted bigint NOT NULL, blob_seq bigint NOT NULL, "
            f"embedding vector({int(dims)}) NOT NULL)"
        )
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_tenant ON {t}(tenant)")
        self._dims[space] = dims

    def upsert(self, space, points):
        """points: (point_id, float64 vector, payload with tenant, document_id, revision,
        input_hash, observed_us, inserted, blob_seq)."""
        t = self.table(space)
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {t} VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::vector) "
                "ON CONFLICT(point_id) DO UPDATE SET tenant=EXCLUDED.tenant, "
                "document_id=EXCLUDED.document_id, revision=EXCLUDED.revision, "
                "input_hash=EXCLUDED.input_hash, observed_us=EXCLUDED.observed_us, "
                "inserted=EXCLUDED.inserted, blob_seq=EXCLUDED.blob_seq, "
                f"embedding=EXCLUDED.embedding WHERE {t}.blob_seq <= EXCLUDED.blob_seq",
                [
                    (
                        point,
                        p["tenant"],
                        p["document_id"],
                        p["revision"],
                        p["input_hash"],
                        p["observed_us"],
                        p["inserted"],
                        p["blob_seq"],
                        _literal(vector),
                    )
                    for point, vector, p in points
                ],
            )

    def delete(self, space, point_ids):
        if point_ids:
            self.conn.execute(
                f"DELETE FROM {self.table(space)} WHERE point_id = ANY(%s::uuid[])",
                (list(point_ids),),
            )

    def count(self, tenant, space):
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {self.table(space)} WHERE tenant=%s", (tenant,)
        ).fetchone()[0]

    def search(self, tenant, space, query, limit, *, include=None, exclude=None):
        t = self.table(space)
        where, params = ["tenant=%s"], [tenant]
        if include is not None:
            where.append("point_id = ANY(%s::uuid[])")
            params.append(include)
        if exclude:
            where.append("NOT (point_id = ANY(%s::uuid[]))")
            params.append(exclude)
        q = _literal(query)
        sql = (
            f"SELECT point_id::text, 1 - (embedding <=> %s::vector) AS score, blob_seq FROM {t} "
            f"WHERE {' AND '.join(where)} ORDER BY embedding <=> %s::vector, point_id LIMIT %s"
        )
        args = [q, *params, q, int(limit)]
        if self.plan is None:
            plan = self.conn.execute("EXPLAIN " + sql, args).fetchall()
            self.plan = [r[0] for r in plan]
        return [
            (point, float(score), int(seq))
            for point, score, seq in self.conn.execute(sql, args).fetchall()
        ]

    def close(self):
        self.conn.close()
