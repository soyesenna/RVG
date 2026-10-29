"""PORT-0 / N1F-1: the storage ports keep the product path free of any backend, the SQLite
adapter keeps the N3 store's meaning, and only a real vector change rebuilds the dense matrix."""

import ast
import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from rvg.contracts import EmbeddingBatch, EmbeddingSpace
from rvg.errors import ConfigurationError, ContractError
from rvg.grounded.store import PassageStore
from rvg.runtime import CallContext
from rvg.storage import load_config, open_storage
from rvg.storage.identity import (
    SEGMENTATION,
    UnitKey,
    VectorKey,
    embedding_input,
    input_hash,
)
from rvg.storage.ports import KnowledgeStateStore, VectorBlobStore
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore

ROOT = Path(__file__).resolve().parents[1]
DIMS = 4
SPACE = EmbeddingSpace(
    provider="fixture",
    model="fixture",
    base_url="fixture://",
    dimensions=DIMS,
    normalize=False,
    similarity_metric="cosine",
    encoding_format="float",
    query_prefix="",
    document_prefix="",
    query_input_type=None,
    document_input_type=None,
    version=1,
)
FP = SPACE.fingerprint

# The engine, the retriever and the backend-independent storage modules: none of them may reach
# a database driver, a connection, a row id or the SQLite-era store object.
CORE = [
    *sorted((ROOT / "src/rvg/universal").glob("*.py")),
    ROOT / "src/rvg/temporal/retriever.py",
    ROOT / "src/rvg/temporal/gate.py",
    *(
        ROOT / "src/rvg/storage" / name
        for name in (
            "ports.py",
            "catalog.py",
            "local_exact.py",
            "release.py",
            "identity.py",
            "digest.py",
            "factory.py",
            "incremental.py",
            "remote.py",
            "projection.py",
        )
    ),
]
ADAPTERS = ("sqlite", "postgres", "pgvector", "qdrant")
DRIVERS = ("sqlite3", "psycopg", "qdrant_client", "pgvector")


def doc(identifier, *, revision=1, day=1, family=None, title=None, text=None):
    text = text or f"Fact {identifier} rev {revision}."
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family or identifier,
            "revision": revision,
            "title": title or f"T {identifier}",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": f"2026-09-{day:02d}T00:00:00Z",
            "observed_at": f"2026-09-{day:02d}T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
        }
    )


class Embedder:
    mode = "fixture"

    def __init__(self, vectors=None):
        self.vectors, self.calls = vectors or {}, 0

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        self.calls += 1
        rows = tuple(
            tuple(self.vectors.get(t, np.arange(1, DIMS + 1) / (1 + len(t) % 5))) for t in texts
        )
        return EmbeddingBatch(rows, SPACE, ("f",), ())


def question(known="2026-09-30T00:00:00Z"):
    return TemporalQuestion(text="q", as_of="2026-09-30T00:00:00Z", known_at=known)


def search(store, emb, **kw):
    retriever = TemporalRetriever(store, emb, max_documents=64, **kw)
    return asyncio.run(
        retriever.search(
            "t", question(), CallContext.with_timeout(30, max_embedding_calls=8), top_k=5
        )
    )


# --- boundaries --------------------------------------------------------------------------------
def test_core_modules_never_reach_a_backend():
    offenders = []
    for path in CORE:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = {id(n) for n in tree.body}
        docstrings = {
            id(n.body[0].value)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                relative = isinstance(node, ast.ImportFrom) and node.level > 0
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""] + [a.name for a in node.names if not node.module]
                )
                for name in names:
                    # `pgvector` is both a driver package and our adapter module: a relative
                    # import names the adapter, an absolute one the driver.
                    if not relative and any(name.split(".")[0] == d for d in DRIVERS):
                        offenders.append(f"{path.name}: import {name}")
                    # Only the composition root (factory) picks an adapter, and only lazily.
                    adapter = name.split(".")[-1] in ADAPTERS and (
                        relative or name.startswith("rvg.storage.")
                    )
                    if adapter and (path.name != "factory.py" or id(node) in top):
                        offenders.append(f"{path.name}: adapter import {name}")
            if isinstance(node, ast.Attribute) and node.attr in ("conn", "cursor", "base"):
                offenders.append(f"{path.name}: .{node.attr}")
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and "rowid" in node.value.lower()
            ):
                offenders.append(f"{path.name}: rowid")
    assert offenders == []


def test_the_sqlite_store_implements_both_ports():
    store = TemporalStore(":memory:")
    assert isinstance(store, KnowledgeStateStore) and isinstance(store, VectorBlobStore)
    described = store.storage.describe()
    assert described["state"] == "sqlite" and described["precision"] == "EXACT_CANONICAL"
    assert len(described["fingerprint"]) == 64


def test_vector_keys_are_content_addressed_and_backend_ids_are_derived():
    a = VectorKey("t", FP, input_hash("x"))
    assert a.point_id() == VectorKey("t", FP, input_hash("x")).point_id()
    others = {
        VectorKey("u", FP, input_hash("x")),
        VectorKey("t", "other", input_hash("x")),
        VectorKey("t", FP, input_hash("y")),
    }
    assert a.point_id() not in {k.point_id() for k in others}
    d = doc("a")
    assert embedding_input(d) == TemporalRetriever.input_text(d) == d.title + "\n" + d.text
    one, two = UnitKey("t", "a", 1), UnitKey("t", "a", 2)
    assert one.revision_key == "a@1" and one.segmentation == SEGMENTATION
    assert one.point_id(FP) != two.point_id(FP)  # two revisions are never one point
    assert one.point_id(FP) != one.point_id("other") and one.point_id(FP) == UnitKey(
        "t", "a", 1
    ).point_id(FP)


# --- deployment configuration ------------------------------------------------------------------
def test_storage_configuration_names_backends_and_secret_references_only(tmp_path):
    config = load_config(ROOT / "examples/storage/sqlite-local.json")
    assert config.state.backend == "sqlite" and len(config.fingerprint()) == 64
    with pytest.raises(ValidationError):  # a connection string is never written in the file
        load_config(
            {"schema_version": 1, "state": {"backend": "postgres", "dsn": "postgresql://u:p@h/db"}}
        )
    with pytest.raises(ValidationError):  # nor a per-knowledge-base section
        load_config(
            {"schema_version": 1, "state": {"backend": "sqlite", "path": "x"}, "kb": {"a": {}}}
        )
    with pytest.raises(ConfigurationError, match="postgres_dsn"):
        open_storage(ROOT / "examples/storage/postgres-local.json", root=tmp_path, environ={})
    storage = open_storage(
        {"schema_version": 1, "state": {"backend": "sqlite", "path": "s.sqlite3"}}, root=tmp_path
    )
    assert storage.index.matrix_dir == tmp_path / "s.sqlite3.dense"
    assert (
        storage.describe()["config_hash"]
        == load_config(
            {"schema_version": 1, "state": {"backend": "sqlite", "path": "s.sqlite3"}}
        ).fingerprint()
    )
    storage.close()


# --- SQLite adapter: migration and vectors -----------------------------------------------------
def legacy_digest(path, tenant, space):
    """`evaluation.knowledge.store_digest` before PORT-0, verbatim over SQL."""
    conn = sqlite3.connect(path)
    h = hashlib.sha256()
    for row in conn.execute(
        "SELECT id,revision,body FROM temporal_documents WHERE tenant=? ORDER BY id,revision",
        (tenant,),
    ):
        h.update(f"d|{row[0]}|{row[1]}|".encode())
        h.update(hashlib.sha256(row[2].encode()).digest())
    for row in conn.execute(
        "SELECT id FROM temporal_revocations WHERE tenant=? ORDER BY id", (tenant,)
    ):
        h.update(f"r|{row[0]}".encode())
    for row in conn.execute(
        "SELECT input_hash,vector FROM passage_vectors WHERE tenant=? AND space=? "
        "ORDER BY input_hash",
        (tenant, space),
    ):
        h.update(f"v|{row[0]}|".encode())
        h.update(hashlib.sha256(row[1].encode()).digest())
    conn.close()
    return h.hexdigest()


def test_json_vectors_migrate_to_binary_value_for_value_and_keep_the_digest(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    rng = np.random.default_rng(1)
    # A store as the N3 code left it: JSON vectors only.
    store = TemporalStore(path)
    docs = [doc(f"d{i}") for i in range(6)]
    for d in docs:
        store.upsert("t", d)
    store.revoke("t", "d2")
    store.close()
    legacy = PassageStore(path)
    raw = {d.key: rng.normal(size=DIMS) for d in docs}
    for d in docs:
        legacy.put_vector("t", FP, embedding_input(d), raw[d.key])
    legacy.close()
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM vector_blobs")
    conn.commit()
    conn.close()
    before = legacy_digest(path, "t", FP)
    store = TemporalStore(path)  # opening migrates
    assert store.migrations()["vectors-json-to-binary/1"]["rows"] == 6
    assert store.migrations()["vectors-json-to-binary/1"]["json_roundtrip_mismatch"] == 0
    for d in docs:
        assert np.array_equal(store.vector("t", FP, embedding_input(d)), raw[d.key])
    assert store.vector_seq("t", FP) == 1
    assert store.digest("t", FP) == before == legacy_digest(path, "t", FP)
    store.close()
    assert TemporalStore(path).migrations()["vectors-json-to-binary/1"]["rows"] == 6  # once


def test_a_vector_rewrite_moves_the_generation_only_when_its_bytes_change(tmp_path):
    store = TemporalStore(tmp_path / "s.sqlite3")
    text = embedding_input(doc("a"))
    assert store.put_vector("t", FP, text, [1.0, 2, 3, 4]) is True
    assert store.put_vector("t", FP, text, np.array([1.0, 2, 3, 4])) is False  # same bytes
    assert store.vector_seq("t", FP) == 1
    assert store.put_vector("t", FP, text, [4.0, 3, 2, 1]) is True
    assert store.vector_seq("t", FP) == 2
    assert np.array_equal(store.base.vector("t", FP, text), [4.0, 3, 2, 1])  # JSON kept in step
    with pytest.raises(ContractError, match="invalid_passage_vector"):
        store.put_vector("t", FP, text, [0.0, 0, 0, 0])


def test_a_corrupted_vector_blob_is_an_error_not_a_value(tmp_path):
    path = tmp_path / "s.sqlite3"
    store = TemporalStore(path)
    text = embedding_input(doc("a"))
    store.put_vector("t", FP, text, [1.0, 2, 3, 4])
    store.conn.execute("UPDATE vector_blobs SET data=?", (np.ones(DIMS).tobytes(),))
    store.conn.commit()
    with pytest.raises(ContractError, match="vector_blob_corrupted"):
        store.vector("t", FP, text)


# --- generations, ledger, outbox ---------------------------------------------------------------
def test_each_change_moves_only_its_own_generation():
    store = TemporalStore(":memory:")
    g0 = store.generations("t")
    store.upsert("t", doc("a"))
    g1 = store.generations("t")
    assert g1.document_seq > g0.document_seq and g1.knowledge_seq == g1.document_seq
    store.revoke("t", "a")
    g2 = store.generations("t")
    assert g2.revocation_seq > g1.knowledge_seq and g2.document_seq == g1.document_seq
    store.revoke("t", "a")  # duplicate: nothing moves (N3)
    assert store.generations("t") == g2
    store.set_access("t", "a", "alice", "allow")
    g3 = store.generations("t")
    assert g3.acl_epoch == 1 and g3.knowledge_seq == g2.knowledge_seq
    store.put_vector("t", FP, "x", [1.0, 0, 0, 0])
    assert store.generations("t") == g3 and store.vector_seq("t", FP) == 1
    assert store.snapshot("t") == g3.knowledge_seq  # the N3 snapshot is unchanged in meaning


def test_the_ledger_and_outbox_are_written_with_the_change_or_not_at_all():
    store = TemporalStore(":memory:")
    store.upsert("t", doc("a"))
    store.revoke("t", "a")
    store.set_access("t", "a", "g", "deny")
    store.put_vector("t", FP, "x", [1.0, 0, 0, 0])
    assert [e["kind"] for e in store.events("t")] == ["insert", "revoke"]
    assert [o["kind"] for o in store.outbox("t")] == ["insert", "revoke", "acl", "vector"]
    before = (store.events("t"), store.outbox("t"))
    with pytest.raises(ValueError, match="immutable_temporal_revision_changed"):
        store.upsert("t", doc("a", text="A different body."))
    assert (store.events("t"), store.outbox("t")) == before


def test_unicode_ids_keep_the_same_canonical_order_as_sqlite():
    store = TemporalStore(":memory:")
    ids = ["b", "A", "é", "가", "Z9", "z_1", "a-2", "Ω"]
    for i in ids:
        store.upsert("t", doc(i))
    catalog = asyncio.run(store.storage.catalogs.get("t"))
    sql = [
        r[0]
        for r in store.conn.execute(
            "SELECT id FROM temporal_documents WHERE tenant='t' ORDER BY id,revision"
        )
    ]
    assert [d.id for d in catalog.documents] == sql == sorted(ids)


# --- N1F-1: what rebuilds the matrix -----------------------------------------------------------
def test_revocation_acl_and_pending_inserts_never_rebuild_the_matrix(tmp_path):
    store = TemporalStore(tmp_path / "s.sqlite3")
    rng = np.random.default_rng(3)
    for i in range(8):
        d = doc(f"d{i}")
        store.upsert("t", d)
        store.put_vector("t", FP, embedding_input(d), rng.normal(size=DIMS))
    emb = Embedder({"q": [1.0, 0, 0, 0]})
    first = search(store, emb)
    builds = store.index.stats["builds"]
    written = store.index.stats["matrix_bytes_written"]
    top = first[1]["seed_ids"][0].split("@")[0]
    store.revoke("t", top)  # revocation: a mask
    assert top not in {k.split("@")[0] for k in search(store, emb)[1]["seed_ids"]}
    store.set_access("t", "d1", "alice", "allow")  # ACL: a mask
    assert "d1@1" not in search(store, emb)[1]["seed_ids"]
    late = doc("late", text="A late pending note.")
    store.upsert("t", late)  # no vector yet: the BM25 pending path
    assert search(store, emb)[1]["pending"] == 1
    assert store.index.stats["builds"] == builds
    assert store.index.stats["matrix_bytes_written"] == written
    store.put_vector("t", FP, embedding_input(late), [1.0, 0, 0, 0])  # a vector: rebuild
    assert search(store, emb)[1]["seed_ids"][0] == "late@1"
    assert store.index.stats["builds"] == builds + 1


def test_a_new_revision_with_the_same_input_reuses_its_vector_without_embedding(tmp_path):
    store = TemporalStore(tmp_path / "s.sqlite3")
    emb = Embedder()
    retriever = TemporalRetriever(store, emb)
    store.upsert("t", doc("a", text="Same words."))
    asyncio.run(retriever.index("t", CallContext.with_timeout(30, max_embedding_calls=8)))
    assert emb.calls == 1
    store.upsert("t", doc("a", revision=2, text="Same words.", day=3))
    out = asyncio.run(retriever.index("t", CallContext.with_timeout(30, max_embedding_calls=8)))
    assert emb.calls == 1 and out["indexed"] == 0 and store.vector_seq("t", FP) == 1
    store.upsert("t", doc("a", revision=3, title="Renamed", text="Same words.", day=5))
    asyncio.run(retriever.index("t", CallContext.with_timeout(30, max_embedding_calls=8)))
    assert emb.calls == 2  # the title is part of the embedding input


def test_the_engine_records_the_deployment_it_ran_on():
    store = TemporalStore(":memory:")
    described = store.storage.describe()
    assert set(described) == {"state", "index", "precision", "config_hash", "fingerprint"}
    assert json.dumps(described)  # plain data: no paths, no credentials
    assert "memory" not in json.dumps(described)


def test_the_vector_change_feed_returns_the_latest_bytes_after_a_generation():
    store = TemporalStore(":memory:")
    store.put_vector("t", FP, "a", [1.0, 0, 0, 0])
    store.put_vector("t", FP, "b", [0.0, 1, 0, 0])
    since = store.vector_seq("t", FP)
    store.put_vector("t", FP, "c", [0.0, 0, 1, 0])
    store.put_vector("t", FP, "a", [2.0, 0, 0, 0])  # a rewrite: only its latest bytes
    with store.scan_changes("t", FP, since) as scan:
        rows = list(scan.rows)
    assert scan.vector_seq == store.vector_seq("t", FP) == since + 2
    assert [r[0] for r in rows] == [input_hash("c"), input_hash("a")]
    assert np.frombuffer(rows[1][2], dtype="<f8").tolist() == [2.0, 0, 0, 0]
    with store.scan_changes("t", FP, scan.vector_seq) as scan:
        assert list(scan.rows) == []
