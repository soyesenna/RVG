"""PORT-2: a vector index in another service (pgvector, Qdrant) behind the same port. On real
services, for every state x index combination: the canonical precision gives the local exact
index's results bit for bit (filters, snapshots, principals, pending documents, projection lag
covered by the exact overlay); the native precision ranks exactly over the service's float32
vectors; late writes never bring back older vectors; projectors are fenced; tenants never mix.

Needs `DEEPJEV_TEST_PGVECTOR_DSN` (PostgreSQL with the vector extension) and/or
`DEEPJEV_TEST_QDRANT_URL`; `DEEPJEV_TEST_POSTGRES_DSN` is used for PostgreSQL state when set.
Without them the tests are skipped — reported as not run, never as passed."""

import asyncio
import os
import uuid

import numpy as np
import pytest
import test_n3_release as n3
from matrix_worker import DIMS, SPACE, Queries, doc

from rvg.contracts import EmbeddingBatch
from rvg.errors import ContractError
from rvg.runtime import CallContext
from rvg.storage.factory import local_storage, remote_storage
from rvg.storage.sqlite import SQLiteState
from rvg.temporal.access import TrustedPrincipal
from rvg.temporal.models import TemporalQuestion
from rvg.temporal.store import TemporalRetriever

PGV = os.environ.get("DEEPJEV_TEST_PGVECTOR_DSN")
QDRANT = os.environ.get("DEEPJEV_TEST_QDRANT_URL")
PG_STATE = os.environ.get("DEEPJEV_TEST_POSTGRES_DSN") or PGV
FP = SPACE.fingerprint
COMBOS = [
    pytest.param("postgres", "pgvector", marks=pytest.mark.skipif(not PGV, reason="no pgvector")),
    pytest.param(
        "postgres",
        "qdrant",
        marks=pytest.mark.skipif(not (QDRANT and PG_STATE), reason="no qdrant/postgres"),
    ),
    pytest.param("sqlite", "qdrant", marks=pytest.mark.skipif(not QDRANT, reason="no qdrant")),
    pytest.param("sqlite", "pgvector", marks=pytest.mark.skipif(not PGV, reason="no pgvector")),
]
PRINCIPALS = [None, TrustedPrincipal("alice", groups=("staff",)), TrustedPrincipal("bob")]


class Q(Queries):
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=float)

    async def embed(self, texts, *, purpose, context):
        return EmbeddingBatch(tuple(tuple(self.vector) for _ in texts), SPACE, ("f",), ())


@pytest.fixture
def deployments(tmp_path):
    """Makes (state, remote storage, local reference storage) for a combination; cleans up."""
    made = []

    def make(state_kind, index_kind, *, precision="canonical", overlay_limit=20000):
        tag = uuid.uuid4().hex[:10]
        if state_kind == "sqlite":
            state = SQLiteState(tmp_path / f"s-{tag}.sqlite3")
        else:
            from rvg.storage.postgres import PostgresState

            state = PostgresState(PG_STATE, schema=f"st_{tag}")
        if index_kind == "pgvector":
            from rvg.storage.pgvector import PgvectorBackend

            backend = PgvectorBackend(PGV, schema=f"ix_{tag}")
        else:
            from rvg.storage.qdrant import QdrantBackend

            backend = QdrantBackend(QDRANT, prefix=f"t{tag}")
        remote = remote_storage(state, backend, precision=precision, overlay_limit=overlay_limit)
        local = local_storage(state)
        made.append((state_kind, state, index_kind, backend, tag))
        return state, remote, local

    yield make
    for state_kind, state, index_kind, backend, tag in made:
        if index_kind == "qdrant":
            for name in list(backend._ready):
                backend.client.delete(f"/collections/{name}")
        if index_kind == "qdrant":
            backend.close()
        if index_kind == "pgvector":
            backend.conn.execute(f"DROP SCHEMA IF EXISTS ix_{tag} CASCADE")
            backend.close()
        if state_kind == "postgres":
            with state.lock:
                state.conn.execute(f"DROP SCHEMA IF EXISTS st_{tag} CASCADE")
        state.close()


def populate(state, n=60, seed=0, tenant="t"):
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        d = doc(i, text=f"Fact number {i} about topic {i % 7}.")
        docs.append(d)
        state.upsert(tenant, d)
        if i % 9 != 0:  # every ninth waits for its vector: the pending path
            state.put_vector(tenant, FP, TemporalRetriever.input_text(d), rng.normal(size=DIMS))
    return docs


def search(storage, query, *, principal=None, allowed=None, at=None, tenant="t", known=None):
    q = TemporalQuestion(
        text="fact topic", as_of="2026-09-20T00:00:00Z", known_at=known or "2026-09-20T00:00:00Z"
    )

    async def go():
        retriever = TemporalRetriever(
            storage, Q(query), max_documents=200, principal=principal, allowed_ids=allowed
        )
        selected, info = await retriever.search(
            tenant, q, CallContext.with_timeout(30, max_embedding_calls=4), top_k=8, at=at
        )
        return (
            [d.key for d in selected],
            info["seed_ids"],
            [r["score"] for r in info.get("ranking", [])],
            info["pending"],
        )

    try:
        return asyncio.run(go())
    except ContractError as exc:
        return ("error", exc.code)


def queries(n=6, seed=11):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=DIMS) for _ in range(n)]


@pytest.mark.parametrize(("state_kind", "index_kind"), COMBOS)
def test_canonical_precision_equals_the_local_exact_index(deployments, state_kind, index_kind):
    state, remote, local = deployments(state_kind, index_kind)
    docs = populate(state)
    report = remote.projector.sync("t", FP, DIMS)
    assert report["status"] == "ok" and remote.index.backend.count("t", FP) == report["written"]
    state.revoke("t", docs[4].id)
    state.set_access("t", docs[5].id, "staff", "allow")
    state.set_access("t", docs[7].id, "bob", "deny")
    snapshot = state.snapshot("t")
    late = doc(900, text="A late fact about topic 3.")
    state.upsert("t", late)
    state.put_vector("t", FP, TemporalRetriever.input_text(late), queries(1, 3)[0])
    ids = [d.id for d in docs]
    for query in queries():
        for principal in PRINCIPALS:
            assert search(remote, query, principal=principal) == search(
                local, query, principal=principal
            )
        assert search(remote, query, allowed=ids[::3]) == search(local, query, allowed=ids[::3])
        assert search(remote, query, at=snapshot) == search(local, query, at=snapshot)
        assert search(remote, query, known="2026-09-01T00:00:00Z") == search(
            local, query, known="2026-09-01T00:00:00Z"
        )
    assert remote.index.stats["overlay_rows"] > 0  # the late document came through the overlay
    assert remote.projector.sync("t", FP, DIMS)["written"] >= 1
    before = remote.index.stats["overlay_rows"]
    search(remote, queries(1)[0])
    assert remote.index.stats["overlay_rows"] == before  # projected: nothing left to overlay


@pytest.mark.parametrize(("state_kind", "index_kind"), COMBOS)
def test_native_precision_ranks_exactly_over_the_float32_vectors(
    deployments, state_kind, index_kind
):
    state, remote, _ = deployments(state_kind, index_kind, precision="native")
    populate(state, n=120, seed=4)
    remote.projector.sync("t", FP, DIMS)
    catalog = asyncio.run(remote.catalogs.get("t"))
    for query in queries(8, seed=5):
        rows, scores = asyncio.run(remote.index.view("t", FP, DIMS, catalog)).rank(
            np.ones(len(catalog), dtype=bool), query, 10
        )
        # float32 oracle over the same projected vectors
        ready = [
            i
            for i, h in enumerate(catalog.input_hashes)
            if state.vector("t", FP, _text(catalog, i)) is not None
        ]
        m = np.asarray([state.vector("t", FP, _text(catalog, i)) for i in ready])
        m32 = (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)
        q32 = (np.asarray(query) / np.linalg.norm(query)).astype(np.float32)
        oracle = m32 @ q32
        order = np.argsort(-oracle, kind="stable")[:10]
        assert [ready[i] for i in order] == rows.tolist()
        assert np.allclose(scores, oracle[order], atol=1e-5)


def _text(catalog, i):
    d = catalog.documents[i]
    return TemporalRetriever.input_text(d)


@pytest.mark.parametrize(("state_kind", "index_kind"), COMBOS)
def test_late_writes_and_fenced_projectors_never_bring_back_older_vectors(
    deployments, state_kind, index_kind
):
    state, remote, local = deployments(state_kind, index_kind)
    docs = populate(state, n=30, seed=6)
    remote.projector.sync("t", FP, DIMS)
    target = docs[2]
    text = TemporalRetriever.input_text(target)
    old = state.vector("t", FP, text)
    state.put_vector("t", FP, text, -old)  # a newer vector...
    remote.projector.sync("t", FP, DIMS)
    point = remote.index._point_ids("t", FP, asyncio.run(remote.catalogs.get("t")))[0]
    row = asyncio.run(remote.catalogs.get("t")).row_of_key[target.key]
    stale = (
        point[row],
        old / np.linalg.norm(old),
        {
            "tenant": "t",
            "document_id": target.id,
            "revision": 1,
            "input_hash": "x",
            "observed_us": 0,
            "inserted": 0,
            "blob_seq": 1,
        },
    )
    remote.index.backend.upsert(FP, [stale])  # ...then a late write of the older one
    for query in [old, -old, *queries(3)]:
        assert search(remote, query) == search(local, query)
    assert remote.index.stats["stale_points"] == 0  # the service kept the newer bytes
    # a second projector cannot lease while the first holds it; the first is fenced after expiry
    from rvg.storage.projection import Projector

    other = Projector(state, state, remote.index.backend, remote.catalogs, holder="other", ttl=5)
    state.projection_lease(
        "t", FP, remote.index.backend.name, remote.projector.holder, now=0, ttl=10**12
    )
    assert other.sync("t", FP, DIMS)["status"] == "busy"


@pytest.mark.parametrize(("state_kind", "index_kind"), COMBOS)
def test_tenants_never_mix_and_invalid_vectors_fail_like_the_local_index(
    deployments, state_kind, index_kind
):
    state, remote, local = deployments(state_kind, index_kind)
    docs = populate(state, n=20, seed=8)
    twin = doc(0, text="Another tenant's secret 999.")
    state.upsert("u", twin)
    state.put_vector(
        "u",
        FP,
        TemporalRetriever.input_text(twin),
        state.vector("t", FP, TemporalRetriever.input_text(docs[1])),
    )
    remote.projector.sync("t", FP, DIMS)
    remote.projector.sync("u", FP, DIMS)
    for query in queries(4):
        got = search(remote, query)
        assert got == search(local, query) and all("secret" not in k for k in got[0])
    state.put_vector("t", FP, TemporalRetriever.input_text(docs[1]), [1.0, 2.0])  # wrong size
    remote.projector.sync("t", FP, DIMS)
    assert (
        search(remote, queries(1)[0])
        == search(local, queries(1)[0])
        == ("error", "temporal_cached_vector_invalid")
    )


@pytest.mark.parametrize(("state_kind", "index_kind"), COMBOS)
def test_an_overlay_past_its_limit_is_index_not_caught_up(deployments, state_kind, index_kind):
    state, remote, _ = deployments(state_kind, index_kind, overlay_limit=5)
    populate(state, n=30, seed=9)
    remote.index.projector = None  # nobody to catch up
    assert search(remote, queries(1)[0]) == ("error", "index_not_caught_up")


N3_SCENARIOS = [
    (n3.test_baseline_every_mode_releases_the_same_answer_when_nothing_changes, {}),
    (n3.test_a_new_revision_during_generation, {}),
    *[(n3.test_a_revocation_during_generation_is_never_released, {"mode": m}) for m in n3.MODES],
    *[
        (n3.test_a_permission_withdrawn_during_generation_is_never_released, {"mode": m})
        for m in n3.MODES
    ],
    (n3.test_documents_under_acl_are_invisible_without_a_matching_principal, {}),
    (n3.test_an_unrelated_insert_blocks_only_the_current_mode, {}),
    (n3.test_one_relevant_change_is_retried_and_two_end_in_stale_after_retry, {}),
    (n3.test_duplicate_events_change_nothing, {}),
    (n3.test_a_late_older_revision_does_not_replace_the_newer_one, {}),
]


class Deployment:
    def __init__(self, storage):
        self.storage = storage

    def __getattr__(self, name):
        return getattr(self.storage.state, name)


@pytest.mark.parametrize(("state_kind", "index_kind"), COMBOS)
@pytest.mark.parametrize(
    ("scenario", "kwargs"), N3_SCENARIOS, ids=[f"{s.__name__}{k}" for s, k in N3_SCENARIOS]
)
def test_n3_release_scenarios_hold_on_remote_indexes(
    deployments, monkeypatch, state_kind, index_kind, scenario, kwargs
):
    def knowledge():
        _, remote, _ = deployments(state_kind, index_kind)
        deployment = Deployment(remote)
        rng = np.random.default_rng(3)
        put(
            deployment,
            n3.doc("p1", "The budget of project P is 500.", family="P"),
            n3.Q + 0.1 * np.eye(n3.DIMS)[1],
        )
        for k in range(25):
            v = rng.normal(size=n3.DIMS)
            v[0] = -abs(v[0]) * 0.1
            put(deployment, n3.doc(f"b{k:02d}", f"Background remark {k} on weather."), v)
        remote.projector.sync("t", n3.SPACE.fingerprint, n3.DIMS)
        return deployment

    def put(store, d, vector):
        store.upsert("t", d)
        store.put_vector("t", n3.SPACE.fingerprint, TemporalRetriever.input_text(d), vector)

    monkeypatch.setattr(n3, "knowledge", knowledge)
    monkeypatch.setattr(n3, "put", put)
    scenario(**kwargs)
