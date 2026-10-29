"""PORT-1: one state contract, two backends. The same event ledger applied to the SQLite and the
PostgreSQL state stores gives the same observations after every step — generations and event
numbers compared by their order within the tenant (SQLite numbers are DB-wide, PostgreSQL's
per tenant), everything else as is — and the product path (N3 release scenarios, dense search)
behaves the same on a PostgreSQL deployment, with no SQLite anywhere.

PostgreSQL tests need a server: `DEEPJEV_TEST_POSTGRES_DSN` (each test works in its own schema
and drops it). Without it they are skipped — a skip is reported as not run, never as a pass."""

import asyncio
import os
import sqlite3
import threading
import time
import uuid

import numpy as np
import pytest
import test_n3_release as n3
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from rvg.errors import ContractError
from rvg.storage.factory import local_storage
from rvg.storage.identity import embedding_input
from rvg.storage.sqlite import SQLiteState
from rvg.temporal.access import TrustedPrincipal
from rvg.temporal.models import TemporalDocument
from rvg.temporal.store import TemporalRetriever

DSN = os.environ.get("DEEPJEV_TEST_POSTGRES_DSN")
needs_postgres = pytest.mark.skipif(not DSN, reason="DEEPJEV_TEST_POSTGRES_DSN not set")
FP = n3.SPACE.fingerprint
PRINCIPALS = {
    "none": None,
    "alice": TrustedPrincipal("alice", groups=("staff",)),
    "bob": TrustedPrincipal("bob"),
    "carol": TrustedPrincipal("carol", groups=("staff", "audit")),
}


def pg_state(schema=None):
    from rvg.storage.postgres import PostgresState

    return PostgresState(DSN, schema=schema or "t_" + uuid.uuid4().hex[:16])


def drop(schema):
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


@pytest.fixture
def pg():
    made = []

    def make(schema=None):
        state = pg_state(schema)
        made.append(state)
        return state

    yield make
    for state in made:
        state.close()  # idempotent
    for schema in {s.schema for s in made}:
        drop(schema)


def doc(identifier, *, revision=1, day=1, family=None, text=None, title=None):
    text = text or f"Fact {identifier} rev {revision}."
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family or identifier.split("-")[0],
            "revision": revision,
            "title": title or f"T {identifier}",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": f"2026-09-{day:02d}T00:00:00Z",
            "observed_at": f"2026-09-{day:02d}T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
        }
    )


def observe(state, tenant, ids):
    """Everything a caller of the state port can see, event numbers as ranks in the tenant."""
    events = state.events(tenant)
    rank = {e["seq"]: i + 1 for i, e in enumerate(events)}
    norm = lambda seq: None if seq is None else (0 if seq == 0 else rank[seq])
    g = state.generations(tenant)
    snapshots = [0, *rank]
    catalog = state.catalog_rows(tenant)
    out = {
        "generations": (
            norm(g.knowledge_seq),
            norm(g.document_seq),
            norm(g.revocation_seq),
            g.acl_epoch,
        ),
        "snapshot": norm(state.snapshot(tenant)),
        "events": [(e["kind"], e["document_id"], e["revision"]) for e in events],
        "catalog": [(i, r, norm(e)) for i, r, e in catalog.rows],
        "catalog_generations": norm(catalog.generations.knowledge_seq),
        "documents": [d.key for d in state.documents(tenant)],
        "known_before_day_5": [
            d.key for d in state.documents(tenant, known_at=doc("x", day=5).observed_at)
        ],
        "revoked": sorted(state.revoked(tenant)),
        "hidden": {k: sorted(state.hidden(tenant, p)) for k, p in PRINCIPALS.items()},
        "acl_epoch": state.acl_epoch(tenant),
        "digest": state.digest(tenant, FP),
        "vector_seq": state.vector_seq(tenant, FP),
        "outbox": [(o["kind"], o["payload"].get("id")) for o in state.outbox(tenant)],
    }
    with state.scan_changes(tenant, FP, 0) as scan:
        out["vector_changes"] = [(h, d, bytes(b), q) for h, d, b, q in scan.rows]
    for at in snapshots:
        seq = 0 if at == 0 else at
        out[f"at_{norm(seq)}"] = (
            [d.key for d in state.documents(tenant, at=seq)],
            sorted(state.revoked(tenant, at=seq)),
            sorted(state.changed_families(tenant, seq)),
        )
    for name, principal in PRINCIPALS.items():
        r = state.authorize_release(tenant, ids, principal)
        out[f"release_{name}"] = (r.ok, r.reason, r.blocked, norm(r.revocation_seq), r.acl_epoch)
    return out


# --- the same ledger, step by step -------------------------------------------------------------
LEDGER = [
    ("upsert", "t", doc("a-1")),
    ("upsert", "u", doc("a-1", text="Other tenant, same id.")),
    ("upsert", "t", doc("가-1", day=3)),
    ("upsert", "t", doc("B-1", day=4)),
    ("upsert", "t", doc("é-1", day=6)),
    ("vector", "t", doc("a-1"), [1.0, 0, 0, 0, 0, 0]),
    ("vector", "t", doc("a-1"), [1.0, 0, 0, 0, 0, 0]),  # identical bytes: no generation
    ("upsert", "t", doc("a-1", revision=2, day=7)),
    ("revoke", "t", "B-1"),
    ("revoke", "t", "B-1"),  # duplicate: no event
    ("acl", "t", "a-1", "staff", "allow"),
    ("acl", "t", "가-1", "bob", "deny"),
    ("acl", "t", "é-1", "audit", "allow"),
    ("acl", "t", "é-1", "carol", "deny"),  # deny beats a group allow
    ("vector", "t", doc("a-1", revision=2, day=7), [0.5, 0.5, 0, 0, 0, 0]),
    ("upsert_changed", "t", doc("a-1", text="A different body.")),  # immutable: refused
    ("acl", "t", "a-1", "staff", None),
    ("revoke", "u", "a-1"),
    ("upsert", "t", doc("Z-1", day=2)),
]


def apply(state, step):
    kind, tenant, *rest = step
    if kind == "upsert":
        state.upsert(tenant, rest[0])
    elif kind == "upsert_changed":
        with pytest.raises(ValueError, match="immutable_temporal_revision_changed"):
            state.upsert(tenant, rest[0])
    elif kind == "vector":
        state.put_vector(tenant, FP, embedding_input(rest[0]), rest[1])
    elif kind == "revoke":
        state.revoke(tenant, rest[0])
    elif kind == "acl":
        state.set_access(tenant, *rest)


@needs_postgres
def test_the_same_ledger_gives_the_same_observations_after_every_step(tmp_path, pg):
    sqlite, postgres = SQLiteState(tmp_path / "s.sqlite3"), pg()
    ids = ["a-1", "가-1", "B-1", "é-1", "Z-1"]
    for n, step in enumerate(LEDGER):
        apply(sqlite, step)
        apply(postgres, step)
        for tenant in ("t", "u"):
            assert observe(sqlite, tenant, ids) == observe(postgres, tenant, ids), (n, step, tenant)
    # canonical order is code points in both, whatever the server's collation
    assert [r[0] for r in postgres.catalog_rows("t").rows] == sorted(
        r[0] for r in postgres.catalog_rows("t").rows
    )
    assert [e["seq"] for e in postgres.events("t")] == list(range(1, len(postgres.events("t")) + 1))


class LedgerMachine(RuleBasedStateMachine):
    """Random ledgers on both stores; observations compared after every step."""

    def __init__(self):
        super().__init__()
        import tempfile

        self.folder = tempfile.mkdtemp()
        self.sqlite = SQLiteState(os.path.join(self.folder, "s.sqlite3"))
        self.postgres = pg_state()
        self.revisions, self.latest = {}, {}

    def teardown(self):
        import shutil

        self.sqlite.close()  # 0.1.0 (parent PUB-4): each example's files, not all forty at the end
        shutil.rmtree(self.folder, ignore_errors=True)
        self.postgres.close()
        drop(self.postgres.schema)

    def both(self, fn):
        for state in (self.sqlite, self.postgres):
            fn(state)

    @rule(k=st.integers(0, 4), tenant=st.sampled_from(["t", "u"]), day=st.integers(1, 9))
    def insert(self, k, tenant, day):
        key = (tenant, k)
        revision = self.revisions.get(key, 0) + 1
        self.revisions[key] = revision
        d = doc(f"d{k}", revision=revision, day=day, family=f"f{k % 2}")
        self.latest[key] = d
        self.both(lambda s: s.upsert(tenant, d))

    @rule(k=st.integers(0, 4), tenant=st.sampled_from(["t", "u"]))
    def duplicate(self, k, tenant):
        d = self.latest.get((tenant, k))
        if d is not None:  # the same revision again: nothing may change
            self.both(lambda s: s.upsert(tenant, d))

    @rule(k=st.integers(0, 4), tenant=st.sampled_from(["t", "u"]))
    def revoke(self, k, tenant):
        self.both(lambda s: s.revoke(tenant, f"d{k}"))

    @rule(
        k=st.integers(0, 4),
        who=st.sampled_from(["alice", "bob", "staff", "audit"]),
        effect=st.sampled_from(["allow", "deny", None]),
    )
    def acl(self, k, who, effect):
        self.both(lambda s: s.set_access("t", f"d{k}", who, effect))

    @rule(k=st.integers(0, 4), scale=st.sampled_from([1.0, 2.0]))
    def vector(self, k, scale):
        d = self.latest.get(("t", k))
        if d is not None:
            text = embedding_input(d)
            self.both(lambda s: s.put_vector("t", FP, text, [scale, k, 1, 0, 0, 0]))

    @invariant()
    def same(self):
        ids = [f"d{k}" for k in range(5)]
        for tenant in ("t", "u"):
            assert observe(self.sqlite, tenant, ids) == observe(self.postgres, tenant, ids)


@needs_postgres
def test_random_ledgers_give_the_same_observations():
    run = LedgerMachine.TestCase
    run.settings = settings(
        max_examples=40,
        stateful_step_count=18,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    run().runTest()


# --- PostgreSQL commit order, fence, rollback --------------------------------------------------
@needs_postgres
def test_concurrent_writers_get_gapless_commit_ordered_event_numbers(pg):
    first = pg()
    writers = [pg(first.schema) for _ in range(8)]
    errors = []

    def work(n, state):
        try:
            for i in range(25):
                state.upsert("t", doc(f"w{n}-{i}", day=1 + i % 9))
                if i % 5 == 0:
                    state.revoke("t", f"w{n}-{i}")
        except Exception as exc:  # noqa: BLE001 - any failure of a writer thread is reported
            errors.append(repr(exc))

    threads = [threading.Thread(target=work, args=(n, s)) for n, s in enumerate(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    events = first.events("t")
    assert [e["seq"] for e in events] == list(range(1, 8 * 30 + 1))  # 200 inserts + 40 revokes
    assert first.snapshot("t") == len(events)
    inserted = {
        (e["document_id"], e["revision"]): e["seq"] for e in events if e["kind"] == "insert"
    }
    rows = first.catalog_rows("t").rows
    assert {(i, r): e for i, r, e in rows} == inserted
    assert [o["id"] for o in first.outbox("t")] == list(range(1, len(first.outbox("t")) + 1))


class SlowRevoke:
    """A revocation that keeps its transaction (and the tenant row) open for a while."""

    def __init__(self, state, seconds):
        self.state, self.seconds = state, seconds
        original = state._outbox

        def slow(conn, *args):
            original(conn, *args)
            time.sleep(self.seconds)

        state._outbox = slow


@needs_postgres
def test_the_release_fence_waits_for_a_writer_and_sees_its_commit(pg):
    reader = pg()
    reader.upsert("t", doc("p-1"))
    writer = pg(reader.schema)
    SlowRevoke(writer, 0.6)
    thread = threading.Thread(target=writer.revoke, args=("t", "p-1"))
    thread.start()
    time.sleep(0.15)  # the writer holds the tenant row now
    t0 = time.monotonic()
    fence = reader.authorize_release("t", {"p-1"}, None)
    waited = time.monotonic() - t0
    thread.join()
    assert waited > 0.3  # it did not read a snapshot from before the writer's commit
    assert not fence.ok and fence.reason == "revoked_before_release" and fence.blocked == ("p-1",)


@needs_postgres
def test_a_failed_transaction_consumes_no_number(pg):
    state = pg()
    state.upsert("t", doc("a-1"))
    before = (state.generations("t"), state.events("t"), state.outbox("t"))
    with pytest.raises(ValueError):
        state.upsert("t", doc("a-1", text="Changed body."))
    original = state._outbox

    def broken(conn, *args):
        original(conn, *args)
        raise RuntimeError("crash after the change, before commit")

    state._outbox = broken
    with pytest.raises(RuntimeError):
        state.revoke("t", "a-1")
    state._outbox = original
    assert (state.generations("t"), state.events("t"), state.outbox("t")) == before
    assert state.revoked("t") == set()
    state.revoke("t", "a-1")
    assert [e["seq"] for e in state.events("t")] == [1, 2]  # no gap left by the failure


@needs_postgres
def test_a_corrupted_vector_is_an_error_on_postgres_too(pg):
    state = pg()
    text = embedding_input(doc("a-1"))
    state.put_vector("t", FP, text, [1.0, 2, 3, 4, 5, 6])
    with state.lock:
        state.conn.execute("UPDATE vectors SET data=%s", (np.ones(6).tobytes(),))
    with pytest.raises(ContractError, match="vector_blob_corrupted"):
        state.vector("t", FP, text)


@needs_postgres
def test_export_import_keeps_numbers_bodies_vectors_and_the_digest(tmp_path, pg):
    source = SQLiteState(tmp_path / "s.sqlite3")
    for step in LEDGER:
        apply(source, step)
    target = pg()
    for tenant in ("t", "u"):
        target.import_tenant(tenant, **source.export_tenant(tenant))
        assert target.digest(tenant, FP) == source.digest(tenant, FP)
        assert target.generations(tenant) == source.generations(tenant)  # numbers preserved
        assert target.catalog_rows(tenant) == source.catalog_rows(tenant)
        assert target.vector_seq(tenant, FP) == source.vector_seq(tenant, FP)
    target.upsert("t", doc("new-1", day=8))  # continues after the imported numbers
    assert target.snapshot("t") == source.snapshot("t") + 1
    with pytest.raises(ValueError, match="import_target_tenant_not_empty"):
        target.import_tenant("t", **source.export_tenant("t"))


# --- the product path on a PostgreSQL deployment ------------------------------------------------
class Deployment:
    """A storage deployment with its state store's methods, as the N3 scenarios use a store."""

    def __init__(self, storage):
        self.storage = storage

    def __getattr__(self, name):
        return getattr(self.storage.state, name)


def pg_knowledge(pg, tmp_path):
    def knowledge():
        deployment = Deployment(local_storage(pg(), matrix_dir=tmp_path / uuid.uuid4().hex))
        rng = np.random.default_rng(3)
        pg_put(
            deployment,
            n3.doc("p1", "The budget of project P is 500.", family="P"),
            n3.Q + 0.1 * np.eye(n3.DIMS)[1],
        )
        for k in range(25):
            v = rng.normal(size=n3.DIMS)
            v[0] = -abs(v[0]) * 0.1
            pg_put(deployment, n3.doc(f"b{k:02d}", f"Background remark {k} on weather."), v)
        return deployment

    return knowledge


def pg_put(store, d, vector):
    store.upsert("t", d)
    store.put_vector("t", FP, TemporalRetriever.input_text(d), vector)


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


@needs_postgres
@pytest.mark.parametrize(
    ("scenario", "kwargs"), N3_SCENARIOS, ids=[f"{s.__name__}{k}" for s, k in N3_SCENARIOS]
)
def test_n3_release_scenarios_hold_on_a_postgres_deployment_without_sqlite(
    scenario, kwargs, pg, tmp_path, monkeypatch
):
    monkeypatch.setattr(n3, "knowledge", pg_knowledge(pg, tmp_path))
    monkeypatch.setattr(n3, "put", pg_put)

    def no_sqlite(*args, **kwargs):
        raise AssertionError("a PostgreSQL deployment opened SQLite")

    monkeypatch.setattr(sqlite3, "connect", no_sqlite)
    scenario(**kwargs)


@needs_postgres
def test_dense_search_is_identical_on_both_deployments_and_survives_a_restart(tmp_path, pg):
    from rvg.temporal.store import TemporalStore

    sqlite = TemporalStore(tmp_path / "s.sqlite3")
    postgres = Deployment(local_storage(pg(), matrix_dir=tmp_path / "pgm"))
    rng = np.random.default_rng(9)
    for i in range(60):
        d = doc(f"d{i:02d}-x", day=1 + i % 9, family=f"f{i % 7}")
        v = rng.normal(size=n3.DIMS)
        for store in (sqlite, postgres):
            store.upsert("t", d)
            if i % 6:
                store.put_vector("t", FP, embedding_input(d), v)
    for store in (sqlite, postgres):
        store.revoke("t", "d03-x")
        store.set_access("t", "d05-x", "bob", "deny")

    def search(store, principal):
        q = n3.TemporalQuestion(
            text="Fact", as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
        )
        retriever = TemporalRetriever(store, n3.Fixed(), max_documents=64, principal=principal)
        selected, info = asyncio.run(
            retriever.search(
                "t", q, n3.CallContext.with_timeout(30, max_embedding_calls=4), top_k=8
            )
        )
        return [d.key for d in selected], info["seed_ids"], [r["score"] for r in info["ranking"]]

    for principal in PRINCIPALS.values():
        assert search(sqlite, principal) == search(postgres, principal)
    schema = postgres.storage.state.schema
    postgres.storage.close()
    reopened = Deployment(local_storage(pg(schema), matrix_dir=tmp_path / "pgm"))
    assert search(reopened, None) == search(sqlite, None)
    assert reopened.storage.index.stats["adopted"] == 1  # the published matrix, not a rebuild
