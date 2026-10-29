"""PORT-3: a live tenant moved to another state store and back (docs/PORT3_PROTOCOL.md).

Copy at an outbox mark, catch-up of the changes after it (idempotent, resumable, gaps refused),
a switch of the deployment pointer under its exclusive lock with verification, release fences of
in-flight requests that refuse once the pointer has moved, and rollback. The target is SQLite,
and PostgreSQL when `DEEPJEV_TEST_POSTGRES_DSN` names a server."""

import json
import multiprocessing
import os
import time
import uuid

import numpy as np
import pytest
import test_n3_release as n3
from migration_worker import writer

from rvg.errors import ContractError
from rvg.storage.factory import local_storage
from rvg.storage.migration import (
    DeploymentPointer,
    Migration,
    PinnedRelease,
    pin,
    switch,
    write_active,
)
from rvg.storage.sqlite import SQLiteState
from rvg.temporal.store import TemporalRetriever

DSN = os.environ.get("DEEPJEV_TEST_POSTGRES_DSN")
FP = n3.SPACE.fingerprint
TARGETS = ["sqlite", pytest.param("postgres", marks=pytest.mark.skipif(not DSN, reason="no PG"))]


def drop(schema):
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


@pytest.fixture
def stores(tmp_path):
    """Named state stores with their specs (for a writer process); PostgreSQL schemas are
    dropped afterwards."""
    made, schemas = [], []

    def make(kind, name):
        if kind == "sqlite":
            spec = {"kind": "sqlite", "path": str(tmp_path / f"{name}.sqlite3")}
            state = SQLiteState(spec["path"])
        else:
            from rvg.storage.postgres import PostgresState

            spec = {"kind": "postgres", "dsn": DSN, "schema": "mig_" + uuid.uuid4().hex[:12]}
            state = PostgresState(DSN, schema=spec["schema"])
            schemas.append(spec["schema"])
        made.append(state)
        return state, spec

    yield make
    for state in made:
        state.close()
    for schema in schemas:
        drop(schema)


def put(state, d, vector):
    state.upsert("t", d)
    state.put_vector("t", FP, TemporalRetriever.input_text(d), vector)


def seed(state):
    rng = np.random.default_rng(3)
    put(state, n3.doc("p1", "The budget of project P is 500.", family="P"), n3.Q + 0.1)
    for k in range(25):
        v = rng.normal(size=n3.DIMS)
        v[0] = -abs(v[0]) * 0.1
        put(state, n3.doc(f"b{k:02d}", f"Background remark {k} on weather."), v)


def churn(state, start, count):
    """Every kind of change, in order: inserts with vectors, a revocation, an ACL change and a
    vector rewrite."""
    rng = np.random.default_rng(start)
    for k in range(start, start + count):
        d = n3.doc(f"c{k:03d}", f"Change {k}.")
        put(state, d, rng.normal(size=n3.DIMS))
        if k % 3 == 0:
            state.revoke("t", f"b{k % 25:02d}")
        if k % 4 == 0:
            state.set_access("t", f"c{k:03d}", "alice", "deny")
        if k % 5 == 0:
            state.put_vector("t", FP, TemporalRetriever.input_text(d), rng.normal(size=n3.DIMS))


def migrated(stores, tmp_path, kind):
    source, _ = stores("sqlite", "A")
    target, _ = stores(kind, "B")
    seed(source)
    return source, target, Migration(source, target, "t", tmp_path / "ledger.json")


@pytest.mark.parametrize("kind", TARGETS)
def test_copy_then_catch_up_reaches_the_same_tenant_with_the_same_numbers(stores, tmp_path, kind):
    source, target, migration = migrated(stores, tmp_path, kind)
    churn(source, 0, 5)
    migration.copy()
    churn(source, 5, 20)  # after the export: only the outbox carries these
    done = migration.catch_up()
    assert done["applied"] > 0
    assert all(migration.verify([FP]).values())
    assert target.snapshot("t") == source.snapshot("t")
    assert target.generations("t") == source.generations("t")


@pytest.mark.parametrize("kind", TARGETS)
def test_a_repeated_or_resumed_catch_up_changes_nothing(stores, tmp_path, kind):
    source, target, migration = migrated(stores, tmp_path, kind)
    mark = migration.copy()
    churn(source, 0, 12)
    first = migration.catch_up(limit=7)  # a crash after seven entries
    resumed = Migration(source, target, "t", tmp_path / "ledger.json").catch_up()
    assert first["applied"] > 0 and resumed["applied"] > 0
    before = (target.events("t"), target.acl_epoch("t"), target.digest("t", FP))
    migration.resume_from(mark)  # every entry delivered again
    again = migration.catch_up()
    assert again["applied"] == 0
    assert (target.events("t"), target.acl_epoch("t"), target.digest("t", FP)) == before
    assert all(migration.verify([FP]).values())


@pytest.mark.parametrize("kind", TARGETS)
def test_a_gap_is_refused_never_skipped(stores, tmp_path, kind):
    source, _target, migration = migrated(stores, tmp_path, kind)
    mark = migration.copy()
    churn(source, 0, 3)
    entries = source.outbox("t", since=mark)
    later = next(e for e in entries[1:] if e["kind"] == "insert")
    migration.resume_from(later["id"] - 1)  # the entries before it were lost
    with pytest.raises(ContractError, match="migration_event_gap"):
        migration.catch_up()


def test_a_non_empty_target_tenant_is_refused(stores, tmp_path):
    _source, target, migration = migrated(stores, tmp_path, "sqlite")
    seed(target)
    with pytest.raises(ValueError, match="import_target_tenant_not_empty"):
        migration.copy()


@pytest.mark.parametrize("kind", TARGETS)
def test_the_switch_verifies_and_moves_writes_to_the_target(stores, tmp_path, kind):
    source, target, migration = migrated(stores, tmp_path, kind)
    pointer = DeploymentPointer(tmp_path / "pointer.json")
    pointer.initialize("A")
    states = {"A": source, "B": target}
    migration.copy()
    churn(source, 0, 6)
    report = switch(pointer, migration, "B", spaces=[FP])
    assert report["record"] == pointer.read() and report["record"]["generation"] == 2
    assert report["final_catch_up"]["applied"] > 0 and all(report["checks"].values())
    frozen = source.generations("t")
    _, record = write_active(pointer, states, lambda s: s.revoke("t", "p1"))
    assert record["deployment"] == "B" and "p1" in target.revoked("t")
    assert source.generations("t") == frozen and "p1" not in source.revoked("t")


@pytest.mark.parametrize("kind", TARGETS)
def test_a_failed_verification_leaves_the_pointer_where_it_was(stores, tmp_path, kind):
    _source, target, migration = migrated(stores, tmp_path, kind)
    pointer = DeploymentPointer(tmp_path / "pointer.json")
    pointer.initialize("A")
    migration.copy()
    target.set_access("t", "b03", "alice", "deny")  # a write that bypassed the pointer
    with pytest.raises(ContractError, match="migration_verification_failed:acl_epoch_equal"):
        switch(pointer, migration, "B", spaces=[FP])
    assert pointer.read()["generation"] == 1 and pointer.read()["deployment"] == "A"


def test_a_failed_shadow_comparison_leaves_the_pointer_where_it_was(stores, tmp_path):
    _, _, migration = migrated(stores, tmp_path, "sqlite")
    pointer = DeploymentPointer(tmp_path / "pointer.json")
    pointer.initialize("A")
    migration.copy()
    with pytest.raises(ContractError, match="migration_verification_failed:shadow_equal"):
        switch(pointer, migration, "B", spaces=[FP], shadow=lambda: False)
    assert pointer.read()["generation"] == 1


def test_a_stale_switch_loses_to_the_one_that_went_first(tmp_path):
    pointer = DeploymentPointer(tmp_path / "pointer.json")
    pointer.initialize("A")
    seen = pointer.read()["generation"]
    with pointer.exclusive():
        pointer.switch("B", expect_generation=seen)
    with pointer.exclusive(), pytest.raises(ContractError, match="generation_conflict"):
        pointer.switch("C", expect_generation=seen)
    assert pointer.read()["deployment"] == "B"


@pytest.mark.parametrize("kind", TARGETS)
def test_a_fence_pinned_before_the_switch_refuses_after_it(stores, tmp_path, kind):
    source, target, migration = migrated(stores, tmp_path, kind)
    pointer = DeploymentPointer(tmp_path / "pointer.json")
    pointer.initialize("A")
    deployments = {"A": local_storage(source), "B": local_storage(target)}
    before, record = pin(pointer, deployments)
    assert record["deployment"] == "A"
    assert before.release.authorize("t", {"p1"}, n3.ALICE).ok
    migration.copy()
    switch(pointer, migration, "B", spaces=[FP])
    target.revoke("t", "p1")  # the old deployment never learns of this
    refused = before.release.authorize("t", {"p1"}, n3.ALICE)
    assert not refused.ok and refused.reason == "deployment_changed"
    after, record = pin(pointer, deployments)
    assert record["deployment"] == "B" and isinstance(after.release, PinnedRelease)
    assert after.release.authorize("t", {"p1"}, n3.ALICE).reason == "revoked_before_release"
    assert after.release.authorize("t", {"b01"}, n3.ALICE).ok


@pytest.mark.parametrize("mode", n3.MODES)
def test_an_answer_in_flight_across_the_switch_is_never_released(stores, tmp_path, mode):
    source, target, migration = migrated(stores, tmp_path, "sqlite")
    pointer = DeploymentPointer(tmp_path / "pointer.json")
    pointer.initialize("A")
    deployments = {"A": local_storage(source), "B": local_storage(target)}
    migration.copy()

    def move():  # inside the writer call: the switch, then a revocation on the new deployment
        switch(pointer, migration, "B", spaces=[FP])
        target.revoke("t", "p1")

    pinned, _ = pin(pointer, deployments)
    result = n3.ask(pinned, move, mode=mode, principal=n3.ALICE)
    assert not n3.released(result) and result["reason"] == "deployment_changed"
    fresh, record = pin(pointer, deployments)
    assert record["deployment"] == "B"
    again = n3.ask(fresh, mode=mode, principal=n3.ALICE)
    assert "p1" not in {u.split("@")[0] for u in again.get("evidence_ids") or []}


@pytest.mark.parametrize("kind", TARGETS)
def test_rollback_catches_the_old_deployment_up_and_switches_back(stores, tmp_path, kind):
    source, target, migration = migrated(stores, tmp_path, kind)
    pointer = DeploymentPointer(tmp_path / "pointer.json")
    pointer.initialize("A")
    migration.copy()
    churn(source, 0, 4)
    forward = switch(pointer, migration, "B", spaces=[FP])
    churn(target, 4, 9)  # writes on the new deployment after the switch
    back = Migration(target, source, "t", tmp_path / "rollback.json")
    back.resume_from(forward["record"]["target_outbox_mark"])
    report = switch(pointer, back, "A", spaces=[FP])
    assert report["record"]["generation"] == 3 and report["record"]["deployment"] == "A"
    assert report["final_catch_up"]["applied"] > 0 and all(report["checks"].values())
    assert source.events("t") == target.events("t")


@pytest.mark.parametrize("kind", TARGETS)
def test_a_switch_under_a_writer_in_another_process_loses_no_write(stores, tmp_path, kind):
    source, source_spec = stores("sqlite", "A")
    target, target_spec = stores(kind, "B")
    seed(source)
    pointer = DeploymentPointer(tmp_path / "pointer.json")
    pointer.initialize("A")
    migration = Migration(source, target, "t", tmp_path / "ledger.json")
    log_path = tmp_path / "writes.json"
    start_at = time.monotonic() + 0.5
    spawn = multiprocessing.get_context("spawn")
    done = spawn.Value("i", 0)
    # Events, not timing (0.1.0, parent docs/PUB_2_REPORT.md): the copy starts once 20 writes are
    # acknowledged on A, and the writer stops 25 writes after it sees the switch. Before, 400 writes
    # 1 ms apart outran the catch-up into PostgreSQL (it applies the outbox until it is empty), so
    # the switch came after the last write. 10 ms apart, the writer stays below the target's apply
    # rate and the catch-up ends while it writes.
    proc = spawn.Process(
        target=writer,
        args=(
            str(tmp_path / "pointer.json"),
            {"A": source_spec, "B": target_spec},
            20000,
            0.01,
            str(log_path),
            start_at,
        ),
        kwargs={"done": done, "after": 25},
    )
    proc.start()
    limit = time.monotonic() + 60
    while done.value < 20:
        assert proc.is_alive() and time.monotonic() < limit, "the writer did not start"
        time.sleep(0.005)
    migration.copy()
    migration.catch_up()
    report = switch(pointer, migration, "B", spaces=[FP])
    proc.join(120)
    assert proc.exitcode == 0
    log = json.loads(log_path.read_text())
    moved = [w for w in log if w["deployment"] == "B"]
    assert moved and len(moved) < len(log)  # the switch happened while the writer ran
    assert {w["generation"] for w in moved} == {2}
    frozen = source.generations("t")
    assert source.outbox_mark("t") == migration.state()["applied"]
    revoked, hidden = target.revoked("t"), target.hidden("t", n3.ALICE)
    for w in log:  # every acknowledged write is in the deployment that is now active
        if w["kind"] == "insert":
            assert target.bodies("t", [(w["document"], 1)])
        elif w["kind"] == "revoke":
            assert w["document"] in revoked
        elif w["kind"] == "acl":
            assert w["document"] in hidden
    last = {}
    for w in log:
        if w["kind"] == "vector":
            last[w["text"]] = w["vector"]
    for text, vector in last.items():
        assert np.array_equal(target.vector("t", FP, text), vector)
    assert source.generations("t") == frozen  # nothing reached the old deployment afterwards
    assert report["writes_paused_seconds"] < 5


@pytest.mark.parametrize("kind", TARGETS)
def test_numbers_with_gaps_from_a_file_shared_by_tenants_are_kept(stores, tmp_path, kind):
    source, target, migration = migrated(stores, tmp_path, kind)
    source.upsert("u", n3.doc("u0", "Another tenant's note."))  # SQLite numbers per file
    migration.copy()
    for k in range(6):
        churn(source, k, 1)
        source.upsert("u", n3.doc(f"u{k + 1}", f"Another tenant's note {k}."))
    migration.catch_up()
    numbers = [e["seq"] for e in source.events("t")]
    assert numbers != list(range(1, len(numbers) + 1))  # gaps: the other tenant's events
    assert [e["seq"] for e in target.events("t")] == numbers
    assert all(migration.verify([FP]).values())


def test_a_sqlite_target_refuses_numbers_another_tenant_of_its_file_holds(stores, tmp_path):
    _, target, migration = migrated(stores, tmp_path, "sqlite")
    target.upsert("u", n3.doc("u1", "Another tenant's note."))  # event number 1 of the file
    with pytest.raises(ContractError, match="import_event_numbers_taken"):
        migration.copy()


@pytest.mark.parametrize("full", [True, False])
def test_a_vector_that_differs_fails_both_verifications(stores, tmp_path, full):
    _, target, migration = migrated(stores, tmp_path, "sqlite")
    pointer = DeploymentPointer(tmp_path / "pointer.json")
    pointer.initialize("A")
    migration.copy()
    key = target.vector_checksums("t", FP)[0][0]
    target.put_vector_blob("t", FP, key, np.arange(n3.DIMS, dtype=float))
    with pytest.raises(ContractError, match="vectors_equal"):
        switch(pointer, migration, "B", spaces=[FP], full=full)
    assert pointer.read()["generation"] == 1
