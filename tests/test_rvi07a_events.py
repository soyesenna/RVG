"""RVI-07A: one source event is one transaction (ACL, revision or revocation, outbox, event-id
ledger), the ledger keys events by (tenant, issuer, event_id) with a payload digest, and an ACL
that allows nobody is an explicit deny-all. Faults are injected at the database driver, just
before one statement, which is how a crash between two writes looks to the store."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import uuid

import pytest
from session_fixtures import A, context, document, fake_session

import rvg
from rvg.errors import ContractError
from rvg.storage.sqlite import SQLiteState
from rvg.temporal.access import DENY_ALL

PG = os.environ.get("DEEPJEV_TEST_POSTGRES_DSN")
STAFF = rvg.TrustedPrincipal("staff", ("employees",))
FINANCE = rvg.TrustedPrincipal("finance", ("employees", "finance-team"))


class FaultConnection:
    def __init__(self, real, fail_on, error, skip=0):
        self._real, self.fail_on, self.error, self.skip, self.fired = real, fail_on, error, skip, 0

    def execute(self, sql, *args, **kwargs):
        if not self.fired and self.fail_on in str(sql):
            if self.skip:
                self.skip -= 1
            else:
                self.fired += 1
                raise self.error("injected")
        return self._real.execute(sql, *args, **kwargs)

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._real, name)


def sqlite_state():
    return (
        SQLiteState(":memory:"),
        {
            "doc": "INSERT INTO temporal_documents",
            "outbox": "INSERT INTO storage_outbox",
            "ledger": "INSERT INTO event_ledger",
        },
        sqlite3.OperationalError,
    )


def postgres_state():
    import psycopg

    from rvg.storage.postgres import PostgresState

    state = PostgresState(PG, schema="rvi07a_" + uuid.uuid4().hex[:10])
    return (
        state,
        {
            "doc": "INSERT INTO documents",
            "outbox": "INSERT INTO outbox",
            "ledger": "INSERT INTO event_ledger",
        },
        psycopg.errors.QueryCanceled,
    )


STATES = [pytest.param(sqlite_state, id="sqlite")] + [
    pytest.param(
        postgres_state,
        id="postgres",
        marks=pytest.mark.skipif(not PG, reason="DEEPJEV_TEST_POSTGRES_DSN not set"),
    )
]


def snapshot_of(state, tenant):
    export = state.export_tenant(tenant)
    return {
        "documents": sorted((d["id"], d["revision"]) for d in export["documents"]),
        "acl": sorted((a["id"], a["principal"], a["effect"]) for a in export["acl"]),
        "revocations": sorted(r["id"] for r in export["revocations"]),
        "acl_epoch": export["acl_epoch"],
        "events": len(export["events"]),
        "outbox": len(state.outbox(tenant)) if hasattr(state, "outbox") else None,
        "ledger": state.ledger(tenant),
    }


def doc(identifier, revision=1, value="100"):
    return document(identifier, [f"The value of {identifier} is {value}."], revision=revision)


def ledger(event_id, digest, issuer="src", kind="revision"):
    return {"issuer": issuer, "event_id": event_id, "digest": digest, "kind": kind}


@pytest.mark.parametrize("make", STATES)
@pytest.mark.parametrize(
    "fail_on,skip",
    [("doc", 0), ("outbox", 1), ("ledger", 0)],
    ids=["F01-revision-row", "F02-revision-outbox", "F03-ledger-row"],
)
def test_a_fault_inside_an_event_leaves_nothing_written(make, fail_on, skip):
    state, sql, error = make()
    try:
        state.upsert("t", doc("d"))
        before = snapshot_of(state, "t")
        real = state.conn
        state.conn = proxy = FaultConnection(real, sql[fail_on], error, skip)
        with pytest.raises(error):
            state.commit_event(
                "t",
                document=doc("d", 2, "120"),
                access={"finance": "allow"},
                keep_access=False,
                ledger=ledger("e1", "x"),
            )
        state.conn = real
        assert proxy.fired == 1
        assert snapshot_of(state, "t") == before  # ACL, revision, outbox, ledger: all or nothing
        assert "d" not in state.hidden("t", STAFF)  # nobody lost access through a failed event
        # F04: the same event again, without a fault, is applied exactly once
        done = state.commit_event(
            "t",
            document=doc("d", 2, "120"),
            access={"finance": "allow"},
            keep_access=False,
            ledger=ledger("e1", "x"),
        )
        assert done["ledger"] == "RECORDED" and done["event_seq"] is not None
        after = snapshot_of(state, "t")
        assert after["documents"] == [("d", 1), ("d", 2)]
        assert after["acl"] == [("d", "finance", "allow")] and len(after["ledger"]) == 1
    finally:
        state.close()


@pytest.mark.parametrize("make", STATES)
def test_the_ledger_replays_the_same_event_and_refuses_another_payload(make):
    state, _, _ = make()
    try:
        first = state.commit_event("t", document=doc("d"), ledger=ledger("e1", "h1"))
        again = state.commit_event("t", document=doc("d"), ledger=ledger("e1", "h1"))
        assert first["ledger"] == "RECORDED" and again["ledger"] == "REPLAYED"
        assert again["event_seq"] == first["event_seq"] and state.snapshot("t") == 1
        before = snapshot_of(state, "t")
        with pytest.raises(ContractError, match="event_id_conflict"):
            state.commit_event(
                "t",
                target="d",
                access={"staff": "allow"},
                keep_access=False,
                ledger=ledger("e1", "h2", kind="access"),
            )
        assert snapshot_of(state, "t") == before
        # the key is (tenant, issuer, event_id): another tenant or issuer is another event
        other = state.commit_event("t2", document=doc("d"), ledger=ledger("e1", "h9"))
        issuer = state.commit_event(
            "t", document=doc("e"), ledger=ledger("e1", "h8", issuer="other-source")
        )
        assert other["ledger"] == issuer["ledger"] == "RECORDED"
        # revocations go through the same ledger
        r1 = state.commit_event("t", revoke="d", ledger=ledger("r1", "rv", kind="revoke"))
        r2 = state.commit_event("t", revoke="d", ledger=ledger("r1", "rv", kind="revoke"))
        assert r1["event_seq"] is not None and r2["ledger"] == "REPLAYED"
        with pytest.raises(ContractError, match="event_id_conflict"):
            state.commit_event("t", revoke="e", ledger=ledger("r1", "other", kind="revoke"))
        assert state.revoked("t") == {"d"}
    finally:
        state.close()


@pytest.mark.parametrize("make", STATES)
def test_concurrent_events_with_one_id_apply_once(make):
    state, _, _ = make()
    try:
        state.upsert("t", doc("d"))
        outcomes, lock = [], threading.Lock()

        def submit(who, digest):
            try:
                state.commit_event(
                    "t",
                    target="d",
                    access={who: "allow"},
                    keep_access=False,
                    ledger=ledger("same", digest, kind="access"),
                )
                result = "applied"
            except ContractError:
                result = "conflict"
            with lock:
                outcomes.append(result)

        threads = [
            threading.Thread(target=submit, args=("finance", "a")),
            threading.Thread(target=submit, args=("staff", "b")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(outcomes) == ["applied", "conflict"]
        assert len(snapshot_of(state, "t")["acl"]) == 1
    finally:
        state.close()


@pytest.mark.parametrize("make", STATES)
def test_deny_all_hides_the_document_from_everyone_and_the_fence_refuses_it(make):
    state, _, _ = make()
    try:
        state.upsert("t", doc("d"))
        state.commit_event("t", target="d", access={DENY_ALL: "allow"}, keep_access=False)
        for principal in (STAFF, FINANCE, rvg.TrustedPrincipal("auditor"), None):
            assert "d" in state.hidden("t", principal)
        assert not state.authorize_release("t", {"d"}, FINANCE).ok
        # clearing the ACL (an empty entry map) makes it visible to the tenant again
        state.commit_event("t", target="d", access={}, keep_access=False)
        assert "d" not in state.hidden("t", STAFF)
    finally:
        state.close()


def test_reserved_names_cannot_be_principals_groups_or_acl_entries():
    with pytest.raises(ValueError, match="reserved_principal_name"):
        rvg.TrustedPrincipal(DENY_ALL)
    with pytest.raises(ValueError, match="reserved_principal_name"):
        rvg.TrustedPrincipal("staff", (DENY_ALL,))
    with pytest.raises(ValueError):
        rvg.DocumentAccess(allow=(DENY_ALL,))


def run(coro):
    return asyncio.run(coro)


def test_the_session_reports_conflicts_replays_and_backend_failures_as_contract_outcomes():
    async def go():
        async with fake_session() as (session, _, store):
            ctx = context(issuer="hr-system")
            await session.ingest([A], ctx)
            ev = {
                "type": "access",
                "event_id": "acl-1",
                "document_id": "a",
                "access": {"allow": ["finance"]},
            }
            first = await session.apply_event(ev, ctx)
            replay = await session.apply_event(ev, ctx)
            with pytest.raises(rvg.OperationError) as conflict:
                await session.apply_event(ev | {"access": {"allow": ["staff"]}}, ctx)
            # the same id from another issuer is another event
            other = await session.apply_event(
                ev | {"access": {"allow": ["staff"]}}, context(issuer="crm")
            )
            # a backend failure inside an ingest item is its row's ERROR, and nothing is written
            real = store.conn
            store.conn = FaultConnection(
                real, "INSERT INTO temporal_documents", sqlite3.OperationalError
            )
            item = {"document": doc("z").model_dump(mode="json"), "access": {"allow": ["finance"]}}
            row = (await session.ingest([item], ctx))["results"][0]
            store.conn = real
            with_acl = [a for a in store.export_tenant("fixture")["acl"] if a["id"] == "z"]
            return first, replay, conflict.value, other, row, with_acl

    first, replay, conflict, other, row, with_acl = run(go())
    assert first["outcome"] == "APPLIED"
    assert replay["outcome"] == "DUPLICATE" and replay["reason"] == "event_id_replayed"
    assert conflict.code == "INVALID_INPUT" and conflict.native_code == "event_id_conflict"
    assert other["outcome"] == "APPLIED"
    assert row["outcome"] == "ERROR" and row["reason"] == "backend:OperationalError"
    assert with_acl == []


def test_events_without_an_id_keep_their_meaning():
    async def go():
        async with fake_session() as (session, _, store):
            ctx = context()
            a = await session.apply_event(
                {"type": "revision", "document": doc("n").model_dump(mode="json")}, ctx
            )
            b = await session.apply_event(
                {"type": "revision", "document": doc("n").model_dump(mode="json")}, ctx
            )
            return a, b, store.ledger("fixture")

    a, b, rows = run(go())
    assert a["outcome"] == "APPLIED" and b["outcome"] == "DUPLICATE"
    assert b["reason"] == "identical_revision_already_stored" and rows == []
