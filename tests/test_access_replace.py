"""SEP-04 D3 (G04-BACKEND): a document's whole ACL replaced in one transaction, on SQLite and
PostgreSQL — all or nothing, the same outbox entries `set_access` writes (so a catch-up replays
them), and no change when the ACL already is what is asked. PostgreSQL needs
`DEEPJEV_TEST_POSTGRES_DSN`; without it those cases are skipped, not passed."""

import os
import uuid

import pytest
from test_backend_conformance import doc

from rvg.storage.sqlite import SQLiteState
from rvg.temporal.access import TrustedPrincipal

DSN = os.environ.get("DEEPJEV_TEST_POSTGRES_DSN")
ALICE, BOB, CAROL = (TrustedPrincipal(n, groups=("staff",)) for n in ("alice", "bob", "carol"))


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param(
            "postgres",
            marks=pytest.mark.skipif(not DSN, reason="DEEPJEV_TEST_POSTGRES_DSN not set"),
        ),
    ]
)
def state(request, tmp_path):
    if request.param == "sqlite":
        made = SQLiteState(tmp_path / "s.sqlite3")
        yield made
        made.close()
        return
    import psycopg

    from rvg.storage.postgres import PostgresState

    schema = "acl_" + uuid.uuid4().hex[:12]
    made = PostgresState(DSN, schema=schema)
    yield made
    made.close()
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def test_replace_sets_the_whole_acl_and_moves_the_epoch_per_change(state):
    state.upsert("t", doc("d1"))
    first = state.replace_access("t", "d1", {"alice": "allow", "carol": "deny"})
    assert first == {"changed": 2, "acl_epoch": 2}
    assert state.hidden("t", BOB) == {"d1"}  # not allowed
    assert state.hidden("t", CAROL) == {"d1"}  # denied
    assert state.hidden("t", ALICE) == set()
    second = state.replace_access("t", "d1", {"bob": "allow"})
    assert second == {"changed": 3, "acl_epoch": 5}  # alice and carol removed, bob added
    assert state.hidden("t", ALICE) == {"d1"} and state.hidden("t", BOB) == set()
    cleared = state.replace_access("t", "d1", {})
    assert cleared == {"changed": 1, "acl_epoch": 6}
    assert state.hidden("t", None) == set()  # no entries: visible to the tenant again


def test_the_same_acl_again_changes_nothing(state):
    state.upsert("t", doc("d1"))
    state.replace_access("t", "d1", {"alice": "allow"})
    mark, generations = state.outbox_mark("t"), state.generations("t")
    assert state.replace_access("t", "d1", {"alice": "allow"}) == {"changed": 0, "acl_epoch": 1}
    assert state.outbox_mark("t") == mark and state.generations("t") == generations


def test_the_outbox_holds_set_access_entries_with_consecutive_epochs(state):
    state.replace_access("t", "d1", {"bob": "deny", "alice": "allow"})
    state.set_access("t", "d1", "carol", "allow")
    entries = [e for e in state.outbox("t") if e["kind"] == "acl"]
    assert [e["payload"] for e in entries] == [
        {"id": "d1", "principal": "alice", "effect": "allow", "epoch": 1},
        {"id": "d1", "principal": "bob", "effect": "deny", "epoch": 2},
        {"id": "d1", "principal": "carol", "effect": "allow", "epoch": 3},
    ]


def test_a_failure_part_way_leaves_the_old_acl(state, monkeypatch):
    state.replace_access("t", "d1", {"alice": "allow"})
    calls = {"n": 0}
    original = state._outbox

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected")
        return original(*args, **kwargs)

    monkeypatch.setattr(state, "_outbox", failing)
    with pytest.raises(RuntimeError):
        state.replace_access("t", "d1", {"bob": "allow", "carol": "allow"})
    monkeypatch.undo()
    assert state.hidden("t", ALICE) == set() and state.hidden("t", BOB) == {"d1"}
    assert state.acl_epoch("t") == 1  # nothing of the failed replacement remains


def test_invalid_entries_are_refused_before_any_change(state):
    for entries in ({"alice": "maybe"}, {"": "allow"}, {"  ": "deny"}):
        with pytest.raises(ValueError):
            state.replace_access("t", "d1", entries)
    assert state.acl_epoch("t") == 0
