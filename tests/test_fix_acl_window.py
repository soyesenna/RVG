"""FIX-ACL-WINDOW (docs/rvi/FIX_ACL_WINDOW_PROTOCOL.md in the parent repository):
`release.acl_moves = after_check` makes the scoped check read visibility only as of the fence (the
principal's hidden ids the release ticket carries) and places the ACL rows committed while it reads
once it has read everything: a row on a document or family the attempt read is UNKNOWN (as before),
any other row is the next event, ordered after the release. Events fire from inside the (fake)
writer call; the writes "during the check" are made after the fence, before the verdict.

The PostgreSQL cases need `DEEPJEV_TEST_POSTGRES_DSN`; without it they are skipped (not passes)."""

import os
import subprocess
import sys
import time

import numpy as np
import pytest
from test_rvi03_scoped_fence import ALICE, PG, Q, World, ask, doc, reasons, released

from rvg.temporal.access import Release
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

BASE = {"mode": "scoped_dependency"}
HOLD = Arm.of("acl-hold", release=BASE)
AFTER = Arm.of("acl-after-check", release={**BASE, "acl_moves": "after_check"})
VECTORS = {**BASE, "vectors": "scoped", "vector_moves": "after_check"}
AFTER_VECTORS = Arm.of("acl-after-check-vectors", release={**VECTORS, "acl_moves": "after_check"})


@pytest.fixture(params=["sqlite", "postgres"])
def world(request, tmp_path):
    if request.param == "postgres" and not PG:
        pytest.skip("no PostgreSQL")
    w = World(request.param, tmp_path)
    yield w
    w.cleanup()


def during_the_check(monkeypatch, write):
    """`write` runs after the fence (and the vector step), before the verdict reads anything."""
    original = UniversalEngine.scope_verdict

    def patched(self, *args, **kwargs):
        write()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(UniversalEngine, "scope_verdict", patched)


def read_background(w):
    """A background note the question's search returns (read, not cited)."""
    _, seen = ask(w, arm=AFTER)
    keys = [k for s in seen if s[0] != "plane" for k in s[1]]
    return next(k.split("@")[0] for k in keys if k.startswith("b"))


def test_acl_moves_is_an_opt_in_of_the_scoped_check():
    for bad in (
        {"mode": "current", "acl_moves": "after_check"},
        {"mode": "as_of_snapshot", "acl_moves": "after_check"},
        {**BASE, "acl_moves": "later"},
    ):
        with pytest.raises(ValueError):
            Arm.of("bad", release=bad)
    assert "acl_moves" not in dict(HOLD.release)
    Arm.of("ok", release={**VECTORS, "retry": 1, "retry_rule": "slack", "acl_moves": "hold"})


def test_the_release_ticket_carries_the_hidden_ids(world):
    world.store.replace_access("t", "far", {"alice": "deny"})
    found = world.storage.release.authorize("t", {"p1"}, ALICE)
    assert found.ok and found.hidden == frozenset(world.store.hidden("t", ALICE)) == {"far"}
    assert found == Release(True, None, found.revocation_seq, found.acl_epoch)
    blocked = world.storage.release.authorize("t", {"far"}, ALICE)
    assert not blocked.ok and blocked.hidden == {"far"}


@pytest.mark.parametrize(
    "kind",
    ["deny_far", "other_principal", "lift_far"],
)
def test_an_unrelated_acl_row_during_the_check_is_the_next_event(world, monkeypatch, kind):
    if kind == "lift_far":
        world.store.replace_access("t", "far", {"alice": "deny"})
    entries = {"deny_far": {"alice": "deny"}, "other_principal": {"bob": "deny"}, "lift_far": {}}
    during_the_check(monkeypatch, lambda: world.store.replace_access("t", "far", entries[kind]))
    result, seen = ask(world, arm=AFTER)
    monkeypatch.undo()
    assert result["status"] == "FULL" and result["short_answer"] == "500"
    assert "acl:CLEAR:after_check" in reasons(result)
    assert result["release"]["acl_epoch"] < world.store.acl_epoch("t")
    fresh, fresh_seen = ask(world, arm=AFTER)
    assert fresh["evidence_ids"] == result["evidence_ids"] and fresh_seen == seen


def test_hold_keeps_the_global_rule_for_the_same_row(world, monkeypatch):
    during_the_check(monkeypatch, lambda: world.store.replace_access("t", "far", {"bob": "deny"}))
    result, _ = ask(world, arm=HOLD)
    assert result["reason"] == "source_changed"
    assert "acl:UNKNOWN:acl_moved_during_check" in reasons(result)


@pytest.mark.parametrize("target", ["cited", "read", "family"])
def test_a_row_on_what_the_attempt_read_is_unknown(world, monkeypatch, target):
    identifier = {"cited": "p1", "family": "s1"}.get(target) or read_background(world)
    during_the_check(
        monkeypatch, lambda: world.store.replace_access("t", identifier, {"alice": "deny"})
    )
    result, _ = ask(world, arm=AFTER)
    assert result["reason"] == "source_changed"
    assert "acl:UNKNOWN:acl_moved_during_check" in reasons(result)


def test_the_check_reads_visibility_as_of_the_fence(world, monkeypatch):
    """A hidden alias of the answer becomes visible before the fence (related: above the
    watermark) and hidden again while the check reads. Read from the state, the check would see it
    hidden and clear it; read as of the fence it is related."""
    world.store.commit_event(
        "t",
        document=doc("al", "The budget of project P is 700.", family="AL", day=15),
        access={"alice": "deny"},
        keep_access=False,
    )
    world.storage.blobs.put_vector("t", world_space(), alias_text(), Q)
    during_the_check(monkeypatch, lambda: world.store.replace_access("t", "al", {"alice": "deny"}))
    result, _ = ask(world, lambda: world.store.replace_access("t", "al", {}), arm=AFTER)
    assert result["reason"] == "scope_changed"
    assert "acl:RELATED:watermark" in reasons(result)


def test_a_document_that_appears_during_the_check_is_the_next_event(world, monkeypatch):
    """The boundary of the rule (protocol §2): the hidden alias is shown only after the fence."""
    world.store.commit_event(
        "t",
        document=doc("al", "The budget of project P is 700.", family="AL", day=15),
        access={"alice": "deny"},
        keep_access=False,
    )
    world.storage.blobs.put_vector("t", world_space(), alias_text(), Q)
    during_the_check(monkeypatch, lambda: world.store.replace_access("t", "al", {}))
    after, _ = ask(world, arm=AFTER)
    assert after["status"] == "FULL" and "acl:CLEAR:after_check" in reasons(after)


def world_space():
    from test_rvi03_scoped_fence import SPACE

    return SPACE.fingerprint


def alias_text():
    from rvg.temporal.retriever import TemporalRetriever

    return TemporalRetriever.input_text(
        doc("al", "The budget of project P is 700.", family="AL", day=15)
    )


def test_an_acl_writer_without_outbox_rows_during_the_check_is_unknown(monkeypatch):
    w = World()
    w.store.replace_access("t", "far", {"carol": "deny"})  # the tenant's epoch row exists
    during_the_check(monkeypatch, lambda: w.bump_acl_without_outbox("far", "bob", "deny"))
    result, _ = ask(w, arm=AFTER)
    assert result["reason"] == "source_changed"
    assert "acl:UNKNOWN:unregistered_acl_writer" in reasons(result)


def test_a_store_without_the_hidden_ids_keeps_the_global_rule(monkeypatch):
    w = World()
    original = w.storage.release.authorize

    def bare(*args, **kwargs):
        found = original(*args, **kwargs)
        return Release(found.ok, found.reason, found.revocation_seq, found.acl_epoch, found.blocked)

    monkeypatch.setattr(w.storage.release, "authorize", bare)
    during_the_check(monkeypatch, lambda: w.store.replace_access("t", "far", {"bob": "deny"}))
    result, _ = ask(w, arm=AFTER)
    assert result["reason"] == "source_changed"
    assert "acl:UNKNOWN:acl_moved_during_check" in reasons(result)


def test_the_vector_step_reads_the_fence_visibility(world, monkeypatch):
    seen = []
    original = UniversalEngine.fallback_changed

    async def recording(self, *args, **kwargs):
        seen.append(kwargs.get("hidden"))
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(UniversalEngine, "fallback_changed", recording)
    world.store.replace_access("t", "far", {"alice": "deny"})
    insert = lambda: world.put(doc("zz", "Another far note on ships.", day=15), -Q)
    result, _ = ask(world, insert, arm=AFTER_VECTORS)
    assert released(result) and seen and seen[-1] == {"far"}
    seen.clear()
    ask(
        world,
        lambda: world.put(doc("zy", "A far note.", day=15), -Q),
        arm=Arm.of("v", release=VECTORS),
    )
    assert seen and seen[-1] is None


# --- the guarantee: released answers equal a fresh run when the late rows are unrelated --------

LATE = [
    lambda w, n: w.store.replace_access("t", "far", {f"u{n}": "deny"}),
    lambda w, n: w.store.replace_access("t", "far", {"alice": "deny"} if n % 2 else {}),
    lambda w, n: w.store.replace_access("t", "b24", {f"v{n}": "deny"}),
    lambda w, n: w.store.replace_access("t", "p1", {"alice": "deny"}),  # cited: held
]
FIRST = [
    lambda w: None,
    lambda w: w.put(doc("zz", "Another far note on ships.", day=15), -Q),
    lambda w: w.store.replace_access("t", "far", {"bob": "deny"}),
    lambda w: w.store.revoke("t", "b23"),
]


@pytest.mark.parametrize("seed", range(24))
def test_a_release_with_late_unrelated_acl_rows_equals_a_fresh_run(seed, monkeypatch):
    rng = np.random.default_rng(700 + seed)
    w = World()
    first = FIRST[int(rng.integers(len(FIRST)))]
    picks = rng.choice(len(LATE), size=int(rng.integers(1, 3)), replace=False)
    during_the_check(monkeypatch, lambda: [LATE[i](w, n) for n, i in enumerate(picks)])
    result, seen = ask(w, lambda: first(w), arm=AFTER)
    monkeypatch.undo()
    if 3 in picks:
        assert not released(result)
    if released(result):
        fresh, fresh_seen = ask(w, arm=AFTER)
        assert fresh["short_answer"] == result["short_answer"]
        assert fresh["evidence_ids"] == result["evidence_ids"]
        assert fresh_seen == seen
    else:
        assert result["reason"] in ("scope_changed", "source_changed")


# --- real PostgreSQL: another process keeps changing far documents' ACL -------------------------

CHURN = """
import os, sys, time
import rvg
schema, root, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
storage = rvg.open_storage(
    {"schema_version": 1,
     "state": {"backend": "postgres", "dsn_env": "RVG_TEST_PG", "schema_name": schema},
     "vectors": {"index": "local_exact", "matrix_dir": root}},
    root=root, environ={"RVG_TEST_PG": os.environ["RVG_TEST_PG"]})
end, n = time.monotonic() + seconds, 0
while time.monotonic() < end:
    n += 1
    entries = {"alice": "deny"} if n % 3 == 0 else {} if n % 3 == 1 else {"u%d" % n: "deny"}
    storage.state.replace_access("t", "far", entries)
    time.sleep(0.005)
print(n)
storage.close()
"""


def churn(w, tmp_path, arm, monkeypatch, seconds=6):
    """Fixture answers take milliseconds; the check is held open for 30 ms (as a real vector step
    reading the catalog does) so the other process's ACL rows land while it reads."""
    during_the_check(monkeypatch, lambda: time.sleep(0.03))
    env = {**os.environ, "RVG_TEST_PG": PG}
    cmd = [sys.executable, "-c", CHURN, w.schema, str(tmp_path / "w"), str(seconds + 2)]
    writer = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
    outcomes = []
    time.sleep(0.5)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        outcomes.append(ask(w, arm=arm)[0])
    written = int(writer.communicate(timeout=60)[0].strip() or 0)
    monkeypatch.undo()
    return outcomes, written


@pytest.mark.skipif(not PG, reason="no PostgreSQL")
def test_postgres_after_check_keeps_releasing_under_acl_churn(tmp_path, monkeypatch):
    w = World("postgres", tmp_path)
    try:
        outcomes, written = churn(w, tmp_path, AFTER, monkeypatch)
    finally:
        w.cleanup()
    assert written > 0 and outcomes
    assert sum("acl:CLEAR:after_check" in reasons(r) for r in outcomes) >= len(outcomes) // 2
    shown = [r for r in outcomes if released(r)]
    assert shown and all(r["short_answer"] == "500" for r in shown)
    for r in outcomes:
        if not released(r):
            assert r["reason"] == "source_changed"
            assert {k for k in reasons(r) if ":CLEAR:" not in k} <= {
                "acl:UNKNOWN:unregistered_acl_writer"
            }
    assert len(shown) / len(outcomes) >= 0.9
