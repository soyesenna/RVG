"""FIX-V9-CHURN (docs/rvi/FIX_V9_CHURN_PROTOCOL.md in the parent repository):
`release.vector_moves = after_check` places the vector rows committed while the release check
reads — after its outbox mark — once the check has read everything: a row whose input the check
read is UNKNOWN (as F2's V9), any other row is the next event, ordered after the release. Events
fire from inside the (fake) writer call; the writes "during the check" are made from inside the
vector step, after its mark.

The PostgreSQL cases need `DEEPJEV_TEST_POSTGRES_DSN`; without it they are skipped (not passes)."""

import os
import subprocess
import sys
import time

import numpy as np
import pytest
from test_rvi03_scoped_fence import PG, SPACE, Q, World, ask, doc, reasons, released
from test_rvi03_vector_scope import put_vector, stored

from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

BASE = {"mode": "scoped_dependency", "vectors": "scoped"}
HOLD = Arm.of("v9-hold", release=BASE)
AFTER = Arm.of("v9-after-check", release={**BASE, "vector_moves": "after_check"})


@pytest.fixture(params=["sqlite", "postgres"])
def world(request, tmp_path):
    if request.param == "postgres" and not PG:
        pytest.skip("no PostgreSQL")
    w = World(request.param, tmp_path)
    yield w
    w.cleanup()


def during_the_check(monkeypatch, write):
    """`write` runs inside the vector step, after its outbox mark was read."""
    original = UniversalEngine.fallback_changed

    async def patched(self, *args, **kwargs):
        write()
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(UniversalEngine, "fallback_changed", patched)


def insert_far(w, name="zz"):
    return lambda: w.put(doc(name, "Another far note on ships.", day=15), -Q)


def shift(w, identifier, delta=2e-3):
    d, v = stored(w, identifier)
    put_vector(w, d, v + delta)


def test_after_check_is_an_opt_in_of_the_vector_scope():
    for bad in (
        {"mode": "scoped_dependency", "vector_moves": "after_check"},  # vectors ignored
        {"mode": "current", "vector_moves": "after_check"},
        {**BASE, "vector_moves": "later"},
    ):
        with pytest.raises(ValueError):
            Arm.of("bad", release=bad)
    assert "vector_moves" not in dict(HOLD.release)


def test_an_unrelated_row_during_the_check_is_the_next_event(world, monkeypatch):
    during_the_check(monkeypatch, lambda: shift(world, "far"))
    result, seen = ask(world, insert_far(world), arm=AFTER)
    assert result["status"] == "FULL" and result["short_answer"] == "500"
    assert "vector:CLEAR:after_check" in reasons(result)
    fresh, fresh_seen = ask(world, arm=AFTER)
    assert fresh["evidence_ids"] == result["evidence_ids"] and fresh_seen == seen


def test_hold_keeps_v9_for_the_same_row(world, monkeypatch):
    during_the_check(monkeypatch, lambda: shift(world, "far"))
    result, _ = ask(world, insert_far(world), arm=HOLD)
    assert result["reason"] == "source_changed"
    assert "vector:UNKNOWN:vector_moved_during_check" in reasons(result)


def test_a_row_whose_input_the_check_read_is_unknown(world, monkeypatch):
    """The inserted note's vector was read by the check (its watermark step); a new value for
    it after the mark may be what the check saw."""
    during_the_check(monkeypatch, lambda: shift(world, "zz"))
    result, _ = ask(world, insert_far(world), arm=AFTER)
    assert result["reason"] == "source_changed"
    assert "vector:UNKNOWN:vector_moved_during_check" in reasons(result)


def test_a_row_of_another_space_or_no_row_is_the_next_event(world, monkeypatch):
    def writes():
        d, _ = stored(world, "far")
        put_vector(world, d, Q, space="e" * 64)
        world.storage.blobs.put_vector("t", SPACE.fingerprint, "no such row", Q)

    during_the_check(monkeypatch, writes)
    result, _ = ask(world, insert_far(world), arm=AFTER)
    assert result["status"] == "FULL"
    assert reasons(result) >= {"vector:CLEAR:after_check"}


def test_a_related_vector_write_before_the_mark_is_still_held(world):
    d, v = stored(world, "p1")
    result, _ = ask(world, lambda: put_vector(world, d, v + 1e-3), arm=AFTER)
    assert result["reason"] == "scope_changed"
    assert "vector:RELATED:scope:document" in reasons(result)


# --- the guarantee: released answers equal a fresh run when the late rows are unrelated --------

LATE = [
    lambda w, n: shift(w, "far", 1e-3 * (n + 1)),
    lambda w, n: w.storage.blobs.put_vector("t", SPACE.fingerprint, f"none {n}", Q),
    lambda w, n: put_vector(w, stored(w, "far")[0], Q, space="e" * 64),
]


def lowest_background(w):
    """The background note ranked last: outside every top-20 window."""
    scores = []
    for d in w.store.documents("t"):
        if d.id.startswith("b"):
            v = np.asarray(w.storage.blobs.vector("t", SPACE.fingerprint, text_of(d)), dtype=float)
            scores.append((float(v @ Q / np.linalg.norm(v)), d.id))
    return min(scores)[1]


def text_of(d):
    from rvg.temporal.retriever import TemporalRetriever

    return TemporalRetriever.input_text(d)


EVENTS = [
    insert_far,
    lambda w: lambda: None,
    lambda w: lambda: None,
    lambda w: lambda: w.store.replace_access("t", "far", {"bob": "deny"}),
    lambda w: lambda: w.store.revoke("t", lowest_background(w)),
    lambda w: lambda: shift(w, "b05", 1e-3),  # a read note: related, held
]


@pytest.mark.parametrize("seed", range(24))
def test_a_release_with_late_unrelated_rows_equals_a_fresh_run(seed, monkeypatch):
    rng = np.random.default_rng(500 + seed)
    w = World()
    first = insert_far(w)  # a vector row before the mark: the vector step runs every time
    extra = EVENTS[1 + int(rng.integers(len(EVENTS) - 1))](w)
    late = [LATE[i] for i in rng.choice(len(LATE), size=int(rng.integers(1, 3)), replace=False)]
    during_the_check(monkeypatch, lambda: [f(w, n) for n, f in enumerate(late)])
    result, seen = ask(w, lambda: (first(), extra()), arm=AFTER)
    monkeypatch.undo()
    if released(result):
        fresh, fresh_seen = ask(w, arm=AFTER)
        assert fresh["short_answer"] == result["short_answer"]
        assert fresh["evidence_ids"] == result["evidence_ids"]
        assert fresh_seen == seen
    else:
        assert result["reason"] in ("scope_changed", "source_changed")


# --- real PostgreSQL: another process keeps inserting far notes with their vectors --------------

CHURN = """
import os, sys, time
import numpy as np
import rvg
from rvg.temporal.models import TemporalDocument
from rvg.temporal.retriever import TemporalRetriever
schema, root, space, seconds = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
storage = rvg.open_storage(
    {"schema_version": 1,
     "state": {"backend": "postgres", "dsn_env": "RVG_TEST_PG", "schema_name": schema},
     "vectors": {"index": "local_exact", "matrix_dir": root}},
    root=root, environ={"RVG_TEST_PG": os.environ["RVG_TEST_PG"]})
end, n = time.monotonic() + seconds, 0
far = np.zeros(8); far[0] = -1.0
while time.monotonic() < end:
    n += 1
    d = TemporalDocument.model_validate({
        "id": "c%d" % n, "family": "C%d" % n, "revision": 1, "title": "Note c%d" % n,
        "text": "Another far note on ships %d." % n, "source": "fixture://c%d" % n,
        "published_at": "2026-09-15T00:00:00Z", "observed_at": "2026-09-15T00:00:00Z",
        "units": [{"id": "u0", "start": 0, "end": len("Another far note on ships %d." % n)}]})
    storage.blobs.put_vector("t", space, TemporalRetriever.input_text(d), far)
    storage.state.upsert("t", d)
    time.sleep(0.02)
print(n)
storage.close()
"""


def churn(w, tmp_path, arm, seconds=6):
    env = {**os.environ, "RVG_TEST_PG": PG}
    cmd = [sys.executable, "-c", CHURN, w.schema, str(tmp_path / "w"), SPACE.fingerprint]
    writer = subprocess.Popen([*cmd, str(seconds + 2)], env=env, stdout=subprocess.PIPE, text=True)
    outcomes = []
    time.sleep(0.5)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        outcomes.append(ask(w, insert_far(w, f"q{len(outcomes)}"), arm=arm)[0])
    written = int(writer.communicate(timeout=60)[0].strip() or 0)
    return outcomes, written


@pytest.mark.skipif(not PG, reason="no PostgreSQL")
def test_postgres_after_check_keeps_releasing_under_vector_churn(tmp_path):
    w = World("postgres", tmp_path)
    try:
        outcomes, written = churn(w, tmp_path, AFTER)
    finally:
        w.cleanup()
    assert written > 0 and outcomes
    shown = [r for r in outcomes if released(r)]
    assert shown and all(r["short_answer"] == "500" for r in shown)
    for r in outcomes:
        if not released(r):
            assert r["reason"] == "source_changed"
            assert {k for k in reasons(r) if ":CLEAR:" not in k} <= {
                "vector:UNKNOWN:vector_moved_during_check",
                "insert:UNKNOWN:pending_vector",
                "acl:UNKNOWN:acl_moved_during_check",
            }
    assert len(shown) / len(outcomes) >= 0.9
