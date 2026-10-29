"""RVI-03 F2 (docs/rvi/RVI03_F2_PROTOCOL.md in the parent repository): `release.vectors = scoped`
places vector writes (a pending row's vector arriving, a stored vector replaced) by relatedness.
Events fire from inside the (fake) writer call, as in `test_rvi03_scoped_fence`. With
`release.vectors = ignore` (the default) the scoped fence is unchanged.

The PostgreSQL cases need `DEEPJEV_TEST_POSTGRES_DSN`; without it they are skipped (not passes)."""

import asyncio
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from test_rvi03_scoped_fence import (
    PG,
    QUESTION,
    SCOPED,
    SPACE,
    Q,
    World,
    ask,
    at_score,
    doc,
    reasons,
    released,
)

from rvg.temporal.models import TemporalQuestion
from rvg.temporal.retriever import TemporalRetriever
from rvg.universal import scoped
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

VECTORS = Arm.of("rvi03-f2-vectors", release={"mode": "scoped_dependency", "vectors": "scoped"})
OTHER_SPACE = "f" * 64


@pytest.fixture(params=["sqlite", "postgres"])
def world(request, tmp_path):
    if request.param == "postgres" and not PG:
        pytest.skip("no PostgreSQL")
    w = World(request.param, tmp_path)
    yield w
    w.cleanup()


def text(d):
    return TemporalRetriever.input_text(d)


def stored(w, identifier):
    d = next(x for x in w.store.documents("t") if x.id == identifier)
    return d, np.asarray(w.storage.blobs.vector("t", SPACE.fingerprint, text(d)), dtype=float)


def put_vector(w, d, vector, space=None):
    w.storage.blobs.put_vector("t", space or SPACE.fingerprint, text(d), vector)


def ranked(w):
    """(id, vector) of every revision with a vector, by cosine with the question, highest first."""
    out = []
    for d in w.store.documents("t"):
        v = w.storage.blobs.vector("t", SPACE.fingerprint, text(d))
        if v is not None:
            v = np.asarray(v, dtype=float)
            out.append((float(v @ Q / np.linalg.norm(v)), d.id, v))
    out.sort(key=lambda r: -r[0])
    return [(i, v) for _, i, v in out]


PENDING_COPY = "The budget of project P is 500 (copy)."


def pending(w, identifier="pp", body=PENDING_COPY, access=None):
    """A new revision written without a vector (the pending path), before the question."""
    d = doc(identifier, body, family=identifier.upper(), day=2)
    if access is None:
        w.put(d)
    else:
        w.store.commit_event("t", document=d, access=access, keep_access=False)
    return d


# --- arm and defaults ------------------------------------------------------------------------


def test_the_vector_scope_is_an_opt_in_of_the_scoped_mode_only():
    assert "vectors" not in dict(SCOPED.release)  # older arms keep their digest
    for bad in (
        {"mode": "current", "vectors": "scoped"},
        {"mode": "latest_at_release", "vectors": "scoped"},
        {"mode": "scoped_dependency", "vectors": "all"},
    ):
        with pytest.raises(ValueError):
            Arm.of("bad", release=bad)
    assert Arm.of("ok", release={"mode": "current", "vectors": "ignore"})


def test_ignore_keeps_the_known_gap_a_pending_arrival_is_released_with_other_inputs():
    """D-P2 as recorded by S6: without the vector scope a pending row's vector arriving during
    the answer changes nothing at the fence, and a fresh run reads other seeds."""
    w = World()
    d = pending(w)
    result, seen = ask(w, lambda: put_vector(w, d, Q))
    assert released(result) and result["release"]["scope"]["checked"] == 0
    _, fresh_seen = ask(w)
    assert fresh_seen != seen


def test_no_vector_write_releases_with_the_mark_in_the_receipt(world):
    result, _ = ask(world, arm=VECTORS)
    assert result["status"] == "FULL" and result["short_answer"] == "500"
    receipt = result["release"]["scope"]
    assert receipt["status"] == "CLEAR" and receipt["checked"] == 0
    assert isinstance(receipt["vector_mark"], int)


# --- related vector writes withhold ------------------------------------------------------------


def related(w, kind):
    if kind == "pending_read":  # VP1: the arriving row was read through the BM25 fallback
        d = pending(w)
        return lambda: put_vector(w, d, Q)
    if kind == "pending_read_far":  # VPF: its vector lands far below the window, still read
        d = pending(w)
        return lambda: put_vector(w, d, -Q)
    if kind == "read_replaced":  # VR1: a read document's vector replaced (N3L L9 on p1)
        d, v = stored(w, "p1")
        return lambda: put_vector(w, d, v + 1e-3)
    if kind == "into_window":  # VW15: a far note takes the vector of a top-20 document
        d, _ = stored(w, "far")
        _, v = ranked(w)[5]
        return lambda: put_vector(w, d, v)
    if kind == "tie":  # VW20: a far note ties the 20th score
        d, _ = stored(w, "far")
        _, v = ranked(w)[19]
        return lambda: put_vector(w, d, v)
    raise ValueError(kind)


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("pending_read", "vector:RELATED:scope:document"),
        ("pending_read_far", "vector:RELATED:scope:document"),
        ("read_replaced", "vector:RELATED:scope:document"),
        ("into_window", "vector:RELATED:watermark"),
        ("tie", "vector:RELATED:watermark"),
    ],
)
def test_a_related_vector_write_withholds(world, kind, reason):
    result, _ = ask(world, related(world, kind), arm=VECTORS)
    assert result["status"] == "ABSTAIN" and result["reason"] == "scope_changed"
    assert reason in reasons(result)


# --- unrelated vector writes: released, and equal to a fresh run ------------------------------


def unrelated(w, kind):
    if kind == "far_replaced":  # VU: N3L L9 on a far note
        d, v = stored(w, "far")
        return lambda: put_vector(w, d, v + 1e-3)
    if kind == "below_window":  # VW25: a far note takes a vector under the 20th score
        d, _ = stored(w, "far")
        scores = w.scores()
        return lambda: put_vector(w, d, at_score((scores[21] + scores[22]) / 2))
    if kind == "hidden_pending":  # VH: the principal cannot retrieve the arriving row
        d = pending(w, "hp", access={"auditors": "allow"})
        return lambda: put_vector(w, d, Q)
    if kind == "other_space":  # VS
        d, _ = stored(w, "far")
        return lambda: put_vector(w, d, Q, space=OTHER_SPACE)
    if kind == "no_row":  # an input no catalog row has
        return lambda: w.storage.blobs.put_vector("t", SPACE.fingerprint, "no such row", Q)
    raise ValueError(kind)


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("far_replaced", "vector:CLEAR:below_watermark"),
        ("below_window", "vector:CLEAR:below_watermark"),
        ("hidden_pending", "vector:CLEAR:not_retrievable"),
        ("other_space", "vector:CLEAR:other_space"),
        ("no_row", "vector:CLEAR:no_row"),
    ],
)
def test_an_unrelated_vector_write_is_released_as_a_fresh_run_would_answer(world, kind, reason):
    result, seen = ask(world, unrelated(world, kind), arm=VECTORS)
    assert result["status"] == "FULL" and result["short_answer"] == "500"
    assert reason in reasons(result) and result["release"]["scope"]["status"] == "CLEAR"
    fresh, fresh_seen = ask(world, arm=VECTORS)
    assert fresh["evidence_ids"] == result["evidence_ids"] and fresh_seen == seen


def test_an_insert_still_follows_the_insert_rule_and_its_vector_row_is_placed_alike(world):
    """An insert writes its vector too (an outbox `vector` row); a far one stays CLEAR."""
    result, _ = ask(
        world, lambda: world.put(doc("zz", "Another far note on ships.", day=15), -Q), arm=VECTORS
    )
    assert result["status"] == "FULL"
    assert {"insert:CLEAR:below_watermark", "vector:CLEAR:below_watermark"} <= reasons(result)


def test_a_vector_write_while_the_check_reads_them_is_unmapped(monkeypatch):
    w = World()
    original = UniversalEngine.fallback_changed

    async def during_the_check(self, *args, **kwargs):
        d, v = stored(w, "far")
        put_vector(w, d, v + 2e-3)
        return await original(self, *args, **kwargs)

    result, _ = ask(w, unrelated(w, "far_replaced"), arm=VECTORS)
    assert released(result)
    monkeypatch.setattr(UniversalEngine, "fallback_changed", during_the_check)
    result, _ = ask(w, unrelated(w, "far_replaced"), arm=VECTORS)
    assert result["reason"] == "source_changed"
    assert "vector:UNKNOWN:vector_moved_during_check" in reasons(result)


def test_a_store_without_an_outbox_maps_no_vector_write():
    items = scoped._vector_items(
        None, "t", scoped.VectorWrites(rows=None), ids=frozenset(), space=SPACE, placed=None
    )
    assert items == [("vector", scoped.UNKNOWN, "no_outbox")]


# --- the BM25 fallback, recomputed -------------------------------------------------------------


def test_the_fallback_is_compared_in_the_order_it_was_consumed():
    """A search that read rows without vectors: the engine recomputes its BM25 fallback over the
    rows still pending and compares it with what the search consumed, in order."""
    w = World()
    a = pending(w, "pa", "The budget of project P is 500 according to the pa note.")
    b = pending(w, "pb", "Project P budget pb note.")
    reads = []

    async def go(consumed, forbid_view=False):
        engine = SimpleNamespace(retriever=TemporalRetriever(w.storage, _Fixed()))
        question = TemporalQuestion(
            text=QUESTION, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
        )
        catalog = await w.storage.catalogs.get("t")
        _, retrieval = await engine.retriever.search("t", question, None, top_k=20)
        got = [r["key"] for r in retrieval["ranking"] if r["via"] == "lexical_pending"]
        ranking = [r for r in retrieval["ranking"] if r["via"] == "dense"]
        ranking += [{"key": k, "score": None, "via": "lexical_pending"} for k in consumed(got)]
        read = scoped.Read(
            QUESTION, 20, frozenset(), frozenset(), {**retrieval, "ranking": ranking}, Q
        )
        reads.append(got)
        auth = w.storage.release.authorize("t", set(), None)
        index, view = engine.retriever.storage.index, engine.retriever.storage.index.view
        if forbid_view:  # the check reads the blob store, never the index's view (a rebuild)

            async def no_view(*args, **kwargs):
                raise AssertionError("the release check rebuilt the index view")

            index.view = no_view
        try:
            return await UniversalEngine.fallback_changed(
                engine, "t", question, [read], catalog, auth
            )
        finally:
            index.view = view

    assert asyncio.run(go(lambda got: got)) is False
    assert asyncio.run(go(lambda got: got, forbid_view=True)) is False
    assert sorted(reads[0]) == sorted([a.key, b.key])
    assert asyncio.run(go(lambda got: list(reversed(got)))) is True
    assert asyncio.run(go(lambda got: got[:1])) is True


class _Fixed:
    mode = "fixture"

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        from rvg.contracts import EmbeddingBatch

        return EmbeddingBatch(tuple(tuple(Q) for _ in texts), SPACE, ("f",), ())


# --- the guarantee, on random sequences of vector writes and events ---------------------------

POOL = [
    ("far_replaced", lambda w, n: unrelated(w, "far_replaced")()),
    ("below_window", lambda w, n: unrelated(w, "below_window")()),
    ("other_space", lambda w, n: unrelated(w, "other_space")()),
    ("no_row", lambda w, n: w.storage.blobs.put_vector("t", SPACE.fingerprint, f"none {n}", Q)),
    ("insert_far", lambda w, n: w.put(doc(f"x{n}", "Another far note on ships.", day=15), -Q)),
    ("read_replaced", lambda w, n: related(w, "read_replaced")()),
    ("into_window", lambda w, n: related(w, "into_window")()),
    ("tie", lambda w, n: related(w, "tie")()),
    (
        "background_replaced",
        lambda w, n: put_vector(w, *_shifted(w, f"b{n % 25:02d}", 1e-3 * (n + 1))),
    ),
]


def arrive_read(w):
    d = pending(w)
    return lambda: put_vector(w, d, -Q)


def arrive_hidden(w):
    d = pending(w, "hp", access={"auditors": "allow"})
    return lambda: put_vector(w, d, Q)


PREPARED = [
    ("none", lambda w: lambda: None),
    ("pending_read", arrive_read),
    ("hidden_pending", arrive_hidden),
]


def _shifted(w, identifier, delta):
    d, v = stored(w, identifier)
    return d, v + delta


# unrelated writes drawn twice as often, so that released answers are common enough to compare
WEIGHTS = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1], dtype=float) / 14


@pytest.mark.parametrize("seed", range(48))
def test_a_released_answer_equals_a_fresh_run_at_the_fence_with_its_vectors(seed):
    rng = np.random.default_rng(100 + seed)
    w = World()
    _, prepare = PREPARED[int(rng.integers(len(PREPARED)))]
    first = prepare(w)
    size = int(rng.integers(1, 4))
    picks = [POOL[i] for i in rng.choice(len(POOL), size=size, replace=False, p=WEIGHTS)]

    def fire():
        first()
        for n, (_, event) in enumerate(picks):
            event(w, n)

    result, seen = ask(w, fire, arm=VECTORS)
    if released(result):
        fresh, fresh_seen = ask(w, arm=VECTORS)
        assert fresh["short_answer"] == result["short_answer"]
        assert fresh["evidence_ids"] == result["evidence_ids"]
        assert fresh_seen == seen
    else:
        assert result["reason"] in ("scope_changed", "source_changed")


# --- real PostgreSQL: another process writes vectors while the answer is made -------------------

WRITER = """
import os, sys, time
import numpy as np
import rvg
from rvg.temporal.retriever import TemporalRetriever
schema, root, mode, space = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
storage = rvg.open_storage(
    {"schema_version": 1,
     "state": {"backend": "postgres", "dsn_env": "RVG_TEST_PG", "schema_name": schema},
     "vectors": {"index": "local_exact", "matrix_dir": root}},
    root=root, environ={"RVG_TEST_PG": os.environ["RVG_TEST_PG"]})
state = storage.state
docs = {d.id: d for d in state.documents("t")}
texts = {i: TemporalRetriever.input_text(d) for i, d in docs.items()}
base = {i: np.asarray(storage.blobs.vector("t", space, t), dtype=float) for i, t in texts.items()}

def shift(identifier, delta):
    # from the stored vector each time (not cumulative): the note stays where it was ranked
    storage.blobs.put_vector("t", space, texts[identifier], base[identifier] + delta)

if mode == "related":
    shift("p1", 1e-3)
else:
    end, n = time.monotonic() + float(sys.argv[5]), 0
    while time.monotonic() < end:
        n += 1
        if n % 2:
            shift("far", 1e-3 * (n % 7 + 1))
        else:
            text = TemporalRetriever.input_text(docs["far"])
            storage.blobs.put_vector("t", "f" * 64, text, np.eye(8)[n % 8] + 0.5)
        time.sleep(0.05)
    print(n)
storage.close()
"""


def writer_process(w, tmp_path, *args):
    env = {**os.environ, "RVG_TEST_PG": PG}
    cmd = [sys.executable, "-c", WRITER, w.schema, str(tmp_path / "w"), args[0], SPACE.fingerprint]
    return [*cmd, *args[1:]], env


@pytest.mark.skipif(not PG, reason="no PostgreSQL")
def test_postgres_a_read_documents_vector_replaced_by_another_process_is_related(tmp_path):
    w = World("postgres", tmp_path)
    try:
        cmd, env = writer_process(w, tmp_path, "related")
        result, _ = ask(
            w, lambda: subprocess.run(cmd, env=env, check=True, timeout=60), arm=VECTORS
        )
        assert result["reason"] == "scope_changed"
        assert "vector:RELATED:scope:document" in reasons(result)
    finally:
        w.cleanup()


@pytest.mark.skipif(not PG, reason="no PostgreSQL")
def test_postgres_scoped_vectors_keep_releasing_under_an_unrelated_vector_writer(tmp_path):
    """Another process keeps replacing a far note's vector and writing vectors of another space.
    Answers keep being released with p1's value; a withheld one may only be a vector write that
    arrived while the check read them."""
    w = World("postgres", tmp_path)
    try:
        cmd, env = writer_process(w, tmp_path, "loop", "8")
        writer = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
        outcomes = []
        time.sleep(0.5)
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            result, _ = ask(w, arm=VECTORS)
            outcomes.append(result)
        written = int(writer.communicate(timeout=60)[0].strip() or 0)
    finally:
        w.cleanup()
    assert written > 0 and outcomes
    shown = [r for r in outcomes if released(r)]
    assert shown and all(r["short_answer"] == "500" for r in shown)
    assert any(r["release"]["scope"]["checked"] > 0 for r in shown)  # it saw the writer
    for r in outcomes:
        if not released(r):
            assert r["reason"] == "source_changed"
            assert {k for k in reasons(r) if ":CLEAR:" not in k} <= {
                "vector:UNKNOWN:vector_moved_during_check"
            }
