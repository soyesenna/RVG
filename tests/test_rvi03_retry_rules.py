"""RVI-03 F3 (docs/rvi/RVI03_F3_PROTOCOL.md in the parent repository): the scoped retry rules —
`retry_rule` "slack" (time to spare and one related document), "cited" (a cited document
changed) — and `revalidate` (R-c: the first draft released again when every input it consumed
reads back byte-identical). Events fire from inside the (fake) writer call, as in
`test_rvi03_scoped_fence`; the arms sit on `release.vectors = scoped` (F2).

The PostgreSQL cases need `DEEPJEV_TEST_POSTGRES_DSN`; without it they are skipped (not passes)."""

import asyncio
import dataclasses
import os
import re
import subprocess
import sys

import httpx
import numpy as np
import pytest
from test_rvi03_scoped_fence import (
    PG,
    QUESTION,
    SPACE,
    Q,
    World,
    doc,
)

from rvg.runtime import CallContext
from rvg.temporal.access import TrustedPrincipal
from rvg.temporal.models import TemporalQuestion
from rvg.temporal.retriever import TemporalRetriever
from rvg.testing.fake_openrouter import FakeOpenRouter, providers
from rvg.transport import JsonTransport
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

ALICE = TrustedPrincipal("alice", groups=("staff",))
BASE = {"mode": "scoped_dependency", "vectors": "scoped"}
ANY = Arm.of("f3-any", release={**BASE, "retry": 1})
SLACK = Arm.of("f3-slack", release={**BASE, "retry": 1, "retry_rule": "slack"})
CITED = Arm.of("f3-cited", release={**BASE, "retry": 1, "retry_rule": "cited"})
REVALIDATE = Arm.of("f3-revalidate", release={**BASE, "retry": 1, "revalidate": True})


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


class Searches(TemporalRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    async def search(self, *args, **kwargs):
        docs, retrieval = await super().search(*args, **kwargs)
        self.seen.append(
            (tuple(retrieval.get("seed_ids") or ()), tuple(sorted(d.key for d in docs)))
        )
        return docs, retrieval


def ask(w, hook=lambda: None, *, arm, context=None, storage=None):
    """One answer; `hook` runs inside the first writer call. Returns (result, writer calls,
    searches seen)."""
    calls = []

    def write(payload):
        calls.append(True)
        if len(calls) == 1:
            hook()
        for unit in payload["evidence"]:
            found = re.search(r"\d+", unit["text"])
            if found:
                return {
                    "answerable": True,
                    "short_answer": found.group(),
                    "sentences": [{"text": unit["text"], "evidence_ids": [unit["id"]]}],
                }
        return {"answerable": False, "short_answer": "", "sentences": []}

    async def go():
        fake = FakeOpenRouter(writer=write)
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(JsonTransport(client, timeout=5, max_attempts=2))
            retriever = Searches(storage or w.storage, _Fixed(), principal=ALICE)
            engine = UniversalEngine(
                retriever=retriever,
                decision=p["decision"],
                writer=p["writer"],
                extractor=p["extractor"],
                arm=arm,
            )
            q = TemporalQuestion(
                text=QUESTION, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
            )
            ctx = context or CallContext.with_timeout(
                60,
                max_decision_calls=400,
                max_generation_calls=24,
                max_embedding_calls=40,
                max_semantic_proposer_calls=16,
                max_surface_proposer_calls=8,
                max_http_attempts=800,
            )
            return await engine.answer("t", q, ctx), len(calls), retriever.seen

    return asyncio.run(go())


class _Fixed:
    mode = "fixture"

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        from rvg.contracts import EmbeddingBatch

        return EmbeddingBatch(tuple(tuple(Q) for _ in texts), SPACE, ("f",), ())


def revision(w, identifier, value, family, n=2):
    return lambda: w.put(
        doc(identifier, f"The budget of project P is {value}.", family=family, revision=n, day=15),
        Q + 0.1 * np.eye(8)[1],
    )


def clocked(seconds_used):
    now = [1000.0]
    ctx = CallContext.with_timeout(
        60,
        clock=lambda: now[0],
        max_decision_calls=400,
        max_generation_calls=24,
        max_embedding_calls=40,
        max_semantic_proposer_calls=16,
        max_surface_proposer_calls=8,
        max_http_attempts=800,
    )
    return ctx, now


# --- arms ------------------------------------------------------------------------------------


def test_the_rules_are_opt_ins_of_a_scoped_retry_only():
    for bad in (
        {"mode": "scoped_dependency", "retry_rule": "slack"},  # no retry
        {"mode": "current", "retry": 1, "retry_rule": "cited"},
        {"mode": "current", "retry": 1, "revalidate": True},
        {"mode": "scoped_dependency", "retry": 1, "retry_rule": "often"},
        {"mode": "scoped_dependency", "retry": 1, "revalidate": "yes"},
    ):
        with pytest.raises(ValueError):
            Arm.of("bad", release=bad)
    assert "retry_rule" not in dict(ANY.release) and "revalidate" not in dict(ANY.release)


# --- R0 and R-a ------------------------------------------------------------------------------


def test_any_regenerates_on_a_related_revision(world):
    result, calls, _ = ask(world, revision(world, "p1", 650, "P"), arm=ANY)
    assert result["short_answer"] == "650" and result["retried_after_change"] is True
    assert calls == 2


def test_slack_regenerates_one_related_document_with_time_to_spare(world):
    result, _, _ = ask(world, revision(world, "p1", 650, "P"), arm=SLACK)
    assert result["short_answer"] == "650" and result["retried_after_change"] is True


def test_slack_holds_when_two_related_documents_changed(world):
    def two():
        revision(world, "p1", 650, "P")()
        world.put(
            doc("s1", "Project P overview, revised.", family="P", revision=2, day=15),
            -Q + np.eye(8)[2],
        )

    result, calls, _ = ask(world, two, arm=SLACK)
    assert result["reason"] == "scope_changed" and result["retry_declined"] == "slack:changes"
    assert calls == 1


def test_slack_holds_without_one_and_a_half_times_the_first_attempt_left():
    w = World()
    ctx, now = clocked(0)

    def slow():
        now[0] += 30  # 30 of 60 seconds used: any would retry (30 >= 30), slack needs 45
        revision(w, "p1", 650, "P")()

    result, _, _ = ask(w, slow, arm=SLACK, context=ctx)
    assert result["retry_declined"] == "slack:deadline"
    ctx, now = clocked(0)
    w2 = World()

    def slow2():
        now[0] += 30
        revision(w2, "p1", 650, "P")()

    again, _, _ = ask(w2, slow2, arm=ANY, context=ctx)
    assert again["retried_after_change"] is True


# --- R-b ---------------------------------------------------------------------------------------


def test_cited_regenerates_when_the_cited_document_changed(world):
    result, _, _ = ask(world, revision(world, "p1", 650, "P"), arm=CITED)
    assert result["short_answer"] == "650" and result["retried_after_change"] is True


def test_cited_regenerates_when_the_cited_document_is_declared_replaced(world):
    declared = lambda: world.put(
        doc("rp", "This notice replaces the note.", family="RP", day=15, supersedes=["p1"]), -Q
    )
    result, calls, _ = ask(world, declared, arm=CITED)
    assert result.get("retried_after_change") is True and calls >= 1
    assert "retry_declined" not in result


def test_cited_holds_when_only_an_uncited_read_document_changed(world):
    _, v = stored(world, "b00")
    change = lambda: world.put(
        doc("b00", "Background remark on weather alpha, revised.", revision=2, day=15), v
    )
    result, calls, _ = ask(world, change, arm=CITED)
    assert result["reason"] == "scope_changed" and result["retry_declined"] == "cited:untouched"
    assert calls == 1


# --- R-c ---------------------------------------------------------------------------------------


def test_revalidation_releases_the_first_draft_when_the_inputs_read_back_identical(world):
    d, v = stored(world, "p1")
    result, calls, seen = ask(
        world,
        lambda: world.storage.blobs.put_vector("t", SPACE.fingerprint, text(d), v + 1e-3),
        arm=REVALIDATE,
    )
    assert result["status"] == "FULL" and result["short_answer"] == "500"
    assert result["revalidated_after_change"] is True and result["release"]["revalidated"] is True
    assert result["first_attempt"]["reason"] == "scope_changed"
    assert calls == 1  # no regeneration
    assert seen[0] == seen[1]  # the revalidation read the first attempt's search back
    fresh, _, fresh_seen = ask(world, arm=REVALIDATE)
    assert fresh["evidence_ids"] == result["evidence_ids"] and fresh_seen[0] == seen[1]


def test_revalidation_regenerates_when_an_input_differs(world):
    result, calls, _ = ask(world, revision(world, "p1", 650, "P"), arm=REVALIDATE)
    assert result["short_answer"] == "650" and result["retried_after_change"] is True
    assert result["revalidation"] == "inputs_differ" and calls == 2


def test_revalidation_regenerates_when_the_acl_epoch_moved(world):
    change = lambda: world.store.replace_access("t", "b00", {"bob": "deny"})
    result, calls, _ = ask(world, change, arm=REVALIDATE)  # a read document's ACL: RELATED
    assert result.get("revalidation") == "acl_epoch_moved"
    assert result["retried_after_change"] is True and calls == 2


def test_revalidation_regenerates_when_a_related_change_lands_during_it(world):
    d, v = stored(world, "p1")
    hits = []

    class Late:
        """The release coordinator: before the revalidation's check, p1's vector moves again."""

        def __init__(self, inner):
            self.inner = inner

        def authorize(self, tenant, document_ids, principal):
            hits.append(True)
            if len(hits) == 2:
                world.storage.blobs.put_vector("t", SPACE.fingerprint, text(d), v + 2e-3)
            return self.inner.authorize(tenant, document_ids, principal)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    storage = dataclasses.replace(world.storage, release=Late(world.storage.release))
    result, calls, _ = ask(
        world,
        lambda: world.storage.blobs.put_vector("t", SPACE.fingerprint, text(d), v + 1e-3),
        arm=REVALIDATE,
        storage=storage,
    )
    assert result["revalidation"] == "changed_again" and calls == 2


def test_revalidation_never_runs_for_the_other_rules(world):
    d, v = stored(world, "p1")
    result, calls, _ = ask(
        world,
        lambda: world.storage.blobs.put_vector("t", SPACE.fingerprint, text(d), v + 1e-3),
        arm=ANY,
    )
    assert "revalidated_after_change" not in result and calls == 2


# --- real PostgreSQL: another process moves a read document's vector --------------------------

MOVER = """
import os, sys
import numpy as np
import rvg
from rvg.temporal.retriever import TemporalRetriever
schema, root, space = sys.argv[1], sys.argv[2], sys.argv[3]
storage = rvg.open_storage(
    {"schema_version": 1,
     "state": {"backend": "postgres", "dsn_env": "RVG_TEST_PG", "schema_name": schema},
     "vectors": {"index": "local_exact", "matrix_dir": root}},
    root=root, environ={"RVG_TEST_PG": os.environ["RVG_TEST_PG"]})
d = next(x for x in storage.state.documents("t") if x.id == "p1")
text = TemporalRetriever.input_text(d)
v = np.asarray(storage.blobs.vector("t", space, text), dtype=float)
storage.blobs.put_vector("t", space, text, v + 1e-3)
storage.close()
"""


@pytest.mark.skipif(not PG, reason="no PostgreSQL")
def test_postgres_revalidation_after_another_process_moved_a_read_vector(tmp_path):
    w = World("postgres", tmp_path)
    try:
        env = {**os.environ, "RVG_TEST_PG": PG}
        cmd = [sys.executable, "-c", MOVER, w.schema, str(tmp_path / "w"), SPACE.fingerprint]
        result, calls, _ = ask(
            w, lambda: subprocess.run(cmd, env=env, check=True, timeout=60), arm=REVALIDATE
        )
        assert result["revalidated_after_change"] is True and result["short_answer"] == "500"
        assert calls == 1
    finally:
        w.cleanup()


# --- the guarantee, on random sequences: a released answer equals a fresh run ---------------------


def _shift(w, identifier, delta):
    d, v = stored(w, identifier)
    w.storage.blobs.put_vector("t", SPACE.fingerprint, text(d), v + delta)


POOL = [
    ("p1_vector", lambda w, n: _shift(w, "p1", 1e-3 * (n + 1))),
    ("far_vector", lambda w, n: _shift(w, "far", 1e-3 * (n + 1))),
    ("background_vector", lambda w, n: _shift(w, f"b{n % 25:02d}", 1e-3 * (n + 1))),
    ("p1_revision", lambda w, n: revision(w, "p1", 650 + n, "P", n=2 + n)()),
    ("far_insert", lambda w, n: w.put(doc(f"x{n}", "Another far note on ships.", day=15), -Q)),
    ("far_acl", lambda w, n: w.store.replace_access("t", "far", {f"u{n}": "deny"})),
]


@pytest.mark.parametrize("arm", [REVALIDATE, SLACK, CITED], ids=["revalidate", "slack", "cited"])
@pytest.mark.parametrize("seed", range(16))
def test_a_released_answer_equals_a_fresh_run(arm, seed):
    rng = np.random.default_rng(300 + seed)
    w = World()
    picks = [POOL[i] for i in rng.choice(len(POOL), size=int(rng.integers(1, 3)), replace=False)]

    def fire():
        for n, (_, event) in enumerate(picks):
            event(w, n)

    result, _, seen = ask(w, fire, arm=arm)
    if result["status"] in ("FULL", "PARTIAL"):
        fresh, _, fresh_seen = ask(w, arm=arm)
        assert fresh["short_answer"] == result["short_answer"]
        assert fresh["evidence_ids"] == result["evidence_ids"]
        assert fresh_seen[0] == seen[-1]
    else:
        assert result["reason"] in ("scope_changed", "source_changed", "stale_after_retry")
