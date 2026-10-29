"""Release-input consistency regressions (parent docs/rvi/RVI_ACL_DEFECT_PROTOCOL.md).

Two defects found by the RVI-03 local runner (parent docs/rvi/RVI03_LOCAL_REPORT.md §9):

- D-ACL: the tenant's ACL changes while an answer is made (a document outside the evidence
  becomes visible or hidden). `current` and `as_of_snapshot` compare only the knowledge number,
  `latest_at_release` only families and seeds, so the answer made before the change is released
  although its search and validity inputs no longer hold at the release instant. This is an
  input-consistency failure at release time, not a proven disclosure of an unauthorised body.
- D-PHANTOM: a new revision in another family declares a replacement of a document the answer
  read (hidden from the principal, or visible but far from the question). `latest_at_release`
  sees neither its family nor a changed search, and releases the replaced value.

Every rule below is absolute: a released answer must equal a fresh run at the state the release
check read (status, short answer, evidence, what every search returned and the validity plane),
whatever the baseline does. The checks never call the code under repair. PostgreSQL cases need
`DEEPJEV_TEST_POSTGRES_DSN`; without it they are skipped (not passes)."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
import uuid

import httpx
import numpy as np
import pytest

import rvg
from rvg.contracts import EmbeddingBatch, EmbeddingSpace
from rvg.runtime import CallContext
from rvg.temporal.access import TrustedPrincipal
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.retriever import TemporalRetriever
from rvg.temporal.store import TemporalStore
from rvg.testing.fake_openrouter import FakeOpenRouter, providers
from rvg.transport import JsonTransport
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

PG = os.environ.get("DEEPJEV_TEST_POSTGRES_DSN")
NEEDS_PG = pytest.mark.skipif(not PG, reason="no PostgreSQL")
BACKENDS = ["sqlite", pytest.param("postgres", marks=NEEDS_PG)]
DIMS = 8
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
QUESTION = "What is the budget of project P?"
Q = np.eye(DIMS)[0]
ALICE = TrustedPrincipal("alice", groups=("staff",))
WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
]
LATEST = Arm.of("acl-latest", release={"mode": "latest_at_release"})
SNAPSHOT = Arm.of("acl-snapshot", release={"mode": "as_of_snapshot"})
# U24 (0.1.0.dev27): `current` (the default before) is an explicit arm; an engine without release
# keys runs the policy's scoped check
CURRENT = Arm.of("acl-current", release={"mode": "current"})


class Fixed:
    mode = "fixture"

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        return EmbeddingBatch(tuple(tuple(Q) for _ in texts), SPACE, ("f",), ())


def doc(identifier, text, *, family=None, revision=1, day=1, supersedes=()):
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family or identifier,
            "revision": revision,
            "title": "Project P" if "budget" in text else f"Note {identifier}",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": f"2026-09-{day:02d}T00:00:00Z",
            "observed_at": f"2026-09-{day:02d}T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
            "supersedes": [{"document_id": t} for t in supersedes],
        }
    )


class World:
    """p1 (family P, the answer, ranked first), 25 background notes without figures (19 of
    them in the top 20), a far note outside every top 20, and a conflicting note c1 (another
    family, the question's own direction) visible only to `finance` at the start."""

    def __init__(self, backend="sqlite", tmp_path=None):
        self.backend, self.cleanup = backend, lambda: None
        if backend == "sqlite":
            self.store = TemporalStore(":memory:")
            self.storage = self.store.storage
        else:
            tag = uuid.uuid4().hex[:10]
            self.schema = f"st_{tag}"
            config = {
                "schema_version": 1,
                "state": {
                    "backend": "postgres",
                    "dsn_env": "RVG_TEST_PG",
                    "schema_name": self.schema,
                },
                "vectors": {"index": "local_exact", "matrix_dir": str(tmp_path / "m")},
            }
            self.storage = rvg.open_storage(config, root=tmp_path, environ={"RVG_TEST_PG": PG})
            self.store = self.storage.state

            def cleanup():
                import psycopg

                self.storage.close()
                with psycopg.connect(PG, autocommit=True) as conn:
                    conn.execute(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")

            self.cleanup = cleanup
        rng = np.random.default_rng(3)
        self.put(
            doc("p1", "The budget of project P is 500.", family="P"), Q + 0.1 * np.eye(DIMS)[1]
        )
        for k in range(25):
            v = rng.normal(size=DIMS)
            v[0] = -abs(v[0]) * 0.1
            self.put(doc(f"b{k:02d}", f"Background remark on weather {WORDS[k % 13]}."), v)
        self.put(doc("far", "A far note on trains."), -Q)
        self.store.commit_event(
            "t",
            document=doc("c1", "The budget of project P is 800.", family="C", day=2),
            access={"finance": "allow"},
            keep_access=False,
        )
        self.vector(doc("c1", "The budget of project P is 800.", family="C", day=2), Q)

    def put(self, d, vector=None):
        self.store.upsert("t", d)
        if vector is not None:
            self.vector(d, vector)

    def vector(self, d, v):
        self.storage.blobs.put_vector("t", SPACE.fingerprint, TemporalRetriever.input_text(d), v)

    def seed_background(self):
        """A background note that the question's search returns (and that is not evidence)."""
        return (
            "b00"
            if self._in_top("b00")
            else next(f"b{k:02d}" for k in range(25) if self._in_top(f"b{k:02d}"))
        )

    def _in_top(self, identifier):
        _, seen = ask(self)
        return any(identifier + "@1" in seeds for seeds, _ in seen["searches"])


def writer_with(hook, times=1):
    fired = []

    def write(payload):
        if len(fired) < times:
            fired.append(True)
            hook()
        for unit in payload["evidence"]:
            found = re.search(r"\bis (\d+)", unit["text"])
            if found:
                return {
                    "answerable": True,
                    "short_answer": found.group(1),
                    "sentences": [{"text": unit["text"], "evidence_ids": [unit["id"]]}],
                }
        return {"answerable": False, "short_answer": "", "sentences": []}

    return write


class Seen(TemporalRetriever):
    """Records each attempt's searches (seeds, documents) and validity plane. An attempt begins
    with a search at the candidate depth that is pinned to a snapshot (latest, as_of) or, in
    current, any candidate-depth search; latest's unpinned re-search is not an attempt."""

    def __init__(self, *args, mode="current", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode, self.attempts = mode, []

    async def search(
        self, tenant, question, context, *, strategy="dense", top_k=4, expand=True, at=None
    ):
        docs, retrieval = await super().search(
            tenant, question, context, strategy=strategy, top_k=top_k, expand=expand, at=at
        )
        if self.mode != "current" and at is None:
            return docs, retrieval
        if top_k >= 20:
            self.attempts.append({"searches": [], "plane": []})
        self.attempts[-1]["searches"].append(
            (tuple(retrieval.get("seed_ids") or ()), tuple(sorted(d.key for d in docs)))
        )
        return docs, retrieval

    def validity_documents(self, tenant, docs, *, at=None):
        found = super().validity_documents(tenant, docs, at=at)
        if self.attempts:
            self.attempts[-1]["plane"].append(tuple(sorted(d.key for d in found)))
        return found


def budget(clock=None):
    extra = {"clock": clock} if clock else {}
    return CallContext.with_timeout(
        60,
        max_decision_calls=400,
        max_generation_calls=24,
        max_embedding_calls=40,
        max_semantic_proposer_calls=16,
        max_surface_proposer_calls=8,
        max_http_attempts=800,
        **extra,
    )


def ask(world, hook=None, *, arm=CURRENT, times=1, principal=ALICE):
    mode = dict(arm.release).get("mode") if arm is not None else "current"

    async def go():
        fake = FakeOpenRouter(writer=writer_with(hook or (lambda: None), times))
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(JsonTransport(client, timeout=5, max_attempts=2))
            retriever = Seen(world.storage, Fixed(), principal=principal, mode=mode)
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
            result = await engine.answer("t", q, budget())
            return result, (retriever.attempts[-1] if retriever.attempts else {})

    return asyncio.run(go())


def released(result):
    return result["status"] in ("FULL", "PARTIAL")


def consistent(world, result, seen, arm=CURRENT):
    """The absolute rule: a released answer equals a fresh run at the state now (the hooks run
    inside the attempt, so the state now is the state the release check read)."""
    if not released(result):
        return True
    fresh, fresh_seen = ask(world, arm=arm)
    return (
        fresh["status"] == result["status"]
        and fresh.get("short_answer") == result.get("short_answer")
        and fresh.get("evidence_ids") == result.get("evidence_ids")
        and fresh_seen == seen
    )


@pytest.fixture(params=BACKENDS)
def world(request, tmp_path):
    w = World(request.param, tmp_path)
    yield w
    w.cleanup()


# --- baseline behaviour that must not change ---------------------------------------------------


def test_nothing_changes_every_mode_releases_the_answer(world):
    for arm in (None, SNAPSHOT, LATEST):
        result, seen = ask(world, arm=arm)
        assert result["status"] == "FULL" and result["short_answer"] == "500"
        assert consistent(world, result, seen, arm)


@pytest.mark.parametrize("arm", [None, SNAPSHOT, LATEST], ids=["current", "as_of", "latest"])
def test_evidence_withdrawn_is_still_blocked_by_the_release_check(world, arm):
    result, _ = ask(
        world, lambda: world.store.replace_access("t", "p1", {"alice": "deny"}), arm=arm
    )
    assert result["reason"] == "permission_changed_before_release"


# --- D-ACL ---------------------------------------------------------------------------------------


def reveal(world):  # c1 becomes visible to alice's group: a registered ACL writer
    return lambda: world.store.replace_access("t", "c1", {"staff": "allow"})


def reveal_one_entry(world):  # the same through the single-entry writer
    return lambda: world.store.set_access("t", "c1", "staff", "allow")


def hide_read(world):  # a document the search returned (not evidence) becomes hidden
    target = world.seed_background()
    return lambda: world.store.replace_access("t", target, {"finance": "allow"})


def unrelated_acl(world):  # a far note nobody's search returns: still a tenant ACL change
    return lambda: world.store.replace_access("t", "far", {"bob": "deny"})


ACL_CASES = {
    "reveal": reveal,
    "reveal_one_entry": reveal_one_entry,
    "hide_read": hide_read,
    "unrelated_acl": unrelated_acl,
}


@pytest.mark.parametrize("case", sorted(ACL_CASES))
def test_current_does_not_release_an_answer_made_before_an_acl_change(world, case):
    result, seen = ask(world, ACL_CASES[case](world))
    assert not released(result) and result["reason"] == "access_changed"
    assert consistent(world, result, seen)


@pytest.mark.parametrize("case", sorted(ACL_CASES))
def test_as_of_snapshot_does_not_release_across_an_acl_change(world, case):
    # ACL has no history, so a snapshot answer cannot be made consistent with the ACL at S.
    result, _ = ask(world, ACL_CASES[case](world), arm=SNAPSHOT)
    assert not released(result) and result["reason"] == "access_changed"


@pytest.mark.parametrize("case", sorted(ACL_CASES))
def test_latest_answers_again_after_an_acl_change(world, case):
    result, seen = ask(world, ACL_CASES[case](world), arm=LATEST)
    assert result.get("retried_after_change") is True
    assert consistent(world, result, seen, LATEST)
    if case.startswith("reveal"):
        assert result["short_answer"] == "800"  # the conflicting note is now in view and first


def test_an_acl_writer_that_bypasses_the_outbox_still_moves_the_epoch_and_is_caught():
    w = World()

    def bypass():
        with w.store.lock, w.store.conn:
            w.store.conn.execute(
                "INSERT OR REPLACE INTO temporal_acl VALUES(?,?,?,?)", ("t", "c1", "staff", "allow")
            )
            w.store.conn.execute(
                "UPDATE temporal_acl_epochs SET epoch=epoch+1 WHERE tenant=?", ("t",)
            )

    result, seen = ask(w, bypass)
    assert not released(result) and result["reason"] == "access_changed"
    assert consistent(w, result, seen)


def test_an_acl_change_every_attempt_ends_latest_in_stale_after_retry():
    w, flips = World(), [0]

    def flip():
        flips[0] += 1
        w.store.replace_access("t", "far", {f"u{flips[0]}": "deny"})

    result, _ = ask(w, flip, arm=LATEST, times=2)
    assert result["reason"] == "stale_after_retry" and flips[0] == 2


# --- D-PHANTOM -----------------------------------------------------------------------------------


def hidden_declaration(world):
    return lambda: world.store.commit_event(
        "t",
        document=doc("h1", "Project P was restated.", family="H", day=15, supersedes=["p1"]),
        access={"auditors": "allow"},
        keep_access=False,
    )


def far_declaration(world):  # visible, another family, far from the question
    return lambda: world.put(
        doc("h2", "Project P restatement notice.", family="H2", day=15, supersedes=["p1"]), -Q
    )


@pytest.mark.parametrize("case", ["hidden", "far_visible"])
def test_latest_does_not_release_a_value_replaced_during_generation(world, case):
    hook = hidden_declaration(world) if case == "hidden" else far_declaration(world)
    result, seen = ask(world, hook, arm=LATEST)
    assert result.get("short_answer") != "500"
    assert consistent(world, result, seen, LATEST)


@pytest.mark.parametrize("case", ["hidden", "far_visible"])
def test_current_already_withholds_a_replacement_declared_during_generation(world, case):
    hook = hidden_declaration(world) if case == "hidden" else far_declaration(world)
    result, _ = ask(world, hook)
    assert result["reason"] == "source_changed"  # a knowledge event: unchanged since N3


def test_an_unrelated_insert_still_does_not_make_latest_answer_again():
    w = World()
    result, _ = ask(w, lambda: w.put(doc("zz", "Another far note.", day=15), -Q), arm=LATEST)
    assert released(result) and not result.get("retried_after_change")


# --- real PostgreSQL: another process writes while the answer is made ----------------------------

WRITER = """
import os, sys
import rvg
storage = rvg.open_storage(
    {"schema_version": 1,
     "state": {"backend": "postgres", "dsn_env": "RVG_TEST_PG", "schema_name": sys.argv[1]},
     "vectors": {"index": "local_exact", "matrix_dir": sys.argv[2]}},
    root=sys.argv[2], environ={"RVG_TEST_PG": os.environ["RVG_TEST_PG"]})
state = storage.state
if sys.argv[3] == "once":
    state.replace_access("t", "c1", {"staff": "allow"})
else:
    import time
    end, n = time.monotonic() + float(sys.argv[4]), 0
    while time.monotonic() < end:
        n += 1
        state.replace_access("t", "far", {f"w{n}": "deny"})
    print(n)
storage.close()
"""


def writer_process(world, tmp_path, *args):
    env = {**os.environ, "RVG_TEST_PG": PG}
    return [sys.executable, "-c", WRITER, world.schema, str(tmp_path / "w"), *args], env


@NEEDS_PG
def test_postgres_an_acl_change_committed_by_another_process_is_caught(tmp_path):
    w = World("postgres", tmp_path)
    try:
        cmd, env = writer_process(w, tmp_path, "once")
        result, seen = ask(w, lambda: subprocess.run(cmd, env=env, check=True, timeout=60))
        assert not released(result) and result["reason"] == "access_changed"
        assert consistent(w, result, seen)
    finally:
        w.cleanup()


@NEEDS_PG
def test_postgres_no_release_spans_an_acl_epoch_under_a_concurrent_writer(tmp_path):
    """A writer process changes the tenant's ACL continuously; every released answer must have
    read, at its release check, the ACL epoch it began with."""
    w = World("postgres", tmp_path)
    starts = []
    original = w.store.acl_epoch

    def recording(tenant):
        value = original(tenant)
        starts.append(value)
        return value

    w.store.acl_epoch = recording
    try:
        cmd, env = writer_process(w, tmp_path, "loop", "8")
        writer = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
        outcomes = []
        time.sleep(0.5)
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            starts.clear()
            result, _ = ask(w)
            first = starts[0] if starts else None
            outcomes.append(
                (
                    result["status"],
                    result.get("reason"),
                    first,
                    (result.get("release") or {}).get("acl_epoch"),
                )
            )
        written = int(writer.communicate(timeout=60)[0].strip() or 0)
    finally:
        w.store.acl_epoch = original
        w.cleanup()
    assert written > 0 and outcomes
    for status, reason, start, fence in outcomes:
        if status in ("FULL", "PARTIAL"):
            assert start is not None and fence == start
        else:
            assert reason in ("access_changed", "source_changed")
    assert any(reason == "access_changed" for _, reason, _, _ in outcomes)
