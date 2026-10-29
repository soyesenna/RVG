"""RVI-03 local step (docs/rvi/RVI03_LOCAL_PROTOCOL.md in the parent repository): the opt-in
scoped dependency fence and the release retry. Events fire from inside the (fake) writer call,
between retrieval and release, as in the N3 tests. Defaults and existing arms are unchanged.

The PostgreSQL cases need `DEEPJEV_TEST_POSTGRES_DSN`; without it they are skipped (not passes)."""

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
from rvg.universal import scoped
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

PG = os.environ.get("DEEPJEV_TEST_POSTGRES_DSN")
BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.skipif(not PG, reason="no PostgreSQL")),
]
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
SIDE = np.eye(DIMS)[7]  # a direction no other fixture vector uses
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
SCOPED = Arm.of("rvi03-scoped", release={"mode": "scoped_dependency"})
SCOPED_RETRY = Arm.of("rvi03-scoped-retry", release={"mode": "scoped_dependency", "retry": 1})
CURRENT_RETRY = Arm.of("rvi03-current-retry", release={"mode": "current", "retry": 1})
# U24 (0.1.0.dev27): `current` (the default before) is an explicit arm now
CURRENT = Arm.of("rvi03-current", release={"mode": "current"})


class Fixed:
    mode = "fixture"

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        return EmbeddingBatch(tuple(tuple(Q) for _ in texts), SPACE, ("f",), ())


def doc(identifier, text, *, family=None, revision=1, day=1, observed=None, supersedes=()):
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family or identifier,
            "revision": revision,
            "title": "Project P" if "budget" in text else f"Note {identifier}",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": f"2026-09-{day:02d}T00:00:00Z",
            "observed_at": f"2026-09-{observed or day:02d}T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
            "supersedes": [{"document_id": t} for t in supersedes],
        }
    )


def at_score(score):
    """A unit vector whose cosine with the question vector is `score`."""
    return score * Q + np.sqrt(1 - score**2) * SIDE


class World:
    """One tenant: the answer document p1 (family P, ranked first), a family sibling s1 with no
    figure, 25 background notes without figures below it and a far note outside every top-20."""

    def __init__(self, backend="sqlite", tmp_path=None):
        self.cleanup = lambda: None
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
                    "schema_name": f"st_{tag}",
                },
                "vectors": {"index": "local_exact", "matrix_dir": str(tmp_path / "m")},
            }
            self.storage = rvg.open_storage(config, root=tmp_path, environ={"RVG_TEST_PG": PG})
            self.store = self.storage.state

            def cleanup():
                import psycopg

                self.storage.close()
                with psycopg.connect(PG, autocommit=True) as conn:
                    conn.execute(f"DROP SCHEMA IF EXISTS st_{tag} CASCADE")

            self.cleanup = cleanup
        rng = np.random.default_rng(3)
        self.put(
            doc("p1", "The budget of project P is 500.", family="P"), Q + 0.1 * np.eye(DIMS)[1]
        )
        self.put(
            doc("s1", "Project P overview and staffing notes.", family="P"), -Q + np.eye(DIMS)[2]
        )
        for k in range(25):
            v = rng.normal(size=DIMS)
            v[0], v[7] = -abs(v[0]) * 0.1, 0.0
            self.put(doc(f"b{k:02d}", f"Background remark on weather {WORDS[k % 13]}."), v)
        self.put(doc("far", "A far note on trains."), -Q)

    def put(self, d, vector=None):
        self.store.upsert("t", d)
        if vector is not None:
            self.storage.blobs.put_vector(
                "t", SPACE.fingerprint, TemporalRetriever.input_text(d), vector
            )

    def scores(self):
        """Cosines of every stored revision with the question, highest first."""
        out = []
        for d in self.store.documents("t"):
            v = self.storage.blobs.vector("t", SPACE.fingerprint, TemporalRetriever.input_text(d))
            if v is not None:
                v = np.asarray(v, dtype=float)
                out.append(float(v @ Q / np.linalg.norm(v)))
        return sorted(out, reverse=True)

    def bump_acl_without_outbox(self, identifier, principal, effect):
        """A writer that bypasses the outbox (as a tenant import does for its ACL)."""
        conn = self.store.conn
        conn.execute(
            "INSERT OR REPLACE INTO temporal_acl VALUES(?,?,?,?)",
            ("t", identifier, principal, effect),
        )
        conn.execute("UPDATE temporal_acl_epochs SET epoch=epoch+1 WHERE tenant=?", ("t",))
        conn.commit()


class Recording(TemporalRetriever):
    """Records what every search returned (seeds and document keys) and the validity plane
    read for it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    async def search(self, *args, **kwargs):
        docs, retrieval = await super().search(*args, **kwargs)
        self.seen.append(
            (tuple(retrieval.get("seed_ids") or ()), tuple(sorted(d.key for d in docs)))
        )
        return docs, retrieval

    def validity_documents(self, tenant, docs, *, at=None):
        found = super().validity_documents(tenant, docs, at=at)
        self.seen.append(("plane", tuple(sorted(d.key for d in found))))
        return found


def writer_with(hook, times=1):
    fired = []

    def write(payload):
        if len(fired) < times:  # the hook runs in the first `times` writer calls
            fired.append(True)
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

    return write


def ask(world, hook=lambda: None, *, arm=SCOPED, principal=ALICE, times=1, context=None):
    async def go():
        fake = FakeOpenRouter(writer=writer_with(hook, times))
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(JsonTransport(client, timeout=5, max_attempts=2))
            retriever = Recording(world.storage, Fixed(), principal=principal)
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
            result = await engine.answer("t", q, ctx)
            return result, retriever.seen

    return asyncio.run(go())


def released(result):
    return result["status"] in ("FULL", "PARTIAL")


def reasons(result):
    return set(
        (result.get("release", {}).get("scope") or result.get("scope") or {}).get("reasons", {})
    )


@pytest.fixture(params=BACKENDS)
def world(request, tmp_path):
    w = World(request.param, tmp_path)
    yield w
    w.cleanup()


# --- defaults --------------------------------------------------------------------------------


def test_existing_arms_and_defaults_are_unchanged():
    recorded = {
        Arm.of("n3-latest_at_release", release={"mode": "latest_at_release"}): "b6906daf",
        Arm.of("n3-as_of_snapshot", release={"mode": "as_of_snapshot"}): "d3be37a6",
        Arm.of("n4-rewrite", cascade={"conflict_partial": "rewrite"}): "4883bf2d",
    }
    for arm, prefix in recorded.items():
        assert arm.digest.startswith(prefix)
    for bad in (
        {"mode": "latest_at_release", "retry": 1},
        {"mode": "as_of_snapshot", "retry": 1},
        {"mode": "scoped_dependency", "retry": 2},
        {"mode": "scoped_dependency", "retry": True},
        {"mode": "scoped"},
    ):
        with pytest.raises(ValueError):
            Arm.of("bad", release=bad)
    # U24 (0.1.0.dev27): scoped dependency is the policy's default, current the rollback arm
    assert rvg.capabilities()["release_modes"]["scoped_dependency"]["status"] == "ADOPTED"
    assert rvg.capabilities()["release_modes"]["current"]["status"] == "IMPLEMENTED"


def test_current_without_retry_still_withholds_on_any_change():
    w = World()
    result, _ = ask(w, lambda: w.put(doc("zz", "Another far note.", day=15), -Q), arm=CURRENT)
    assert result["reason"] == "source_changed" and "retried_after_change" not in result


def test_no_change_releases_with_a_receipt(world):
    result, _ = ask(world)
    assert result["status"] == "FULL" and result["short_answer"] == "500"
    assert result["release"]["mode"] == "scoped_dependency"
    receipt = result["release"]["scope"]
    assert receipt["status"] == "CLEAR" and receipt["checked"] == 0
    assert receipt["guarantee"] == "authority+watermark+cutoff"


# --- unrelated changes -----------------------------------------------------------------------


def unrelated(world, kind):
    if kind == "insert":
        return lambda: world.put(doc("zz", "Another far note on ships.", day=15), -Q)
    if kind == "revoke":
        return lambda: world.store.revoke("t", "far")
    return lambda: world.store.replace_access("t", "far", {"bob": "deny"})


@pytest.mark.parametrize("kind", ["insert", "revoke", "acl"])
def test_an_unrelated_change_does_not_withhold_a_scoped_answer(world, kind):
    result, _ = ask(world, unrelated(world, kind))
    assert result["status"] == "FULL" and result["short_answer"] == "500"
    assert result["release"]["scope"]["status"] == "CLEAR"
    assert result["release"]["scope"]["checked"] == 1


@pytest.mark.parametrize("kind", ["insert", "revoke"])
def test_the_same_unrelated_change_withholds_current(kind):
    w = World()
    result, _ = ask(w, unrelated(w, kind), arm=CURRENT)
    assert result["reason"] == "source_changed"


# --- related changes -------------------------------------------------------------------------


def related(world, kind):
    if kind == "revision":
        return lambda: world.put(
            doc("p1", "The budget of project P is 650.", family="P", revision=2, day=15),
            Q + 0.1 * np.eye(DIMS)[1],
        )
    if kind == "family":  # a new document of the answer's family, far from the question
        return lambda: world.put(
            doc("s2", "Project P staffing and dates.", family="P", day=15),
            -Q + np.eye(DIMS)[2],
        )
    if kind == "hidden_supersedes":  # a phantom: hidden, another family, aimed at p1

        def go():
            world.store.commit_event(
                "t",
                document=doc(
                    "h1", "Project P was restated.", family="H", day=15, supersedes=["p1"]
                ),
                access={"auditors": "allow"},
                keep_access=False,
            )

        return go
    if kind == "alias":
        return lambda: world.put(
            doc("al", "The budget of project P is 700.", family="AL", day=15), Q
        )
    if kind == "low_rank":
        scores = world.scores()
        inside = (scores[13] + scores[14]) / 2  # between the 14th and 15th of the top 20
        return lambda: world.put(
            doc("lr", "The budget of project P is 710.", family="LR", day=15), at_score(inside)
        )
    if kind == "acl_visible":
        world.store.commit_event(
            "t",
            document=doc("c1", "The budget of project P is 800.", family="C", day=2),
            access={"finance": "allow"},
            keep_access=False,
        )
        world.storage.blobs.put_vector(
            "t",
            SPACE.fingerprint,
            TemporalRetriever.input_text(
                doc("c1", "The budget of project P is 800.", family="C", day=2)
            ),
            Q,
        )
        return lambda: world.store.replace_access("t", "c1", {"staff": "allow"})
    raise ValueError(kind)


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("revision", "insert:RELATED:scope:document"),
        ("family", "insert:RELATED:scope:family"),
        ("hidden_supersedes", "insert:RELATED:scope:supersedes"),
        ("alias", "insert:RELATED:watermark"),
        ("low_rank", "insert:RELATED:watermark"),
        ("acl_visible", "acl:RELATED:watermark"),
    ],
)
def test_a_related_change_withholds_a_scoped_answer(world, kind, reason):
    result, _ = ask(world, related(world, kind))
    assert result["status"] == "ABSTAIN" and result["reason"] == "scope_changed"
    assert reason in reasons(result)


def test_an_acl_change_that_reveals_a_document_withholds_current():
    # dev5 released here (the N3 contract: ACL is no event); the release-input consistency fix
    # (branch rvg/rvi-acl, 43a8122) withholds current whenever the tenant's ACL epoch moved.
    w = World()
    result, _ = ask(w, related(w, "acl_visible"), arm=CURRENT)
    assert result["status"] == "ABSTAIN" and result["reason"] == "access_changed"


@pytest.mark.parametrize("kind", ["revoke", "deny"])
def test_the_release_check_still_runs_first(world, kind):
    if kind == "revoke":
        result, _ = ask(world, lambda: world.store.revoke("t", "p1"))
        assert result["reason"] == "revoked_before_release"
    else:
        result, _ = ask(world, lambda: world.store.replace_access("t", "p1", {"alice": "deny"}))
        assert result["reason"] == "permission_changed_before_release"


# --- boundaries: released, and equal to a fresh run -------------------------------------------


def boundary(world, kind):
    if kind == "below_watermark":
        scores = world.scores()
        below = (scores[21] + scores[22]) / 2  # under the 20th
        return lambda: world.put(
            doc("bw", "The budget of project P is 720.", family="BW", day=15), at_score(below)
        )
    if kind == "cutoff":  # a new document observed after the question's known_at
        return lambda: world.put(
            doc("ct", "The budget of project P is 650.", family="CT", day=15, observed=25), Q
        )
    if kind == "hidden_no_declaration":
        return lambda: world.store.commit_event(
            "t",
            document=doc("hn", "The budget of project P is 730.", family="HN", day=15),
            access={"auditors": "allow"},
            keep_access=False,
        )
    raise ValueError(kind)


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("below_watermark", "insert:CLEAR:below_watermark"),
        ("cutoff", "insert:CLEAR:cutoff"),
        ("hidden_no_declaration", "insert:CLEAR:not_retrievable"),
    ],
)
def test_a_change_outside_the_guarantee_is_released_as_a_fresh_run_would_answer(
    world, kind, reason
):
    result, seen = ask(world, boundary(world, kind))
    assert result["status"] == "FULL" and result["short_answer"] == "500"
    assert reason in reasons(result)
    fresh, fresh_seen = ask(world)  # the same store, after the change, nothing during the run
    assert fresh["short_answer"] == result["short_answer"]
    assert fresh["evidence_ids"] == result["evidence_ids"]
    assert fresh_seen == seen


def test_a_revision_of_a_read_document_beyond_the_cutoff_is_still_related(world):
    """The validity plane reads every revision of a read document whatever its observation
    time (the gate then discards it), so its input changes: RELATED (RVI-03 local run 1, B2)."""
    late = doc("p1", "The budget of project P is 650.", family="P", revision=2, day=15, observed=25)
    result, _ = ask(world, lambda: world.put(late, Q + 0.1 * np.eye(DIMS)[1]))
    assert result["reason"] == "scope_changed" and "insert:RELATED:scope:document" in reasons(
        result
    )
    again, _ = ask(world, arm=SCOPED_RETRY)  # nothing moves now: the cutoff keeps the old value
    assert again["short_answer"] == "500"


# --- unmapped changes fall back to the global rule --------------------------------------------


def test_a_new_revision_without_a_vector_falls_back_to_the_global_rule(world):
    result, _ = ask(world, lambda: world.put(doc("pv", "A pending far note.", day=15)))
    assert result["reason"] == "source_changed" and "insert:UNKNOWN:pending_vector" in reasons(
        result
    )


def test_an_acl_writer_that_bypasses_the_outbox_is_unmapped():
    w = World()
    w.store.commit_event(
        "t",
        document=doc("c1", "The budget of project P is 800.", family="C", day=2),
        access={"finance": "allow"},
        keep_access=False,
    )
    w.storage.blobs.put_vector(
        "t",
        SPACE.fingerprint,
        TemporalRetriever.input_text(
            doc("c1", "The budget of project P is 800.", family="C", day=2)
        ),
        Q,
    )
    result, _ = ask(w, lambda: w.bump_acl_without_outbox("c1", "staff", "allow"))
    assert result["status"] == "ABSTAIN" and result["reason"] == "source_changed"
    assert "acl:UNKNOWN:unregistered_acl_writer" in reasons(result)


def test_a_changed_policy_hash_is_unmapped(monkeypatch):
    w = World()

    def swap():
        monkeypatch.setattr(scoped, "universal_hash", lambda: "another-policy")

    result, _ = ask(w, swap)
    assert result["reason"] == "source_changed" and "policy:UNKNOWN:policy_changed" in reasons(
        result
    )


# --- retry -----------------------------------------------------------------------------------


def test_a_scoped_retry_answers_with_the_revision_now_in_force(world):
    result, _ = ask(world, related(world, "revision"), arm=SCOPED_RETRY)
    assert result["status"] == "FULL" and result["short_answer"] == "650"
    assert result["retried_after_change"] is True
    assert result["first_attempt"]["reason"] == "scope_changed"


def test_an_unrelated_change_is_not_retried():
    w = World()
    result, _ = ask(w, unrelated(w, "insert"), arm=SCOPED_RETRY)
    assert result["short_answer"] == "500" and "retried_after_change" not in result


def test_the_retry_is_bounded_to_one():
    w, count = World(), [0]

    def every_time():
        count[0] += 1
        w.put(
            doc(
                "p1",
                f"The budget of project P is {600 + count[0]}.",
                family="P",
                revision=1 + count[0],
                day=15 + count[0],
            ),
            Q + 0.1 * np.eye(DIMS)[1],
        )

    result, _ = ask(w, every_time, arm=SCOPED_RETRY, times=2)  # first attempt and retry
    assert result["reason"] == "stale_after_retry" and count[0] == 2


def test_no_retry_starts_with_less_time_left_than_the_first_attempt_took():
    w, now = World(), [1000.0]
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

    def slow_and_changed():
        now[0] += 40  # the first attempt has used 40 of 60 seconds
        related(w, "revision")()

    result, _ = ask(w, slow_and_changed, arm=SCOPED_RETRY, context=ctx)
    assert result["status"] == "ABSTAIN" and result["reason"] == "retry_budget_exhausted"
    assert result["retry_error"] == "deadline_below_first_attempt"


def test_the_retry_ablation_retries_on_any_change():
    w = World()
    result, _ = ask(w, unrelated(w, "insert"), arm=CURRENT_RETRY)
    assert result["short_answer"] == "500" and result["retried_after_change"] is True
    assert result["first_attempt"]["reason"] == "source_changed"


# --- the guarantee, on random event sequences --------------------------------------------------

POOL = [
    ("insert_far", lambda w, n: w.put(doc(f"x{n}", "Another far note on ships.", day=15), -Q)),
    ("revoke_far", lambda w, n: w.store.revoke("t", "far")),
    ("acl_far", lambda w, n: w.store.replace_access("t", "far", {f"u{n}": "deny"})),
    ("revoke_background", lambda w, n: w.store.revoke("t", f"b{n % 25:02d}")),
    ("revision", lambda w, n: related(w, "revision")()),
    ("family", lambda w, n: related(w, "family")()),
    (
        "alias",
        lambda w, n: w.put(
            doc(f"al{n}", "The budget of project P is 700.", family=f"AL{n}", day=15), Q
        ),
    ),
    (
        "below",
        lambda w, n: w.put(
            doc(f"bw{n}", "The budget of project P is 720.", family=f"BW{n}", day=15),
            at_score(-0.9),
        ),
    ),
    ("hidden_supersedes", lambda w, n: related(w, "hidden_supersedes")()),
    (
        "late_revision",  # beyond the cutoff, a read document's id
        lambda w, n: w.put(
            doc(
                "p1", "The budget of project P is 660.", family="P", revision=3, day=16, observed=26
            ),
            Q + 0.1 * np.eye(DIMS)[1],
        ),
    ),
    ("late_new", lambda w, n: boundary(w, "cutoff")()),
]


@pytest.mark.parametrize("seed", range(24))
def test_a_scoped_release_equals_a_fresh_run_at_the_fence(seed):
    rng = np.random.default_rng(seed)
    w = World()
    picks = [POOL[i] for i in rng.choice(len(POOL), size=int(rng.integers(1, 4)), replace=False)]

    def fire():
        for n, (_, event) in enumerate(picks):
            event(w, n)

    result, seen = ask(w, fire)
    if released(result):
        fresh, fresh_seen = ask(w)
        assert fresh["short_answer"] == result["short_answer"]
        assert fresh["evidence_ids"] == result["evidence_ids"]
        assert fresh_seen == seen
    else:
        assert result["reason"] in ("scope_changed", "source_changed")


# --- real PostgreSQL: another process writes while the answer is made ----------------------------

WRITER = """
import os, sys, time
import numpy as np
import rvg
from rvg.temporal.models import TemporalDocument
from rvg.temporal.retriever import TemporalRetriever
schema, root, mode, space = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
storage = rvg.open_storage(
    {"schema_version": 1,
     "state": {"backend": "postgres", "dsn_env": "RVG_TEST_PG", "schema_name": schema},
     "vectors": {"index": "local_exact", "matrix_dir": root}},
    root=root, environ={"RVG_TEST_PG": os.environ["RVG_TEST_PG"]})
state = storage.state

def doc(identifier, text, family, revision=1):
    return TemporalDocument.model_validate({
        "id": identifier, "family": family, "revision": revision, "title": "Note " + identifier,
        "text": text, "source": "fixture://" + identifier,
        "published_at": "2026-09-15T00:00:00Z", "observed_at": "2026-09-15T00:00:00Z",
        "units": [{"id": "u0", "start": 0, "end": len(text)}]})

if mode == "revision":
    state.upsert("t", doc("p1", "The budget of project P is 650.", "P", revision=2))
else:
    end, n = time.monotonic() + float(sys.argv[5]), 0
    while time.monotonic() < end:
        n += 1
        if n % 2:
            state.replace_access("t", "far", {"w%d" % n: "deny"})
        else:
            d = doc("x%d" % n, "Another far note on ships.", "X%d" % n)
            state.upsert("t", d)
            far = np.zeros(8); far[0] = -1.0
            storage.blobs.put_vector("t", space, TemporalRetriever.input_text(d), far)
    print(n)
storage.close()
"""


def writer_process(world, tmp_path, *args):
    env = {**os.environ, "RVG_TEST_PG": PG}
    cmd = [sys.executable, "-c", WRITER, world.schema, str(tmp_path / "w"), args[0]]
    return [*cmd, SPACE.fingerprint, *args[1:]], env


@pytest.mark.skipif(not PG, reason="no PostgreSQL")
def test_postgres_a_revision_committed_by_another_process_is_related(tmp_path):
    w = World("postgres", tmp_path)
    try:
        cmd, env = writer_process(w, tmp_path, "revision")
        result, _ = ask(w, lambda: subprocess.run(cmd, env=env, check=True, timeout=60))
        assert result["reason"] == "scope_changed"
        assert "insert:RELATED:scope:document" in reasons(result)
    finally:
        w.cleanup()


@pytest.mark.skipif(not PG, reason="no PostgreSQL")
def test_postgres_scoped_keeps_releasing_under_an_unrelated_writer_process(tmp_path):
    """Another process changes a far note's ACL and adds far notes (each followed by its vector)
    continuously. Scoped answers keep being released with the value of p1; a withheld one may
    only be the global fallback (a note seen before its vector, or the ACL moving during the
    check). A note seen before its vector by the answer's own search (`pending` > 0) leaves
    that search without an exact watermark: every change outside is then UNKNOWN
    (`no_exact_watermark`) — the same safe fallback, found in 2 of 10 runs on PostgreSQL (0.1.0,
    parent docs/PUB_2_REPORT.md)."""
    w = World("postgres", tmp_path)
    try:
        cmd, env = writer_process(w, tmp_path, "loop", "8")
        writer = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
        outcomes = []
        time.sleep(0.5)
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            result, _ = ask(w)
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
            # the global fallback only: a note seen before its vector, or the ACL moving while
            # the check read it (nothing the writer did is related to p1)
            assert r["reason"] == "source_changed"
            allowed = {"insert:UNKNOWN:pending_vector", "acl:UNKNOWN:acl_moved_during_check"}
            if any(a["retrieval"].get("pending") for a in r.get("audit") or []):
                allowed |= {"insert:UNKNOWN:no_exact_watermark", "acl:UNKNOWN:no_exact_watermark"}
            assert {k for k in reasons(r) if ":CLEAR:" not in k} <= allowed
