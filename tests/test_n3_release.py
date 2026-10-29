"""N3 (docs/UNIVERSAL_N3_PROTOCOL.md): what is released when the knowledge base, its revocations
or a principal's permissions change while an answer is being made — per release mode, with
events fired from inside the (fake) writer call, i.e. between retrieval and release."""

import asyncio
import re

import httpx
import numpy as np
import pytest

from rvg.contracts import EmbeddingBatch, EmbeddingSpace
from rvg.runtime import CallContext
from rvg.temporal.access import TrustedPrincipal
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore
from rvg.testing.fake_openrouter import FakeOpenRouter, providers
from rvg.transport import JsonTransport
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

DIMS = 6
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
MODES = ("current", "as_of_snapshot", "latest_at_release")


class Fixed:
    mode = "fixture"

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        return EmbeddingBatch(tuple(tuple(Q) for _ in texts), SPACE, ("f",), ())


def doc(identifier, text, *, family=None, revision=1, day=1):
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family or identifier,
            "revision": revision,
            "title": "Project P" if family == "P" else f"Note {identifier}",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": f"2026-09-{day:02d}T00:00:00Z",
            "observed_at": f"2026-09-{day:02d}T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
        }
    )


def put(store, d, vector):
    store.upsert("t", d)
    store.base.put_vector("t", SPACE.fingerprint, TemporalRetriever.input_text(d), vector)


def knowledge():
    store = TemporalStore(":memory:")
    rng = np.random.default_rng(3)
    put(store, doc("p1", "The budget of project P is 500.", family="P"), Q + 0.1 * np.eye(DIMS)[1])
    for k in range(25):  # background: below the target, 20 of them inside the candidate list
        v = rng.normal(size=DIMS)
        v[0] = -abs(v[0]) * 0.1
        put(store, doc(f"b{k:02d}", f"Background remark {k} on weather."), v)
    return store


def writer_with(hook):
    fired = []

    def write(payload):
        if not fired:
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


def ask(store, hook=lambda: None, *, mode="current", principal=None):
    # U24 (0.1.0.dev27): `current` is an explicit arm now that the policy's default is scoped
    arm = Arm.of(f"n3-{mode}", release={"mode": mode})

    async def go():
        fake = FakeOpenRouter(writer=writer_with(hook))
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(JsonTransport(client, timeout=5, max_attempts=2))
            retriever = TemporalRetriever(store, Fixed(), principal=principal)
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
            ctx = CallContext.with_timeout(
                60,
                max_decision_calls=400,
                max_generation_calls=24,
                max_embedding_calls=40,
                max_semantic_proposer_calls=16,
                max_surface_proposer_calls=8,
                max_http_attempts=800,
            )
            return await engine.answer("t", q, ctx)

    return asyncio.run(go())


def released(result):
    return result["status"] in ("FULL", "PARTIAL")


def test_baseline_every_mode_releases_the_same_answer_when_nothing_changes():
    for mode in MODES:
        result = ask(knowledge(), mode=mode)
        assert result["status"] == "FULL" and result["short_answer"] == "500"
        assert result["release"]["mode"] == mode


def test_a_new_revision_during_generation():
    new = doc("p1", "The budget of project P is 650.", family="P", revision=2, day=15)
    out = {}
    for mode in MODES:
        store = knowledge()
        out[mode] = ask(store, lambda s=store: put(s, new, Q + 0.1 * np.eye(DIMS)[1]), mode=mode)
    assert out["current"]["reason"] == "source_changed"  # any change: nothing released
    # a fixed snapshot answers for S, and says which S
    assert out["as_of_snapshot"]["short_answer"] == "500"
    assert (
        out["as_of_snapshot"]["release"]["knowledge_snapshot"]
        < out["as_of_snapshot"]["snapshot"] + 1
    )
    # latest: the family moved, one fresh attempt answers with the revision now in force
    assert out["latest_at_release"]["retried_after_change"] is True
    assert out["latest_at_release"]["short_answer"] == "650"


@pytest.mark.parametrize("mode", MODES)
def test_a_revocation_during_generation_is_never_released(mode):
    store = knowledge()
    result = ask(store, lambda: store.revoke("t", "p1"), mode=mode)
    assert not released(result)
    assert result["reason"] in {"revoked_before_release", "source_changed"}


@pytest.mark.parametrize("mode", MODES)
def test_a_permission_withdrawn_during_generation_is_never_released(mode):
    store = knowledge()
    store.set_access("t", "p1", "staff", "allow")
    assert released(ask(store, mode=mode, principal=ALICE))
    result = ask(
        store, lambda: store.set_access("t", "p1", "alice", "deny"), mode=mode, principal=ALICE
    )
    assert not released(result) and result["reason"] == "permission_changed_before_release"
    assert result["blocked_documents"] == ["p1"]


def test_documents_under_acl_are_invisible_without_a_matching_principal():
    store = knowledge()
    store.set_access("t", "p1", "finance-team", "allow")
    for principal in (None, ALICE):
        result = ask(store, principal=principal)
        assert "p1@1#u0" not in (result.get("evidence_ids") or [])
    assert "p1" in store.hidden("t", ALICE) and "b00" not in store.hidden("t", ALICE)
    assert released(ask(store, principal=TrustedPrincipal("bob", groups=("finance-team",))))


def test_an_unrelated_insert_blocks_only_the_current_mode():
    unrelated = doc("zz", "Background remark on trains.", day=15)
    out = {}
    for mode in MODES:
        store = knowledge()
        out[mode] = ask(store, lambda s=store: put(s, unrelated, -Q), mode=mode)
    assert out["current"]["reason"] == "source_changed"
    assert released(out["as_of_snapshot"]) and released(out["latest_at_release"])
    assert not out["latest_at_release"].get("retried_after_change")


def test_one_relevant_change_is_retried_and_two_end_in_stale_after_retry():
    store = knowledge()
    new = doc("p1", "The budget of project P is 650.", family="P", revision=2, day=15)
    once = ask(store, lambda: put(store, new, Q + 0.1 * np.eye(DIMS)[1]), mode="latest_at_release")
    assert once.get("retried_after_change") is True and once["short_answer"] == "650"
    store2, count = knowledge(), [0]

    def every_time():  # a new revision during the first attempt and again during the retry
        count[0] += 1
        revision = doc(
            "p1",
            f"The budget of project P is {600 + count[0]}.",
            family="P",
            revision=1 + count[0],
            day=15 + count[0],
        )
        put(store2, revision, Q + 0.1 * np.eye(DIMS)[1])

    result = ask_with_repeating_hook(store2, every_time)
    assert result["reason"] == "stale_after_retry" and count[0] == 2  # bounded: one retry


def ask_with_repeating_hook(store, hook):
    """A writer that fires its hook on every answer's first write (so also in the retry)."""
    arm = Arm.of("n3-latest", release={"mode": "latest_at_release"})
    writes = {"n": 0}

    def write(payload):
        writes["n"] += 1
        if writes["n"] in (1, 2) and len(payload["evidence"]) <= 3:
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
            engine = UniversalEngine(
                retriever=TemporalRetriever(store, Fixed()),
                decision=p["decision"],
                writer=p["writer"],
                extractor=p["extractor"],
                arm=arm,
            )
            q = TemporalQuestion(
                text=QUESTION, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
            )
            ctx = CallContext.with_timeout(
                60,
                max_decision_calls=400,
                max_generation_calls=24,
                max_embedding_calls=40,
                max_semantic_proposer_calls=16,
                max_surface_proposer_calls=8,
                max_http_attempts=800,
            )
            return await engine.answer("t", q, ctx)

    return asyncio.run(go())


def test_duplicate_events_change_nothing():
    store = knowledge()
    before = store.snapshot("t")
    store.upsert("t", doc("p1", "The budget of project P is 500.", family="P"))  # same body
    assert store.snapshot("t") == before
    store.revoke("t", "b01")
    once = store.snapshot("t")
    store.revoke("t", "b01")  # a duplicate revocation used to record an event (N3 fix)
    assert store.snapshot("t") == once
    # so a duplicate during generation no longer makes the current mode abstain
    assert released(ask(store, lambda: store.revoke("t", "b01")))


def test_a_late_older_revision_does_not_replace_the_newer_one():
    store = knowledge()
    put(store, doc("p1", "The budget of project P is 700.", family="P", revision=3, day=10), Q)
    put(store, doc("p1", "The budget of project P is 650.", family="P", revision=2, day=12), Q)
    result = ask(store)
    assert result["short_answer"] == "700"


def test_a_document_without_its_vector_is_found_lexically_and_survives_a_restart(tmp_path):
    path = tmp_path / "store.sqlite3"
    store = TemporalStore(path)
    put(store, doc("b0", "Background remark on weather."), -Q)
    store.upsert("t", doc("p1", "The budget of project P is 500.", family="P"))  # no vector yet
    first = ask(store)
    assert first["short_answer"] == "500" and first["audit"][0]["retrieval"]["pending"] == 1
    store.close()
    again = TemporalStore(path)  # restart: caches are rebuilt from the store
    assert ask(again)["short_answer"] == "500"
    put(again, doc("p1", "The budget of project P is 500.", family="P"), Q)
    assert ask(again)["audit"][0]["retrieval"]["pending"] == 0
