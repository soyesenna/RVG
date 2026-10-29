import asyncio
from datetime import datetime

import pytest
from pydantic import ValidationError

from rvg.contracts import EmbeddingSpace
from rvg.providers.mock import MockEmbeddingProvider
from rvg.runtime import CallContext
from rvg.temporal.gate import select_evidence
from rvg.temporal.models import Replacement, SourceSpan, TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore


def stamp(day):
    return datetime.fromisoformat(f"2026-09-{day:02d}T00:00:00+00:00")


def doc(identifier="a", **kwargs):
    text = kwargs.pop("text", "Budget is 500.\nDeadline is September 30.")
    split = text.find("\n")
    units = (
        (SourceSpan(id="u0", start=0, end=len(text)),)
        if split < 0
        else (
            SourceSpan(id="u0", start=0, end=split),
            SourceSpan(id="u1", start=split + 1, end=len(text)),
        )
    )
    return TemporalDocument(
        id=identifier,
        family="f",
        revision=kwargs.pop("revision", 1),
        title="Notice",
        text=text,
        published_at=kwargs.pop("published_at", stamp(1)),
        observed_at=kwargs.pop("observed_at", stamp(1)),
        source="fixture",
        units=units,
        **kwargs,
    )


def question(as_of=20, known_at=20):
    return TemporalQuestion(
        text="What is the budget?", as_of=stamp(as_of), known_at=stamp(known_at)
    )


def test_partial_replacement_preserves_other_field_and_future_revision():
    original = doc()
    correction = doc(
        "b",
        text="Only the budget changes to 600.",
        published_at=stamp(5),
        observed_at=stamp(5),
        supersedes=(Replacement(document_id="a", unit_ids=("u0",)),),
    )
    future = doc(
        "a",
        revision=2,
        text="Budget is 700.",
        published_at=stamp(10),
        observed_at=stamp(10),
        effective_from=stamp(25),
    )
    result = asyncio.run(select_evidence([original, correction, future], question()))
    assert result.statuses == {
        "a@1#u0": "SUPERSEDED",
        "a@1#u1": "ELIGIBLE",
        "b@1#u0": "ELIGIBLE",
        "a@2#u0": "FUTURE_EFFECTIVE",
    }


def test_history_observation_revocation_and_immutable_revision():
    store = TemporalStore(":memory:")
    store.upsert("t", doc())
    store.upsert("t", doc("b", published_at=stamp(5), observed_at=stamp(15)))
    assert [d.id for d in store.documents("t", known_at=stamp(10))] == ["a"]
    assert store.documents("other") == []
    assert [d.id for d in store.documents("t", allowed_ids={"b"})] == ["b"]
    with pytest.raises(ValueError, match="immutable"):
        store.upsert("t", doc(text="Changed."))
    snapshot = store.snapshot("t")
    store.revoke("t", "a")
    assert store.snapshot("t") > snapshot
    assert store.documents("t", known_at=stamp(10)) == []
    store.close()


def test_expiry_boundary_and_naive_times():
    result = asyncio.run(select_evidence([doc(effective_until=stamp(20))], question()))
    assert set(result.statuses.values()) == {"EXPIRED"}
    with pytest.raises(ValidationError):
        doc(published_at=datetime(2026, 9, 1))  # noqa: DTZ001 -- deliberately invalid input


def test_missing_replacement_reference_never_silently_approved():
    result = asyncio.run(
        select_evidence([doc("b", supersedes=(Replacement(document_id="absent"),))], question())
    )
    assert not result.complete


def test_expired_successor_does_not_resurrect_old_revision():
    old = doc()
    new = doc(revision=2, published_at=stamp(5), observed_at=stamp(5), effective_until=stamp(15))
    result = asyncio.run(select_evidence([old, new], question()))
    assert not result.active


def test_lower_authority_cannot_override_primary():
    correction = doc(
        "b",
        text="Budget is 600.",
        published_at=stamp(5),
        observed_at=stamp(5),
        authority="secondary",
        supersedes=(Replacement(document_id="a"),),
    )
    result = asyncio.run(select_evidence([doc(), correction], question()))
    assert not result.complete
    assert result.statuses["a@1#u0"] == "ELIGIBLE"


def test_semantic_amendment_is_scoped_to_unit():
    class Gateway:
        async def ask(self, stage, state, questions):
            return {
                key: {
                    "choice": "REPLACES"
                    if "Budget" in state["pairs"][i]["older"]["text"]
                    else "UNRELATED",
                    "probabilities": {
                        "REPLACES": 1.0 if i == 0 else 0.0,
                        "UNRELATED": 0.0 if i == 0 else 1.0,
                        "UNKNOWN": 0.0,
                        "CONFLICTS": 0.0,
                    },
                }
                for i, key in enumerate(questions)
            }

    amendment = doc(
        "b",
        text="Only budget changes to 600; other terms remain.",
        published_at=stamp(5),
        observed_at=stamp(5),
    )
    result = asyncio.run(select_evidence([doc(), amendment], question(), gateway=Gateway()))
    assert result.statuses["a@1#u0"] == "SUPERSEDED"
    assert result.statuses["a@1#u1"] == "ELIGIBLE"


def test_pending_overlay_and_incremental_index_are_available():
    async def run():
        store = TemporalStore(":memory:")
        space = EmbeddingSpace(
            "mock", "mock", "mock://", 8, True, "cosine", "float", "", "", None, None, 1
        )
        retriever = TemporalRetriever(store, MockEmbeddingProvider(space))
        context = CallContext.with_timeout(10, max_embedding_calls=20)
        store.upsert("t", doc())
        assert (await retriever.index("t", context))["indexed"] == 1
        assert (await retriever.index("t", context))["indexed"] == 0
        store.upsert(
            "t",
            doc("b", text="Budget correction to 600.", published_at=stamp(5), observed_at=stamp(5)),
        )
        docs, info = await retriever.search("t", question(), context)
        assert {d.id for d in docs} == {"a", "b"}
        assert info["pending"] == 1
        assert (await retriever.index("t", context))["indexed"] == 1
        store.revoke("t", "b")
        docs, _ = await retriever.search("t", question(), context)
        assert [d.id for d in docs] == ["a"]
        store.close()

    asyncio.run(run())


def test_family_expansion_recovers_unranked_amendment_without_acl_leak():
    async def run():
        store = TemporalStore(":memory:")
        store.upsert("t", doc(text="Budget budget budget is 500."))
        store.upsert(
            "t",
            doc(
                "b",
                text="Only the monetary ceiling changes to 600.",
                published_at=stamp(5),
                observed_at=stamp(5),
            ),
        )
        retriever = TemporalRetriever(store)
        ctx = CallContext.with_timeout(5)
        seeds, _ = await retriever.search(
            "t", question(), ctx, strategy="lexical", top_k=1, expand=False
        )
        assert [d.id for d in seeds] == ["a"]
        closure, _ = await retriever.search("t", question(), ctx, strategy="lexical", top_k=1)
        assert {d.id for d in closure} == {"a", "b"}
        scoped = TemporalRetriever(store, allowed_ids={"a"})
        visible, _ = await scoped.search("t", question(), ctx, strategy="lexical", top_k=1)
        assert [d.id for d in visible] == ["a"]
        store.close()

    asyncio.run(run())
