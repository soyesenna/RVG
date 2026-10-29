"""K8-1 (docs/K8_1_PROTOCOL.md): long revisions are retrieved by segment. A search returns the
segment that matches, not the whole revision; the family expansion takes, from another revision
of a long document, only the segment in the same place (R2); the revision judge sees only the
selected segments (R3); a revision that cannot be segmented without cutting is blocked, listed
and never embedded; a revision that fits is one row exactly as before (K8-0 identity)."""

import asyncio

import numpy as np
import pytest

from rvg.contracts import EmbeddingBatch, EmbeddingSpace
from rvg.errors import ContractError
from rvg.runtime import CallContext
from rvg.storage.identity import UnitKey
from rvg.storage.segments import SegmentView
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore

TOPICS = ("budget", "staff", "schedule", "risk", "vendor", "audit")
DIMS = len(TOPICS) + 1
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


def vector_of(text):
    """Word counts of the topics (plus a constant): a segment about one topic points at it."""
    words = text.lower().split()
    return (*(float(words.count(t)) for t in TOPICS), 0.1)


class Topics:
    mode = "fixture"

    def __init__(self):
        self.embedded = 0

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        self.embedded += len(texts) if purpose == "document" else 0
        return EmbeddingBatch(tuple(vector_of(t) for t in texts), SPACE, ("f",), ())


def long_document(identifier, *, revision=1, figure=500, family=None, per_topic=6, day=1):
    """Six topic sections of `per_topic` sentences each, about 1 KB per sentence: well over the
    8 KB embedding input, so several segments."""
    text, units = "", []
    for t, topic in enumerate(TOPICS):
        for k in range(per_topic):
            value = figure if topic == "budget" and k == 0 else 100 + t * 10 + k
            sentence = f"The {topic} figure {k} is {value}. " + " ".join([topic] * 150) + "."
            start = len(text) + (1 if text else 0)
            text += (" " if text else "") + sentence
            units.append({"id": f"u{t}x{k}", "start": start, "end": start + len(sentence)})
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family or identifier,
            "revision": revision,
            "title": f"Report {identifier}",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": f"2026-09-{day:02d}T00:00:00Z",
            "observed_at": f"2026-09-{day:02d}T00:00:00Z",
            "units": units,
        }
    )


def short_document(identifier, text, *, family=None, revision=1):
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family or identifier,
            "revision": revision,
            "title": f"Note {identifier}",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": "2026-09-01T00:00:00Z",
            "observed_at": "2026-09-01T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
        }
    )


def search(store, text, *, top_k=1):
    async def go():
        embedding = Topics()
        retriever = TemporalRetriever(store, embedding)
        report = await retriever.index("t", CallContext.with_timeout(30, max_embedding_calls=20))
        q = TemporalQuestion(
            text=text, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
        )
        docs, info = await retriever.search(
            "t", q, CallContext.with_timeout(30, max_embedding_calls=4), top_k=top_k
        )
        return docs, info, report, retriever

    return asyncio.run(go())


def test_a_long_revision_is_found_by_the_segment_that_matches():
    store = TemporalStore(":memory:")
    store.upsert("t", long_document("r"))
    docs, info, report, _ = search(store, "what is the staff figure")
    (doc,) = docs
    assert isinstance(doc, SegmentView) and doc.key == "r@1"
    assert info["seed_ids"][0].startswith("r@1~s")
    assert len(doc.segments) == 1 and len(doc.units) < len(doc.document.units)
    assert any(u.id.startswith("u1") for u in doc.units)  # the staff section is in it
    assert [e["id"] for e in doc.evidence()] == [f"r@1#{u.id}" for u in doc.units]
    assert report["indexed"] == report["pending_at_start"] > 1  # one vector per segment


def test_another_revision_contributes_only_its_segment_in_the_same_place():
    store = TemporalStore(":memory:")
    store.upsert("t", long_document("r", revision=1, figure=500, day=1))
    store.upsert("t", long_document("r", revision=2, figure=700, day=5))
    docs, info, _, _ = search(store, "what is the budget figure", top_k=1)
    by_key = {d.key: d for d in docs}
    assert set(by_key) == {"r@1", "r@2"}
    seed = next(d for d in docs if d.key + "~" in info["seed_ids"][0] + "~")
    other = next(d for d in docs if d is not seed)
    assert len(other.segments) == len(seed.segments) == 1  # not every segment of r@1
    assert other.segments[0].ordinal == seed.segments[0].ordinal  # the same place
    assert [u.id for u in other.units] == [u.id for u in seed.units]
    dumped = other.model_dump(mode="json")
    assert len(dumped["text"]) < len(other.document.text)  # R3: never the whole revision
    assert "The budget figure 0 is" in dumped["text"]


def test_a_revision_that_cannot_be_segmented_is_blocked_listed_and_never_embedded():
    store = TemporalStore(":memory:")
    too_long = short_document("x", "risk " * 2000 + "end.")  # one unit over the limit
    store.upsert("t", too_long)
    store.upsert("t", short_document("y", "The risk figure is 3."))
    docs, _, report, retriever = search(store, "what is the risk figure", top_k=5)
    assert [d.key for d in docs] == ["y@1"] and report["blocked"] == 1
    catalog = asyncio.run(retriever.storage.catalogs.get("t"))
    assert catalog.blocked == {"x@1": "unsupported_segment:unit_too_long:u0"}


def test_a_revision_that_fits_keeps_its_k8_0_identity_and_vector():
    store = TemporalStore(":memory:")
    d = short_document("y", "The staff figure is 12.")
    store.upsert("t", d)
    store.put_vector("t", SPACE.fingerprint, TemporalRetriever.input_text(d), vector_of(d.text))
    docs, info, report, retriever = search(store, "staff")
    assert docs == [d] and info["seed_ids"] == ["y@1"] and report["pending_at_start"] == 0
    catalog = asyncio.run(retriever.storage.catalogs.get("t"))
    assert catalog.unit_keys == (UnitKey("t", "y", 1),)
    assert catalog.row_keys == ("y@1",)


def test_segments_are_distinct_index_points():
    store = TemporalStore(":memory:")
    store.upsert("t", long_document("r"))
    *_, retriever = search(store, "budget")
    catalog = asyncio.run(retriever.storage.catalogs.get("t"))
    points = {k.point_id(SPACE.fingerprint) for k in catalog.unit_keys}
    assert len(points) == len(catalog) > 1
    assert {k.segmentation for k in catalog.unit_keys} == {"units-greedy/1"}


def test_the_overflow_bound_is_unchanged_for_whole_revisions():
    store = TemporalStore(":memory:")
    for k in range(30):
        store.upsert("t", short_document(f"f{k}", f"Budget note {k}: budget {k}.", family="F"))
    with pytest.raises(ContractError, match="temporal_context_overflow"):
        search(store, "budget")


def test_views_of_one_revision_merge_into_one_document():
    store = TemporalStore(":memory:")
    store.upsert("t", long_document("r"))
    docs, info, _, _ = search(store, "budget staff", top_k=2)
    assert len(docs) == 1 and len(info["seed_ids"]) == 2
    assert len(docs[0].segments) == 2
    assert np.all(np.diff([u.start for u in docs[0].units]) > 0)


def test_the_engine_answers_from_one_segment_of_a_long_revision():
    import httpx
    from test_n3_release import writer_with

    from rvg.testing.fake_openrouter import FakeOpenRouter, providers
    from rvg.transport import JsonTransport
    from rvg.universal.engine import UniversalEngine

    store = TemporalStore(":memory:")
    store.upsert("t", long_document("r"))

    async def go():
        embedding = Topics()
        retriever = TemporalRetriever(store, embedding)
        await retriever.index("t", CallContext.with_timeout(30, max_embedding_calls=20))
        fake = FakeOpenRouter(writer=writer_with(lambda: None))
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(JsonTransport(client, timeout=5, max_attempts=2), dims=DIMS)
            engine = UniversalEngine(
                retriever=retriever,
                decision=p["decision"],
                writer=p["writer"],
                extractor=p["extractor"],
            )
            q = TemporalQuestion(
                text="What is the vendor figure?",
                as_of="2026-09-20T00:00:00Z",
                known_at="2026-09-20T00:00:00Z",
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

    result = asyncio.run(go())
    assert result["status"] == "FULL"
    shown = result["evidence_ids"]
    assert shown and all(e.startswith("r@1#") for e in shown)
    assert len(shown) < len(long_document("r").units)  # a segment, not the whole revision
    assert any(e.startswith("r@1#u4") for e in shown)  # the vendor section
