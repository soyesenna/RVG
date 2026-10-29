"""A self-contained answer through the public API, the real provider code and the HTTP-level
fake OpenRouter (no network): the smoke the packaging tests and an installed wheel run.

    python tests/packaging/smoke.py        # prints one JSON line
"""

import asyncio
import json

import httpx

import rvg
from rvg.testing.fake_openrouter import FakeOpenRouter, providers


def document(identifier, title, sentences):
    text, units, offset = " ".join(sentences), [], 0
    for i, part in enumerate(sentences):
        units.append({"id": f"u{i}", "start": offset, "end": offset + len(part)})
        offset += len(part) + 1
    return rvg.TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": identifier,
            "revision": 1,
            "title": title,
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": "2026-09-01T00:00:00Z",
            "observed_at": "2026-09-01T00:00:00Z",
            "units": units,
        }
    )


async def answer(text="What is the budget of project A?"):
    store = rvg.TemporalStore(":memory:")
    for d in (
        document("a", "Project A", ["The budget of project A is 500.", "It ends in October."]),
        document("b", "Project B", ["Project B has no budget yet.", "It is led by Kim."]),
    ):
        store.upsert("fixture", d)
    fake = FakeOpenRouter()
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(rvg.JsonTransport(client, timeout=5, max_attempts=3))
            retriever = rvg.TemporalRetriever(store, p["embedding"])
            await retriever.index(
                "fixture", rvg.CallContext.with_timeout(10, max_embedding_calls=20)
            )
            engine = rvg.UniversalEngine(
                retriever=retriever,
                decision=p["decision"],
                writer=p["writer"],
                extractor=p["extractor"],
            )
            question = rvg.TemporalQuestion(
                text=text, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
            )
            context = rvg.CallContext.with_timeout(
                60,
                max_http_attempts=400,
                max_decision_calls=200,
                max_embedding_calls=40,
                max_generation_calls=12,
                max_semantic_proposer_calls=8,
                max_surface_proposer_calls=4,
            )
            return await engine.answer("fixture", question, context)
    finally:
        store.close()


def main():
    result = asyncio.run(answer())
    print(
        json.dumps(
            {
                "status": result["status"],
                "short_answer": result.get("short_answer"),
                "evidence_ids": result.get("evidence_ids"),
                "config_hash": result["config_hash"],
                "rvg_file": rvg.__file__,
            }
        )
    )


if __name__ == "__main__":
    main()
