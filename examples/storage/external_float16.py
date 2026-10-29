"""Answer from an external float16 vector set (`vectors.source: external_float16`;
docs/external-vectors.md), end to end and offline: the real provider code talks to the HTTP-level
fake OpenRouter (no network, no model).

1. A plain SQLite deployment ingests two documents (the fake provider embeds them into the state
   store) and answers a question.
2. `export_external_set` writes the store's vectors as a float16 set: `vectors/manifest.json`,
   one shard and the row hash file. (`write_external_set` writes one from your own
   `(title, text, vector)` rows the same way.)
3. `storage.json` opens a second deployment — a new state store with the same documents — beside
   the set: no document is embedded again, the answer is the same and the index's precision is
   EXACT_NATIVE.

    python examples/storage/external_float16.py [workdir]

Prints one JSON object. The fake provider is a deterministic test double: the statuses show that
the parts connect, not how well any model answers."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import httpx

import rvg
from rvg.storage.external import export_external_set
from rvg.testing.fake_openrouter import FakeOpenRouter, providers

AS_OF = "2026-09-20T00:00:00Z"
TENANT = "example"
DOCUMENTS = [
    ("budget", "Budget", ["The budget of project A is 500.", "It ends soon."]),
    ("lead", "Lead", ["Project B is led by Kim."]),
]
QUESTION = {"text": "What is the budget of project A?", "as_of": AS_OF, "known_at": AS_OF}


def revision(identifier, title, sentences):
    text, units, offset = " ".join(sentences), [], 0
    for i, part in enumerate(sentences):
        units.append({"id": f"u{i}", "start": offset, "end": offset + len(part)})
        offset += len(part) + 1
    return {
        "id": identifier,
        "family": identifier,
        "revision": 1,
        "title": title,
        "text": text,
        "source": f"example://{identifier}",
        "published_at": "2026-09-01T00:00:00Z",
        "observed_at": "2026-09-01T00:00:00Z",
        "units": units,
    }


def ctx(principal=None, request_id=None):
    budget = rvg.CallContext.with_timeout(
        60,
        max_http_attempts=400,
        max_decision_calls=200,
        max_embedding_calls=40,
        max_generation_calls=12,
        max_semantic_proposer_calls=8,
        max_surface_proposer_calls=4,
    )
    return rvg.RequestContext(
        tenant=TENANT, principal=principal, request_id=request_id, call=budget
    )


def embedded(fake, since):
    """The texts the fake provider embedded after request `since`."""
    out = []
    for url, body in fake.bodies[since:]:
        if url.endswith("/embeddings"):
            out += body["input"] if isinstance(body["input"], list) else [body["input"]]
    return out


async def deployment(storage, settings, workdir, transport, *, export_to=None):
    """Ingest the documents and answer the question. With `export_to`, also write the vectors
    the state store now holds as an external set there (before the session closes its store)."""
    async with rvg.open_session(
        storage, settings, root=workdir, providers=providers(transport)
    ) as session:
        ingest = await session.ingest([revision(*d) for d in DOCUMENTS], ctx())
        answer = await session.answer(QUESTION, ctx(rvg.TrustedPrincipal("alice"), "req-1"))
        result, described = answer["result"], session.storage.describe()
        manifest = None
        if export_to is not None:
            manifest = export_external_set(
                session.storage.state, TENANT, session.embedding.describe(), export_to
            )
        return manifest, {
            "ingest": [r["outcome"] for r in ingest["results"]],
            "status": result["status"],
            "short_answer": result.get("short_answer"),
            "evidence_ids": result.get("evidence_ids"),
            "index": described["index"],
            "precision": described["precision"],
        }


async def main(workdir: Path):
    settings = rvg.load_settings(workdir)  # mock mode unless the environment says otherwise
    fake = FakeOpenRouter()
    documents = {f"{title}\n{' '.join(text)}" for _, title, text in DOCUMENTS}
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
        transport = rvg.JsonTransport(client, timeout=5, max_attempts=3)

        # 1. a plain deployment (vectors in the state store); 2. its vectors as a float16 set
        plain = {"schema_version": 1, "state": {"backend": "sqlite", "path": "plain.sqlite3"}}
        manifest, first = await deployment(
            plain, settings, workdir, transport, export_to=workdir / "vectors"
        )
        first["documents_embedded"] = len(documents & set(embedded(fake, 0)))

        # 3. a new deployment beside the set
        config = {
            "schema_version": 1,
            "state": {"backend": "sqlite", "path": "external.sqlite3"},
            "vectors": {
                "source": "external_float16",
                "external": {"manifest": str(manifest.relative_to(workdir))},
            },
        }
        (workdir / "storage.json").write_text(json.dumps(config, indent=2) + "\n")
        since = len(fake.bodies)
        _, second = await deployment(workdir / "storage.json", settings, workdir, transport)
        second["documents_embedded"] = len(documents & set(embedded(fake, since)))

    return {
        "plain": first,
        "external": second,
        "same_answer": (first["status"], first["short_answer"], first["evidence_ids"])
        == (second["status"], second["short_answer"], second["evidence_ids"]),
        "set": sorted(p.name for p in (workdir / "vectors").iterdir()),
    }


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="rvg-ext-"))
    print(json.dumps(asyncio.run(main(target)), ensure_ascii=False))
