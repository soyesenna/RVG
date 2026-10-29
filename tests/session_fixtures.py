"""Shared fixtures of the session API tests: documents, budgets, fake-provider sessions and the
JSON schemas of the public outputs."""

from __future__ import annotations

import contextlib
import json
from importlib.resources import files

import httpx
import jsonschema
from referencing import Registry, Resource

import rvg
from rvg.testing.fake_openrouter import FakeOpenRouter, providers

AS_OF = "2026-09-20T00:00:00Z"


def document(identifier, sentences, *, revision=1, family=None, published="2026-09-01T00:00:00Z"):
    text, units, offset = " ".join(sentences), [], 0
    for i, part in enumerate(sentences):
        units.append({"id": f"u{i}", "start": offset, "end": offset + len(part)})
        offset += len(part) + 1
    return rvg.TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family or identifier,
            "revision": revision,
            "title": identifier.upper(),
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": published,
            "observed_at": published,
            "units": units,
        }
    )


A = document("a", ["The budget of project A is 500.", "It ends in October."])
B = document("b", ["Project B has no budget yet.", "It is led by Kim."])


def budget(seconds=60):
    return rvg.CallContext.with_timeout(
        seconds,
        max_http_attempts=400,
        max_decision_calls=200,
        max_embedding_calls=40,
        max_generation_calls=12,
        max_semantic_proposer_calls=8,
        max_surface_proposer_calls=4,
    )


def question(text="What is the budget of project A?"):
    return {"text": text, "as_of": AS_OF, "known_at": AS_OF}


def context(tenant="fixture", principal=None, **kw):
    return rvg.RequestContext(tenant=tenant, principal=principal, call=budget(), **kw)


@contextlib.asynccontextmanager
async def fake_session(store=None, fake=None, settings=None):
    """A session over an in-memory store (unless given) and the HTTP-level fake OpenRouter."""
    fake = fake or FakeOpenRouter()
    store = store if store is not None else rvg.TemporalStore(":memory:")
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
        p = providers(rvg.JsonTransport(client, timeout=5, max_attempts=3))
        session = rvg.open_session(store, settings or rvg.Settings(_env_file=None), providers=p)
        try:
            yield session, fake, store
        finally:
            await session.aclose()


def _registry():
    root = files("rvg") / "resources" / "schemas" / "api"
    resources = []
    for name in (p.name for p in root.iterdir() if p.name.endswith(".schema.json")):
        body = json.loads((root / name).read_text(encoding="utf-8"))
        resources.append((body["$id"], Resource.from_contents(body)))
    return Registry().with_resources(resources)


REGISTRY = _registry()


def validate(value, name):
    schema = REGISTRY.contents(f"urn:rvg:runtime-api:0.1:{name}")
    jsonschema.Draft202012Validator(schema, registry=REGISTRY).validate(
        json.loads(json.dumps(value))
    )
