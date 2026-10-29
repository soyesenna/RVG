"""A consumer that uses only the public API (`import rvg`) — the shape of a benchmark submission
adapter. It runs every neutral operation once against a fresh SQLite deployment, with the real
provider code talking to the HTTP-level fake OpenRouter (no network, no model), and prints what
each operation returned, one JSON object.

    python examples/api/public_consumer.py [workdir]

The fake provider is a deterministic test double: the statuses show that the operations connect,
not how well any model answers."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import httpx

import rvg
from rvg.testing.fake_openrouter import FakeOpenRouter, providers

AS_OF = "2026-09-20T00:00:00Z"


def revision(identifier, sentences):
    text, units, offset = " ".join(sentences), [], 0
    for i, part in enumerate(sentences):
        units.append({"id": f"u{i}", "start": offset, "end": offset + len(part)})
        offset += len(part) + 1
    return {
        "id": identifier,
        "family": identifier,
        "revision": 1,
        "title": identifier.upper(),
        "text": text,
        "source": f"example://{identifier}",
        "published_at": "2026-09-01T00:00:00Z",
        "observed_at": "2026-09-01T00:00:00Z",
        "units": units,
    }


async def main(workdir: Path):
    storage = {"schema_version": 1, "state": {"backend": "sqlite", "path": "state.sqlite3"}}
    settings = rvg.load_settings(workdir)  # mock mode unless the environment says otherwise
    fake = FakeOpenRouter()
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
        transport = rvg.JsonTransport(client, timeout=5, max_attempts=3)
        async with rvg.open_session(
            storage, settings, root=workdir, providers=providers(transport)
        ) as session:
            staff = rvg.TrustedPrincipal("alice", groups=("staff",))  # from the auth layer

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
                    tenant="example", principal=principal, request_id=request_id, call=budget
                )

            out = {"api_version": rvg.API_VERSION}
            out["capabilities_policy"] = session.capabilities()["policy"]["sha256"]
            out["runtime"] = session.describe_runtime()["distribution"]
            ingest = await session.ingest(
                [
                    rvg.IngestItem(
                        document=rvg.TemporalDocument.model_validate(
                            revision("budget", ["The budget of project A is 500.", "It ends soon."])
                        ),
                        access=rvg.DocumentAccess(allow=("staff",)),
                    ),
                    revision("lead", ["Project B is led by Kim."]),
                ],
                ctx(),
            )
            out["ingest"] = [r["outcome"] for r in ingest["results"]]
            out["readiness"] = {k: v["state"] for k, v in ingest["readiness"].items()}
            event = await session.apply_event(
                {"type": "revoke", "event_id": "ev-1", "document_id": "lead"}, ctx()
            )
            out["event"] = event["outcome"]
            question = {
                "text": "What is the budget of project A?",
                "as_of": AS_OF,
                "known_at": AS_OF,
            }
            answer = await session.answer(question, ctx(staff, "req-1"))
            out["answer"] = {
                "status": answer["result"]["status"],
                "short_answer": answer["result"].get("short_answer"),
                "evidence_ids": answer["result"].get("evidence_ids"),
            }
            outsider = await session.answer(question, ctx(rvg.TrustedPrincipal("bob"), "req-2"))
            out["answer_without_permission"] = outsider["result"]["status"]
            out["health"] = {
                level: (await session.health(level, ctx())).get("status")
                for level in ("liveness", "readiness")
            }
            lineage = await session.export_artifact_lineage("req-1", ctx(staff))
            out["lineage"] = {
                "completeness": lineage["completeness"],
                "kinds": sorted({a["kind"] for a in lineage["artifacts"]}),
            }
            try:
                await session.export_artifact_lineage("req-1", ctx(rvg.TrustedPrincipal("bob")))
            except rvg.OperationError as exc:
                out["lineage_of_another_principal"] = exc.to_dict()["error"]["native_code"]
    out["fake_provider_requests"] = len(fake.bodies)
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="rvg-api-"))
    print(json.dumps(asyncio.run(main(target)), ensure_ascii=False))
