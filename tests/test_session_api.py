"""SEP-04 A1 (G04-API, G04-OBS): the runtime session delegates to the existing engine, reports
commits and readiness honestly, and observes without changing anything or leaking what the caller
may not see."""

import ast
import asyncio
import json
from pathlib import Path

import httpx
import pytest
from session_fixtures import A, B, budget, context, document, fake_session, question, validate

import rvg
from rvg.testing.fake_openrouter import FakeOpenRouter, providers

ALICE = rvg.TrustedPrincipal("alice")
BOB = rvg.TrustedPrincipal("bob")


def run(coro):
    return asyncio.run(coro)


# --- G04-API: delegation ------------------------------------------------------------------------


def test_the_session_answer_is_the_engine_answer_unchanged():
    async def through_session():
        async with fake_session() as (session, _, _):
            await session.ingest([A, B], context())
            return await session.answer(question(), context(request_id="r1"))

    async def direct():
        store, fake = rvg.TemporalStore(":memory:"), FakeOpenRouter()
        for d in (A, B):
            store.upsert("fixture", d)
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(rvg.JsonTransport(client, timeout=5, max_attempts=3))
            retriever = rvg.TemporalRetriever(store, p["embedding"])
            await retriever.index("fixture", budget())
            engine = rvg.UniversalEngine(
                retriever=retriever,
                decision=p["decision"],
                writer=p["writer"],
                extractor=p["extractor"],
            )
            q = rvg.TemporalQuestion.model_validate(question())
            return await engine.answer("fixture", q, budget())

    envelope, engine_result = run(through_session()), run(direct())
    assert envelope["result"] == engine_result
    assert envelope["result"]["status"] == "FULL" and envelope["request_id"] == "r1"
    validate(envelope, "answer_envelope")


def test_the_engine_is_called_once_per_answer(monkeypatch):
    calls = []
    original = rvg.UniversalEngine.answer

    async def counted(self, *args, **kwargs):
        calls.append(1)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(rvg.UniversalEngine, "answer", counted)

    async def go():
        async with fake_session() as (session, _, _):
            await session.ingest([A, B], context())
            await session.answer(question(), context())

    run(go())
    assert calls == [1]


def test_the_session_module_assembles_objects_and_decides_nothing():
    source = Path(rvg.session.__file__ if hasattr(rvg, "session") else "").read_text()
    imported = {
        n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom) and n.module
    }
    decision_modules = {"control", "temporal.gate", "approval.core", "grounded.engine"}
    assert not {m for m in imported if any(m.startswith(d) for d in decision_modules)}


# --- ingest and events --------------------------------------------------------------------------


def test_ingest_reports_stored_duplicate_and_rejected_separately():
    changed = document("a", ["The budget of project A is 900.", "It ends in October."])
    early = {**A.model_dump(mode="json"), "id": "c", "observed_at": "2026-08-01T00:00:00Z"}

    async def go():
        async with fake_session() as (session, _, _):
            first = await session.ingest([A, B], context())
            again = await session.ingest([A, changed, early], context(), prepare_vectors=False)
            return first, again

    first, again = run(go())
    assert [r["outcome"] for r in first["results"]] == ["STORED", "STORED"]
    assert first["readiness"]["durable_commit"]["state"] == "READY"
    assert first["readiness"]["embedding_ready"]["state"] == "READY"
    assert first["readiness"]["index_visible"]["reason"] == "query_time_view:local_exact"
    assert [r["outcome"] for r in again["results"]] == ["DUPLICATE", "REJECTED", "REJECTED"]
    assert again["results"][1]["reason"] == "immutable_temporal_revision_changed"
    assert again["results"][2]["reason"].endswith("observation_before_publication")
    assert again["knowledge_seq"] == first["knowledge_seq"] == 2  # nothing new was stored
    for result in (first, again):
        validate(result, "ingest_result")


def test_vectors_not_prepared_are_reported_pending_not_ready():
    async def go():
        async with fake_session() as (session, fake, _):
            result = await session.ingest([A], context(), prepare_vectors=False)
            return result, len(fake.bodies)

    result, calls = run(go())
    assert calls == 0  # no embedding call was made
    assert result["vectors"]["status"] == "NOT_REQUESTED" and result["vectors"]["pending"] == 1
    assert result["readiness"]["embedding_ready"]["state"] == "NOT_READY"
    assert "lexical_fallback" in result["readiness"]["index_visible"]["reason"]


def test_events_revoke_and_replace_access_with_content_duplicates():
    async def go():
        async with fake_session() as (session, _, store):
            await session.ingest([A, B], context())
            out = []
            for event in (
                {
                    "type": "access",
                    "event_id": "e1",
                    "document_id": "a",
                    "access": {"allow": ["alice"]},
                },
                {
                    "type": "access",
                    "event_id": "e2",
                    "document_id": "a",
                    "access": {"allow": ["alice"]},
                },
                {"type": "revoke", "event_id": "e3", "document_id": "b"},
                {"type": "revoke", "event_id": "e4", "document_id": "b"},
                {"type": "access", "event_id": "e5", "document_id": "a", "access": None},
            ):
                out.append(await session.apply_event(event, context()))
            return out, store.hidden("fixture", BOB), store.revoked("fixture")

    results, hidden_from_bob, revoked = run(go())
    assert [r["outcome"] for r in results] == [
        "APPLIED",
        "DUPLICATE",
        "APPLIED",
        "DUPLICATE",
        "APPLIED",
    ]
    assert results[0]["access"] == {"changed": 1, "acl_epoch": 1}
    assert results[2]["event_seq"] == 3 and results[3]["event_seq"] is None
    assert hidden_from_bob == set() and revoked == {"b"}  # the ACL was cleared at the end
    for result in results:
        validate(result, "event_result")


def test_an_acl_that_allows_nobody_is_an_explicit_deny_all():
    """RVI-07A (intended change): `allow=()` was UNSUPPORTED_CAPABILITY; it is now stored as a
    deny-all that hides the document from every principal, group and anonymous caller."""

    async def go():
        async with fake_session() as (session, _, store):
            await session.ingest([A], context())
            event = await session.apply_event(
                {"type": "access", "document_id": "a", "access": {"allow": []}}, context()
            )
            item = {"document": B.model_dump(mode="json"), "access": {"allow": []}}
            row = (await session.ingest([item], context()))["results"][0]
            seen = {
                who: {"a", "b"} - store.hidden("fixture", p)
                for who, p in (
                    ("staff", rvg.TrustedPrincipal("staff", ("employees",))),
                    ("none", None),
                )
            }
            return event, row, seen

    event, row, seen = run(go())
    assert event["outcome"] == "APPLIED" and event["access"]["changed"] == 1
    assert row["outcome"] == "STORED"
    assert seen == {"staff": set(), "none": set()}
    validate(event, "event_result")


def test_an_unknown_event_type_is_an_operation_error():
    async def go():
        async with fake_session() as (session, _, _):
            with pytest.raises(rvg.OperationError) as caught:
                await session.apply_event({"type": "delete", "document_id": "a"}, context())
            return caught.value

    assert run(go()).code == "INVALID_INPUT"


# --- trust boundary -----------------------------------------------------------------------------


def test_a_principal_is_an_object_from_the_trusted_context_never_a_string():
    async def go():
        async with fake_session() as (session, _, _):
            with pytest.raises(rvg.OperationError) as caught:
                await session.answer(question(), context(principal="alice"))
            with pytest.raises(rvg.OperationError) as extra:
                await session.answer(question() | {"principal_id": "admin"}, context())
            with pytest.raises(rvg.OperationError) as gold:
                await session.answer(question() | {"expected_answer": "500"}, context())
            return caught.value, extra.value, gold.value

    principal, extra, gold = run(go())
    assert principal.code == "FORBIDDEN_INPUT"
    assert extra.code == gold.code == "FORBIDDEN_INPUT"  # a request carries no identity or gold


def test_acl_filters_retrieval_and_the_selection_only_narrows():
    async def go():
        async with fake_session() as (session, _, _):
            item = {"document": A.model_dump(mode="json"), "access": {"allow": ["alice"]}}
            await session.ingest([item, B], context())
            as_alice = await session.answer(question(), context(principal=ALICE))
            as_bob = await session.answer(question(), context(principal=BOB))
            narrowed = await session.answer(
                question(), context(principal=ALICE, selected_document_ids=("b",))
            )
            return as_alice, as_bob, narrowed

    as_alice, as_bob, narrowed = run(go())
    assert as_alice["result"]["status"] == "FULL"
    assert as_bob["result"]["status"] == "ABSTAIN"
    assert narrowed["result"]["status"] == "ABSTAIN"


# --- lineage ------------------------------------------------------------------------------------


def test_lineage_names_the_released_units_and_their_vector_input():
    async def go():
        async with fake_session() as (session, _, _):
            await session.ingest([A, B], context())
            envelope = await session.answer(question(), context(request_id="q1"))
            return envelope, await session.export_artifact_lineage("q1", context())

    envelope, lineage = run(go())
    validate(lineage, "lineage")
    assert lineage["completeness"] == "COMPLETE" and lineage["missing_reasons"] == []
    units = [a for a in lineage["artifacts"] if a["kind"] == "evidence_unit"]
    vectors = [a for a in lineage["artifacts"] if a["kind"] == "vector"]
    result = envelope["result"]
    cited = {u for s in result["sentences"] for u in s["evidence_ids"]}
    assert {u["ref"] for u in units} == cited  # the cited units, not the writer's window
    assert set(result["evidence_ids"]) > cited  # (0.1.0: the window also has uncited units)
    assert {v["document_id"] for v in vectors} == {u.split("@")[0] for u in cited}
    a0 = next(u for u in units if u["ref"] == "a@1#u0")
    vector = next(v for v in vectors if v["derived_from"] == ["a@1#u0"])
    assert a0["span"] == {"start": 0, "end": 31}
    assert vector["state"] == "PRESENT" and vector["ref"] == "a@1#s0"
    assert vector["source_sha256"] == a0["source_sha256"]
    assert lineage["release"] == envelope["result"]["release"]


@pytest.mark.parametrize("change", ["revoke", "access"])
def test_lineage_hides_a_source_revoked_or_restricted_after_the_answer(change):
    async def go():
        async with fake_session() as (session, _, _):
            item = {"document": A.model_dump(mode="json"), "access": {"allow": ["alice"]}}
            await session.ingest([item, B], context())
            await session.answer(question(), context(principal=ALICE, request_id="q2"))
            event = (
                {"type": "revoke", "document_id": "a"}
                if change == "revoke"
                else {"type": "access", "document_id": "a", "access": {"allow": ["bob"]}}
            )
            await session.apply_event(event, context())
            return await session.export_artifact_lineage("q2", context(principal=ALICE))

    lineage = run(go())
    dumped = json.dumps(lineage)
    assert lineage["completeness"] == "PARTIAL"
    assert lineage["missing_reasons"] == ["ACCESS_RESTRICTED_OR_REVOKED"]
    # the answer cites a only (0.1.0: b was in the writer's window, uncited, and is not listed)
    assert lineage["artifacts"] == []
    sha_of_a = __import__("hashlib").sha256(A.text.encode()).hexdigest()
    assert "a@1" not in dumped and "fixture://a" not in dumped and sha_of_a not in dumped
    validate(lineage, "lineage")


def test_lineage_of_another_principal_tenant_or_unknown_request_is_unknown():
    async def go():
        async with fake_session() as (session, _, _):
            await session.ingest([A, B], context())
            await session.answer(question(), context(principal=ALICE, request_id="q3"))
            codes = []
            for ref, ctx in (
                ("q3", context(principal=BOB)),
                ("q3", context(tenant="other", principal=ALICE)),
                ("nope", context(principal=ALICE)),
            ):
                with pytest.raises(rvg.OperationError) as caught:
                    await session.export_artifact_lineage(ref, ctx)
                codes.append(caught.value.native_code)
            return codes

    assert run(go()) == ["lineage_reference_unknown"] * 3


def test_an_unreleased_answer_has_no_evidence_to_export():
    async def go():
        async with fake_session() as (session, _, _):
            await session.ingest([B], context())
            envelope = await session.answer(question("Who leads project B?"), context())
            lineage = await session.export_artifact_lineage(envelope["request_id"], context())
            return envelope, lineage

    envelope, lineage = run(go())
    assert envelope["result"]["status"] == "ABSTAIN"
    assert lineage["artifacts"] == [] and lineage["notes"] == ["no_released_evidence"]


def test_a_segmented_revision_names_the_segment_that_holds_the_unit():
    filler = "This paragraph describes the general background of the programme in detail."
    sentences = [f"{filler} Part {chr(97 + i % 26)}{chr(97 + i // 26)}." for i in range(160)]
    sentences[120] = "The budget of project Z is 700."
    long = document("z", sentences)

    async def go():
        async with fake_session() as (session, _, _):
            await session.ingest([long], context())
            record = {
                "tenant": "fixture",
                "principal_id": None,
                "status": "FULL",
                "evidence_ids": ("z@1#u120",),
                "knowledge_snapshot": 1,
                "release": None,
                "config_hash": None,
                "storage": None,
                "recorded_at": "2026-09-25T00:00:00Z",
            }
            return session._lineage_of("q4", record, context())

    lineage = run(go())
    vector = next(a for a in lineage["artifacts"] if a["kind"] == "vector")
    unit = next(a for a in lineage["artifacts"] if a["kind"] == "evidence_unit")
    assert vector["ref"].startswith("z@1#s") and vector["ref"] != "z@1#s0"
    assert (
        vector["span"]["start"]
        <= unit["span"]["start"]
        < unit["span"]["end"]
        <= vector["span"]["end"]
    )
    assert vector["state"] == "PRESENT"


# --- health, description, lifecycle -------------------------------------------------------------


def state_of(store):
    state = store.storage.state
    return (state.generations("fixture"), state.outbox_mark("fixture"), state.digest("fixture"))


def test_health_reads_and_changes_nothing():
    async def go():
        async with fake_session() as (session, _, store):
            await session.ingest([A, B], context())
            before = state_of(store)
            live = await session.health("liveness")
            ready = await session.health("readiness", context())
            return before, state_of(store), live, ready

    before, after, live, ready = run(go())
    assert before == after
    assert live["status"] == "OK" and ready["status"] == "OK"
    assert {c["component"] for c in live["checks"]} == {
        "session",
        "policy_resource",
        "prompt_resources",
        "providers",
        "provider_reachability",
    }
    assert "state_store" in {c["component"] for c in ready["checks"]}
    assert live["unverified"] == ["provider_reachability"]
    validate(live, "health")
    validate(ready, "health")


def test_readiness_of_a_broken_store_is_not_ready_with_a_type_name_only():
    async def go():
        async with fake_session() as (session, _, store):
            store.storage.state.conn.close()
            return await session.health("readiness", context())

    ready = run(go())
    check = next(c for c in ready["checks"] if c["component"] == "state_store")
    assert ready["status"] == "NOT_READY" and check["code"] == "read_failed:ProgrammingError"


def test_description_and_capabilities_carry_no_secret():
    secret = "sk-or-v1-" + "x" * 40
    settings = rvg.Settings(_env_file=None, openrouter_api_key=secret)

    async def go():
        async with fake_session(settings=settings) as (session, _, _):
            return session.describe_runtime(), session.capabilities()

    described, caps = run(go())
    validate(described, "runtime_description")
    assert secret not in json.dumps(described) and secret not in json.dumps(caps)
    assert described["policy"]["sha256"] == rvg.policy_hash()
    assert described["install"]["source_commit"] is None
    assert caps["session_operations"]["fixed_draft_verification"]["implemented"] is False
    assert caps["session"]["availability"] == "UNKNOWN"


def test_closing_is_idempotent_and_later_operations_fail_not_ready():
    async def go():
        async with fake_session() as (session, _, _):
            await session.aclose()
            await session.aclose()
            with pytest.raises(rvg.OperationError) as caught:
                await session.ingest([A], context())
            live = await session.health("liveness")
            return caught.value, live

    error, live = run(go())
    assert error.code == "NOT_READY" and error.native_code == "session_closed"
    assert live["status"] == "NOT_READY"


def test_a_session_closes_only_what_it_opened(tmp_path):
    config = {"schema_version": 1, "state": {"backend": "sqlite", "path": "s.sqlite3"}}

    async def go():
        fake = FakeOpenRouter()
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(rvg.JsonTransport(client, timeout=5, max_attempts=3))
            owned = rvg.open_session(
                config, rvg.Settings(_env_file=None), root=tmp_path, providers=p
            )
            await owned.aclose()
            given = rvg.TemporalStore(":memory:")
            borrowed = rvg.open_session(given, rvg.Settings(_env_file=None), providers=p)
            await borrowed.aclose()
            return owned, given

    owned, given = run(go())
    with pytest.raises(Exception):  # noqa: B017 - the owned SQLite connection is closed
        owned.storage.state.snapshot("fixture")
    assert given.snapshot("fixture") == 0  # the caller's store is still open
