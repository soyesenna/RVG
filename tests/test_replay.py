"""N0: an answer is a deterministic function of (code, policy, sources, request, recorded provider
exchanges). Recording at the HTTP-attempt level and replaying through the unchanged provider
stack must reproduce every decision field; a missing exchange is an ERROR, never a live call."""

import asyncio
import json
import time

import httpx

from rvg.runtime import CallContext
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore
from rvg.testing.fake_openrouter import FakeOpenRouter, providers
from rvg.testing.records import answer_one, compare, manifest, signature
from rvg.universal import universal_hash
from rvg.universal.engine import UniversalEngine
from rvg.universal.replay import RecordingTransport, ReplayTransport, Tape, exchange_key
from rvg.universal.trace import ReplayOracle, canonical_bytes

TENANT = "fixture-tenant"


def document(identifier, title, sentences, *, day="01"):
    text = " ".join(sentences)
    units, offset = [], 0
    for i, part in enumerate(sentences):
        units.append({"id": f"u{i}", "start": offset, "end": offset + len(part)})
        offset += len(part) + 1
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": identifier,
            "revision": 1,
            "title": title,
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": f"2026-09-{day}T00:00:00Z",
            "observed_at": f"2026-09-{day}T00:00:00Z",
            "units": units,
        }
    )


DOCS = [
    document("a", "Project A", ["The budget of project A is 500.", "It ends in October."]),
    document("b", "Project B", ["Project B has no budget yet."]),
    document("c", "Project C", ["The budget of project C is 720.", "It started in May."]),
]
QUESTIONS = {
    "q1": "What is the budget of project A?",
    "q2": "What is the budget of project C?",
    "q3": "When does project B end?",
}


def question(text):
    return TemporalQuestion(
        text=text, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z", locale="en-US"
    )


def context(clock=None):
    extra = {"clock": clock} if clock is not None else {}
    return CallContext.with_timeout(
        60,
        **extra,
        max_http_attempts=400,
        max_decision_calls=200,
        max_embedding_calls=40,
        max_generation_calls=12,
        max_semantic_proposer_calls=8,
        max_surface_proposer_calls=4,
        max_schema_repairs=1,
    )


def store_with(docs=DOCS):
    store = TemporalStore(":memory:")
    for doc in docs:
        store.upsert(TENANT, doc)
    return store


async def run(store, transport, questions, *, tape=None, index=False, clock=None, oracles=None):
    p = providers(transport)
    retriever = TemporalRetriever(store, p["embedding"])
    if index:
        await retriever.index(TENANT, context())
    engine = UniversalEngine(
        retriever=retriever, decision=p["decision"], writer=p["writer"], extractor=p["extractor"]
    )
    return await asyncio.gather(
        *[
            answer_one(
                engine,
                TENANT,
                question(text),
                request_id=rid,
                context=context(clock),
                tape=tape,
                oracle=(oracles or {}).get(rid),
            )
            for rid, text in questions.items()
        ]
    )


def record(fake, questions=QUESTIONS, docs=DOCS):
    async def go():
        store, tape = store_with(docs), Tape()
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            transport = RecordingTransport(client, tape, timeout=5, max_attempts=3)
            records = await run(store, transport, questions, tape=tape, index=True)
        return store, tape, records

    return asyncio.run(go())


def replay(store, tape, questions=QUESTIONS, *, jitter_seed=None, recorded=None):
    transport = ReplayTransport(tape, jitter_seed=jitter_seed)
    oracles = {r["request_id"]: ReplayOracle(r["trace_nodes"]) for r in recorded or []}
    records = asyncio.run(run(store, transport, questions, clock=lambda: 0.0, oracles=oracles))
    return transport, records


def test_replay_reproduces_every_recorded_decision_and_uses_every_exchange():
    store, tape, recorded = record(FakeOpenRouter())
    statuses = {r["request_id"]: r["result"]["status"] for r in recorded}
    assert statuses["q1"] == "FULL" and statuses["q2"] == "FULL"
    transport, replayed = replay(store, tape)
    for a, b in zip(recorded, replayed, strict=True):
        assert compare(a, b) == {
            "equal": True,
            "fields": [],
            "max_float_difference": 0.0,
            "trace_nodes_equal": True,
        }
        assert a["usage"] | {"reported_cost_usd": 0} == b["usage"] | {"reported_cost_usd": 0}
        assert transport.unused(a["request_id"]) == []


def test_replay_with_shuffled_completion_order_is_identical():
    store, tape, recorded = record(FakeOpenRouter())
    for seed in (1, 2, 3):
        _, shuffled = replay(store, tape, jitter_seed=seed)
        assert [signature(r) for r in shuffled] == [signature(r) for r in recorded]


def test_a_missing_exchange_is_an_error_and_never_a_live_call():
    store, tape, recorded = record(FakeOpenRouter(), {"q1": QUESTIONS["q1"]})
    writer_calls = [
        e for e in tape.entries if e["request_id"] == "q1" and e["url"].endswith("/completions")
    ]
    kept = Tape([e for e in tape.entries if e not in writer_calls])
    # The replay transport has no HTTP client: a fallback to a live call could not even start.
    _, replayed = replay(store, kept, {"q1": QUESTIONS["q1"]})
    assert replayed[0]["result"]["status"] == "ERROR"
    assert replayed[0]["result"]["reason"] == "replay_miss"
    assert compare(recorded[0], replayed[0])["equal"] is False


def test_a_retried_429_replays_with_the_same_attempt_accounting():
    store, tape, recorded = record(FakeOpenRouter(faults={("decisions", 0): 429}))
    retried = [e for e in tape.entries if len(e["attempts"]) > 1]
    assert len(retried) == 1 and [a["status"] for a in retried[0]["attempts"]] == [429, 200]
    _, replayed = replay(store, tape)
    for a, b in zip(recorded, replayed, strict=True):
        assert compare(a, b)["equal"] and a["usage"]["http_attempts"] == b["usage"]["http_attempts"]


def test_a_malformed_distribution_and_its_isolated_retry_are_replayed():
    one = {"q1": QUESTIONS["q1"]}  # its third decision call is the claim gate's choice
    store, tape, recorded = record(FakeOpenRouter(faults={("decisions", 2): "malformed"}), one)
    assert any(
        (e.get("stage") or "").endswith(":isolated") for e in tape.entries if e["request_id"]
    )
    _, replayed = replay(store, tape, one)
    assert [signature(r) for r in replayed] == [signature(r) for r in recorded]


def test_a_writer_timeout_is_error_and_replays_as_the_same_error():
    store, tape, recorded = record(FakeOpenRouter(faults={("chat", 0): "timeout"}))
    errors = [r for r in recorded if r["result"]["status"] == "ERROR"]
    assert len(errors) == 1 and errors[0]["result"]["reason"] == "ambiguous_timeout"
    _, replayed = replay(store, tape)
    assert [signature(r) for r in replayed] == [signature(r) for r in recorded]


def test_writer_refusal_and_missing_evidence_replay_as_the_same_abstain():
    refuse = FakeOpenRouter(
        writer=lambda payload: {"answerable": False, "short_answer": "", "sentences": []}
    )
    store, tape, recorded = record(refuse, {"q1": QUESTIONS["q1"]})
    assert recorded[0]["result"]["status"] == "ABSTAIN"
    _, replayed = replay(store, tape, {"q1": QUESTIONS["q1"]})
    assert signature(replayed[0]) == signature(recorded[0])
    store, tape, recorded = record(FakeOpenRouter(), {"q1": QUESTIONS["q1"]}, docs=[])
    assert recorded[0]["result"]["reason"] == "no_active_evidence"
    _, replayed = replay(store, tape, {"q1": QUESTIONS["q1"]})
    assert signature(replayed[0]) == signature(recorded[0])


def test_evidence_order_is_part_of_the_request_hash_and_key_order_is_not():
    a = {"evidence": [{"id": "x", "text": "1"}, {"id": "y", "text": "2"}], "q": "?"}
    b = {"q": "?", "evidence": [{"text": "1", "id": "x"}, {"text": "2", "id": "y"}]}
    c = {"q": "?", "evidence": list(reversed(a["evidence"]))}
    assert canonical_bytes(a) == canonical_bytes(b) != canonical_bytes(c)
    headers = {"Authorization": "Bearer secret", "ai-model-id": "m"}
    assert exchange_key("u", headers, a) == exchange_key("u", {"ai-model-id": "m"}, b)
    assert exchange_key("u", headers, a) != exchange_key("u", headers, c)


def test_a_code_change_that_alters_what_the_writer_sees_is_caught_by_replay(monkeypatch):
    store, tape, _recorded = record(FakeOpenRouter(), {"q1": QUESTIONS["q1"]})
    from rvg.control import cascade

    policy = {**cascade._U, "retrieval": {**cascade._U["retrieval"], "write_top": 1}}
    monkeypatch.setattr(cascade, "_U", policy)
    _, replayed = replay(store, tape, {"q1": QUESTIONS["q1"]})
    assert replayed[0]["result"]["reason"] == "replay_miss"


def test_the_manifest_carries_every_audit_field_and_no_secret():
    _store, tape, recorded = record(FakeOpenRouter(), {"q1": QUESTIONS["q1"]})
    m = manifest(recorded[0], store_digest="d")
    assert m["config_hash"] == universal_hash() and m["knowledge"]["store_digest"] == "d"
    ranking = m["retrieval"][0]["ranking"]
    assert ranking and all(isinstance(r["score"], float) for r in ranking)
    writer = m["exchanges"]["writer"][0]
    assert writer["evidence"] and writer["raw"] and writer["finish_reason"] == "stop"
    stages = {d["stage"] for d in m["exchanges"]["decision"]}
    assert "control_instance_self_contained" in stages or stages
    assert m["decision"]["status"] == "FULL" and m["decision"]["assessment"]["accepted"]
    assert m["answer"]["citations"] == [["a@1#u0"]]
    assert "secret" not in str(tape.entries) and "test-key" not in str(tape.entries)


def test_every_exchange_hangs_under_a_traced_operation_and_the_answer_tree_is_recorded():
    _store, _tape, recorded = record(FakeOpenRouter(), {"q1": QUESTIONS["q1"]})
    nodes = recorded[0]["trace_nodes"]
    ops = {n["operation"] for n in nodes}
    assert {"answer", "retrieve", "validity_gate", "stage1", "decision"} <= ops
    assert {"generation:surface", "generation:semantic", "embedding:query"} <= ops
    by_id = {n["node_id"]: n for n in nodes}
    for entry in recorded[0]["tape"]:
        parent = by_id[entry["parent"]]
        assert parent["operation"].split(":")[0] in {"decision", "generation", "embedding"}
    writer = next(n for n in nodes if n["operation"] == "generation:surface")
    assert writer["schema_name"] == "temporal_answer_v1" and writer["parser_version"]
    assert writer["prompt_version"] == "control-writer-1" and writer["output_hash"]
    root = next(n for n in nodes if n["operation"] == "answer")
    assert root["parent_ids"] == [] and by_id[writer["parent_ids"][0]]["operation"] == "stage1"


def test_a_run_cache_hit_is_traced_and_makes_no_exchange():
    from rvg.gateway import DecisionGateway
    from rvg.testing.fake_openrouter import providers as make
    from rvg.universal.trace import CURRENT_REQUEST, CURRENT_TRACE, RequestTrace

    async def go():
        tape, trace = Tape(), RequestTrace(run_id="r", request_id="x", policy_hash="p")
        CURRENT_REQUEST.set("x")
        CURRENT_TRACE.set(trace)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(FakeOpenRouter().handler)
        ) as client:
            gateway = DecisionGateway(make(RecordingTransport(client, tape))["decision"], context())
            q = {"b": {"type": "boolean", "instructions": "Is it?"}}
            first = await gateway.ask("s", {"x": 1}, q)
            second = await gateway.ask("s", {"x": 1}, q)
        return tape, trace, first, second

    tape, trace, first, second = asyncio.run(go())
    assert first == second and len(tape.entries) == 1
    assert [n["cached"] for n in trace.nodes if n["operation"] == "decision"] == [False, True]


def test_a_frozen_clock_never_expires_so_replay_does_not_redecide_deadlines():
    ctx = CallContext.with_timeout(0.001, clock=lambda: 5.0)
    before = ctx.remaining()
    time.sleep(0.01)
    assert ctx.remaining() == before > 0


def test_evaluation_references_never_reach_a_runtime_request():
    from rvg.testing.records import runtime_question

    canary = "CANARY-GOLD-7f31"
    example = {
        "question": QUESTIONS["q1"],
        "locale": "en-US",
        "as_of": "2026-09-20T00:00:00Z",
        "answer": canary,
        "gold_ids": ["a"],
        "expected_status": "FULL",
        "dataset_id": canary,
    }
    q = runtime_question(example)
    assert canary not in q.model_dump_json()
    _store, tape, _recorded = record(FakeOpenRouter(), {"q1": q.text})
    assert canary not in json.dumps([e["body"] for e in tape.entries])


def two_sentence_writer(payload):
    """Two sentences on one unit: both pick the same verbatim control sentence, so the qualifier
    checks send byte-identical requests concurrently from two sites (the N0 R1-replay failure)."""
    unit = next(u for u in payload["evidence"] if "500" in u["text"])
    return {
        "answerable": True,
        "short_answer": "500",
        "sentences": [
            {"text": unit["text"], "evidence_ids": [unit["id"]]},
            {"text": "Project A has a budget of 500.", "evidence_ids": [unit["id"]]},
        ],
    }


def record_async(fake, questions):
    async def go():
        store, tape = store_with(), Tape()
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.async_handler)) as client:
            transport = RecordingTransport(client, tape, timeout=5, max_attempts=3)
            records = await run(store, transport, questions, tape=tape, index=True)
        return store, tape, records

    return asyncio.run(go())


def test_identical_concurrent_requests_with_different_samples_replay_exactly():
    fake = FakeOpenRouter(writer=two_sentence_writer, latency=0.003, sampling=True)
    one = {"q1": QUESTIONS["q1"]}
    store, tape, recorded = record_async(fake, one)
    assert recorded[0]["ambiguous_occurrences"], "the fixture must produce identical requests"
    assert not any(n.get("cached") for n in recorded[0]["trace_nodes"])  # live: all in flight
    for seed in (None, 1, 2):
        transport, replayed = replay(store, tape, one, jitter_seed=seed, recorded=recorded)
        c = compare(recorded[0], replayed[0])
        assert c["equal"], c
        assert transport.unused("q1") == []


def test_a_recorded_run_cache_decision_is_replayed_whatever_the_replay_timing():
    """Live: two identical asks on one gateway, the second after the first finished -> a cache
    hit; two concurrent identical asks -> both miss. The replay takes both from the record."""
    from rvg.gateway import DecisionGateway
    from rvg.testing.fake_openrouter import providers as make
    from rvg.universal.trace import CURRENT_REPLAY, CURRENT_REQUEST, CURRENT_TRACE, RequestTrace

    q = {"b": {"type": "boolean", "instructions": "Is it?"}}

    async def asks(provider, oracle=None):
        trace = RequestTrace(run_id="r", request_id="x", policy_hash="p")
        CURRENT_REQUEST.set("x")
        CURRENT_TRACE.set(trace)
        CURRENT_REPLAY.set(oracle)
        gateway = DecisionGateway(provider, context())
        first = await gateway.ask("s", {"x": 1}, q)
        second = await gateway.ask("s", {"x": 1}, q)
        concurrent = await asyncio.gather(
            gateway.ask("t", {"x": 2}, q), gateway.ask("t", {"x": 2}, q)
        )
        return trace, [first, second, *concurrent]

    async def live():
        tape = Tape()
        fake = FakeOpenRouter(latency=0.002, sampling=True)
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.async_handler)) as client:
            trace, answers = await asks(make(RecordingTransport(client, tape))["decision"])
        return tape, trace, answers

    tape, trace, answers = asyncio.run(live())
    flags = [n["cached"] for n in trace.nodes if n["operation"] == "decision"]
    assert flags == [False, True, False, False] and len(tape.entries) == 3
    oracle = ReplayOracle(trace.export())

    async def again():
        return await asks(make(ReplayTransport(tape))["decision"], oracle)

    trace2, answers2 = asyncio.run(again())
    assert [n["cached"] for n in trace2.nodes if n["operation"] == "decision"] == flags
    assert answers2 == answers
