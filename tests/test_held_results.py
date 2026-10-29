"""0.1.0 (parent docs/PUB_2_REPORT.md): a result that releases nothing (ABSTAIN,
UNRESOLVED_SCOPE, ERROR) shows no value of the draft the controller held — no short answer, no
claim sentence, no paired draft — on every path: the controller's holds (claims not supported,
the question not answered, the judge failing), the release check's holds and a stage failure.
The draft stays in the opt-in handoff packet; an evaluation reads it from the record's
`held_draft`, beside the result."""

import asyncio

import httpx
from test_decision_cache import LowAnswered
from test_handoff_packet import (
    QUESTION,
    Doubting,
    Fixed,
    ask,
    canonical,
    knowledge,
    once,
    without_packet,
)

from rvg.runtime import CallContext
from rvg.temporal.models import TemporalQuestion
from rvg.temporal.store import TemporalRetriever
from rvg.testing.fake_openrouter import FakeOpenRouter, providers
from rvg.testing.records import answer_one
from rvg.transport import JsonTransport
from rvg.universal.engine import UniversalEngine, withhold_drafts

DRAFT = ('"500"', "project P is 500")  # the draft's short answer and its claim sentence


def Malformed():
    return FakeOpenRouter(faults={("decisions", i): "malformed" for i in range(60)})


def shows_no_draft(result):
    public = canonical(without_packet(result))
    return result.get("short_answer") is None and not any(d in public for d in DRAFT)


def test_every_controller_hold_shows_no_draft_value():
    for make in (Doubting, LowAnswered, Malformed):
        result, _ = ask(knowledge(), make=make)
        assert result["status"] not in ("FULL", "PARTIAL"), make
        assert shows_no_draft(result), make
        for key in ("assessment", "assessment_before_recovery"):
            assert not (result.get(key) or {}).get("claims"), (make, key)


def test_the_packet_still_holds_the_draft():
    result, _ = ask(knowledge(), make=Doubting, packet=True)
    assert result["status"] == "ABSTAIN" and shows_no_draft(result)
    assert any(d in canonical(result["handoff"]) for d in DRAFT)


def test_holds_of_the_release_check_show_no_draft_value():
    store = knowledge()
    result, _ = ask(store, writer=once(lambda: store.revoke("t", "p1")))
    assert result["reason"] == "revoked_before_release" and shows_no_draft(result)


def test_a_released_answer_keeps_its_short_answer():
    result, _ = ask(knowledge())
    assert result["status"] == "FULL" and result["short_answer"] == "500"


def test_a_paired_draft_and_any_text_are_withheld_too():
    held = {
        "status": "ABSTAIN",
        "short_answer": "7",
        "text": "x",
        "sentences": [{"text": "x", "evidence_ids": ["a@1#u0"]}],
        "assessment": {"claims": ["c"], "labels": ["UNKNOWN"]},
        "trace": {"consistency": {"partner_short_answer": "8", "agreed": False}},
    }
    out = withhold_drafts(held)
    assert (out["short_answer"], out["text"], out["sentences"]) == (None, "", [])
    assert out["assessment"] == {"claims": [], "labels": ["UNKNOWN"]}
    assert out["trace"]["consistency"] == {"partner_short_answer": None, "agreed": False}
    assert held["short_answer"] == "7"  # the input is not changed


def test_an_evaluation_record_keeps_the_held_draft_beside_the_result():
    async def go():
        fake = Doubting()
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(JsonTransport(client, timeout=5, max_attempts=2))
            engine = UniversalEngine(
                retriever=TemporalRetriever(knowledge(), Fixed()),
                decision=p["decision"],
                writer=p["writer"],
                extractor=p["extractor"],
            )
            q = TemporalQuestion(
                text=QUESTION, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
            )
            ctx = CallContext.with_timeout(
                60, max_decision_calls=400, max_generation_calls=24, max_http_attempts=800
            )
            return await answer_one(engine, "t", q, request_id="r1", context=ctx)

    record = asyncio.run(go())
    assert record["result"]["status"] == "ABSTAIN" and record["result"]["short_answer"] is None
    assert record["held_draft"]["short_answer"] == "500"
    assert record["held_draft"]["assessment.claims"]


def test_an_older_recording_reads_through_the_contract_and_its_draft_is_compared():
    from rvg.testing.records import compare, contract_view

    held = {
        "status": "ABSTAIN",
        "short_answer": "500",
        "assessment": {"claims": ["c"], "labels": ["UNKNOWN"]},
    }
    old = {"result": held, "trace_nodes": []}
    new = {"result": withhold_drafts(held), "trace_nodes": []}
    view, withheld = contract_view(old)
    assert compare(view, new)["equal"] and not compare(old, new)["equal"]
    assert withheld == {"short_answer": "500", "assessment.claims": ["c"]}
    released = {"result": {"status": "FULL", "short_answer": "500"}, "trace_nodes": []}
    assert contract_view(released) == (released, None)
