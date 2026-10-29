"""U30 (parent docs/UNIVERSAL_U30_PROTOCOL.md §12): `cascade.scope_top1`, an opt-in arm for the
requests where the scope is RESOLVED but the top-ranked unit's title is not the anchor (its
document left the writing pool). S2 "entity-tie-v1" writes from the top unit's title when the
anchor title occurs in that document's retrieved units; S3 "dual-draft-v1" also writes and judges a
first draft from the top unit's title, keeps it when only it is released, withholds when both are
released with different short answers, and otherwise goes on as the policy. No model call."""

import asyncio

import pytest
from test_cascade import QUESTION, Controller, Writer, retriever, run, unit

from rvg.control import cascade
from rvg.providers.generation import GenerationResult
from rvg.temporal.models import AnswerDraft
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm


def ranked(top_text="Revenue was 42. See Alpha for the group."):
    # the top unit is Beta's; Alpha holds the next four, so Alpha dominates the scope and Beta
    # leaves the pool
    return [unit(1, "Beta", top_text)] + [unit(i, "Alpha") for i in range(2, 6)]


def test_off_is_the_cascade_default():
    # `cascade.run` keeps `off` as its own default; the engine passes the policy's value
    # (dual-draft-v1 since 0.1.0.dev42, U32)
    a = run(retrieve=retriever(ranked()), writer=Writer())
    b = run(retrieve=retriever(ranked()), writer=Writer(), scope_top1="off")
    assert a == b and "scope_top1" not in a["trace"]
    assert a["scope"]["status"] == "RESOLVED" and a["scope"]["anchor"] == "Alpha"
    assert a["status"] != "FULL"  # the answer is only in B's unit, which left the pool


def test_s2_writes_from_the_top_title_when_the_anchor_title_occurs_there():
    w = Writer()
    found = run(retrieve=retriever(ranked()), writer=w, scope_top1="entity-tie-v1")
    log = found["trace"]["scope_top1"]
    assert log["anchor_in_top1"] is True and log["chosen"] == "top1"
    assert found["status"] == "FULL" and found["evidence_ids"] == ["u1"]
    # without the anchor title in the top document, the policy's pool stays
    found = run(
        retrieve=retriever(ranked("Revenue was 42.")), writer=Writer(), scope_top1="entity-tie-v1"
    )
    assert found["trace"]["scope_top1"]["chosen"] == "anchor" and found["status"] != "FULL"


def test_s3_keeps_the_top_draft_when_only_it_is_released():
    found = run(
        retrieve=retriever(ranked("Revenue was 42.")), writer=Writer(), scope_top1="dual-draft-v1"
    )
    log = found["trace"]["scope_top1"]
    assert log["chosen"] == "top1" and log["top1_status"] == "FULL"
    assert found["status"] == "FULL" and found["short_answer"] == "42" and found["stage"] == 1


class Two:
    """Answers from any evidence: 41 from Alpha's units, 42 from Beta's."""

    provider, model, mode = "fake", "fake-llm", "mock"

    async def generate_structured(self, request, context):
        ev = request.payload["evidence"]
        value = "42" if any(e["title"] == "Beta" for e in ev) else "41"
        draft = {
            "answerable": True,
            "short_answer": value,
            "sentences": [{"text": f"Revenue was {value}.", "evidence_ids": [ev[0]["id"]]}],
        }
        return GenerationResult(AnswerDraft.model_validate(draft), {})


def test_s3_withholds_when_both_are_released_with_different_answers():
    found = run(retrieve=retriever(ranked()), writer=Two(), scope_top1="dual-draft-v1")
    assert found["status"] == "ABSTAIN" and found["reason"] == "scope_top1_disagree"
    assert found["trace"]["scope_top1"]["chosen"] == "disagree"
    # the policy releases the anchor's draft
    assert run(retrieve=retriever(ranked()), writer=Two())["short_answer"] == "41"


def test_nothing_changes_when_the_top_unit_is_the_anchors():
    same = [unit(i, "Alpha", "Revenue was 42.") for i in range(1, 6)]
    for mode in ("entity-tie-v1", "dual-draft-v1"):
        a = run(retrieve=retriever(same), writer=Writer())
        b = run(retrieve=retriever(same), writer=Writer(), scope_top1=mode)
        assert a == b and "scope_top1" not in b["trace"]


def test_arms_and_the_engine():
    for mode in ("entity-tie-v1", "dual-draft-v1"):
        arm = Arm.of("S", cascade={"scope_top1": mode, "evidence_window": "neighbors-v1"})
        eng = UniversalEngine(retriever=None, decision=None, writer=None, extractor=None, arm=arm)
        assert eng.scope_top1() == mode
    # U32 (dev42): the policy's dual draft without an arm or under an arm without the key; an arm
    # may still turn it off
    assert (
        UniversalEngine(retriever=None, decision=None, writer=None, extractor=None).scope_top1()
        == "dual-draft-v1"
    )
    off = Arm.of("R", cascade={"scope_top1": "off"})
    eng = UniversalEngine(retriever=None, decision=None, writer=None, extractor=None, arm=off)
    assert eng.scope_top1() == "off"
    with pytest.raises(ValueError, match="unknown_scope_top1"):
        Arm.of("bad", cascade={"scope_top1": "top1-always"})
    with pytest.raises(ValueError, match="unknown_scope_top1"):
        asyncio.run(cascade.run(QUESTION, scope_top1="x", **_bare()))


def _bare():
    from test_cascade import Asker, CallContext, Contained

    ctx = CallContext.with_timeout(5, max_decision_calls=20, max_generation_calls=9)
    return {
        "retrieve": retriever(ranked()),
        "writer": Writer(),
        "controller": Controller(),
        "asker": Asker(Contained(0.9), ctx),
        "context": ctx,
    }
