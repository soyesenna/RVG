"""N6-2 arms (parent docs/N6_2_TECHQA_PROTOCOL.md §1): the answer-form contract and the neighbour
window are experiment keys; with them off the writing pool and instruction are the policy's. The
neighbour window is the policy's since 0.1.0.dev17 (parent docs/N6_2_TECHQA_REPORT.md); the
answer-form contract stays an arm. No model call."""

import asyncio

import pytest

from rvg.control import cascade
from rvg.control import writer as w
from rvg.temporal.models import AnswerDraft
from rvg.universal.experiment import Arm


def unit(doc, k, start, rev=1, text="x"):
    return {
        "id": f"{doc}@{rev}#u{k}",
        "document_id": doc,
        "revision": rev,
        "start": start,
        "text": text,
    }


def test_the_arm_keys_are_accepted_and_unknown_values_refused():
    arm = Arm.of("FN", cascade={"writer_contract": "form-v1", "evidence_window": "neighbors-v1"})
    assert dict(arm.cascade)["evidence_window"] == "neighbors-v1"
    with pytest.raises(ValueError):
        Arm.of("bad", cascade={"writer_contract": "free"})
    with pytest.raises(ValueError):
        Arm.of("bad", cascade={"evidence_window": "all"})


def test_the_policy_window_is_the_neighbours_and_an_arm_can_still_turn_it_off():
    from rvg.universal import universal
    from rvg.universal.engine import UniversalEngine

    assert universal()["cascade"]["evidence_window"] == "neighbors-v1"
    engine = lambda arm=None: UniversalEngine(
        retriever=None, decision=None, writer=None, extractor=None, arm=arm
    )
    assert engine().evidence_window() == "neighbors-v1"
    assert engine(Arm.of("F", cascade={"writer_contract": "form-v1"})).evidence_window() == (
        "neighbors-v1"
    )
    assert engine(Arm.of("dev16", cascade={"evidence_window": "none"})).evidence_window() == "none"
    assert engine().writer_contract() == "W0"


def test_without_the_window_the_pool_is_the_writing_pool_alone():
    pool = [unit("a", k, 10 * k) for k in range(6)]
    assert cascade.windowed(pool, 3) == cascade.budgeted(pool[:3])
    assert cascade.windowed(pool, 3, "none") == cascade.budgeted(pool[:3])


def test_the_window_adds_gate_passed_neighbours_of_the_same_revision_in_document_order():
    a = [unit("a", k, 10 * k) for k in range(7)]
    other_rev = unit("a", 4, 40, rev=2)
    b = [unit("b", k, 10 * k) for k in range(3)]
    pool = [a[3], b[1], a[0], other_rev, a[2], a[4], b[0], b[2], a[6]]  # a[5] did not pass the gate
    got = [u["id"] for u in cascade.windowed(pool, 2, "neighbors-v1")]
    assert got == ["a@1#u2", "a@1#u3", "a@1#u4", "b@1#u0", "b@1#u1", "b@1#u2"]
    got = [u["id"] for u in cascade.windowed(pool, 3, "neighbors-v1")]
    assert "a@1#u5" not in got and "a@2#u4" not in got and got[:2] == ["a@1#u0", "a@1#u2"]


def test_neighbours_go_first_when_the_byte_budget_does_not_hold():
    big = "y" * 7000
    pool = [unit("a", k, 10 * k, text=big) for k in range(5)]
    got = [u["id"] for u in cascade.windowed(pool[1:2] + pool[3:4] + pool, 2, "neighbors-v1")]
    assert {"a@1#u1", "a@1#u3"} <= set(got) and len(got) == 2


class Recorder:
    def __init__(self):
        self.requests = []

    async def generate_structured(self, request, context):
        self.requests.append(request)
        draft = AnswerDraft(answerable=False, short_answer="", sentences=[])
        return type("R", (), {"value": draft})()


@pytest.mark.parametrize(
    "contract, base, version",
    [("W0", w.BASE, "control-writer-1"), ("form-v1", w.FORM, "control-writer-1-form")],
)
def test_the_contract_picks_the_instruction_and_marks_the_prompt_version(contract, base, version):
    writer = Recorder()
    question = {"text": "How do I fix it?", "as_of": "2026-09-20T00:00:00Z", "locale": "en-US"}
    asyncio.run(w.write_answer(writer, question, [unit("a", 0, 0)], None, contract=contract))
    request = writer.requests[0]
    assert request.system.startswith(base) and request.prompt_version == version
