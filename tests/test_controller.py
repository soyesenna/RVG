import asyncio
import re

from rvg.contracts import BooleanAnswer, ChoiceAnswer, DecisionBatch
from rvg.control.compose import Slots
from rvg.control.controller import UNIVERSAL_PROFILE, AnswerController, ControlPolicy
from rvg.control.repair import Rewrites
from rvg.control.signals import ANY, SIGNALS
from rvg.providers.generation import GenerationResult
from rvg.runtime import CallContext
from rvg.temporal.models import ExtractedClaims

QUESTION = {"text": "What are the budget and deadline?", "locale": "en-US"}
EVIDENCE = [
    {"id": "a", "title": "T", "text": "Budget is 500."},
    {"id": "b", "title": "T", "text": "Deadline is October 1."},
]


def draft(budget):
    return {
        "answerable": True,
        "short_answer": f"{budget}; October 1",
        "sentences": [
            {"text": f"Budget is {budget}.", "evidence_ids": ["a"]},
            {"text": "Deadline is October 1.", "evidence_ids": ["b"]},
        ],
    }


class Decision:
    provider, model, mode = "fake", "fake-jev", "mock"

    async def evaluate(self, request, context):
        state, answers = request.state, {}
        wrong = lambda text: "999" in text
        for qid, q in request.questions.items():
            if q.type == "choice":
                bad = wrong(state["claims"][int(qid[1:])])
                top = "CONTRADICTED" if bad else "SUPPORTED"
                answers[qid] = ChoiceAnswer(
                    choice=top, probabilities={k: 0.94 if k == top else 0.02 for k in q.criteria}
                )
            elif re.fullmatch(r"[abd]\d+_\d+", qid):  # slot relations: answers, strict, depends
                i, j = map(int, qid[1:].split("_"))
                answers[qid] = BooleanAnswer(
                    probability_true=0.95 if qid[0] in "ab" and i == j else 0.02
                )
            elif qid == "qualifiers":
                widened = "for anyone" in state["sentence"]
                answers[qid] = BooleanAnswer(probability_true=0.1 if widened else 0.95)
            elif qid == "substance":
                hollow = "stated in the notice" in state["sentence"]
                answers[qid] = BooleanAnswer(probability_true=0.1 if hollow else 0.95)
            elif qid[0] == "r":  # refutation sweep over uncited units
                answers[qid] = BooleanAnswer(probability_true=0.02)
            else:  # draft-level and sentence-level checks
                failing = qid != "coverage" and any(wrong(c) for c in state.get("claims", []))
                answers[qid] = BooleanAnswer(probability_true=0.2 if failing else 0.95)
        return DecisionBatch(answers, self.provider, self.model)


class Generator:
    provider, model, mode = "fake", "fake-llm", "mock"

    def __init__(self):
        self.calls = []

    async def generate_structured(self, request, context):
        self.calls.append(request.prompt_version)
        if request.response_type is ExtractedClaims:
            sentences = request.payload["draft"]["sentences"]
            value = {
                "sentences": [
                    {"sentence_index": i, "claims": [s["text"]]} for i, s in enumerate(sentences)
                ],
                "short_answer_claims": [s["text"] for s in sentences],
            }
        elif request.response_type is Rewrites:
            value = {
                "rewrites": [{"sentence_index": 0, "evidence_id": "a", "quote": "Budget is 500."}]
            }
        else:
            assert request.response_type is Slots
            value = {"slots": [{"id": "s0", "ask": "budget"}, {"id": "s1", "ask": "deadline"}]}
        return GenerationResult(request.response_type.model_validate(value), {})


def run(policy, value):
    generator = Generator()
    controller = AnswerController(Decision(), generator, policy)
    ctx = CallContext.with_timeout(
        5, max_decision_calls=50, max_generation_calls=9, max_semantic_proposer_calls=9
    )
    return asyncio.run(controller.control(QUESTION, EVIDENCE, value, ctx)), generator


def test_supported_draft_is_released_in_full_without_extra_work():
    result, generator = run(ControlPolicy(), draft(500))
    assert result["status"] == "FULL" and result["action"] is None
    # The adopted direct path needs no claim extraction; only the requested items are asked.
    assert generator.calls == ["control-slots-1"]
    _, extracting = run(ControlPolicy(path="extraction"), draft(500))
    assert sorted(extracting.calls) == ["control-slots-1", "temporal-extraction-2"]


def test_the_universal_profile_asks_only_the_checks_it_names():
    # The one profile every knowledge base runs under: answered .84, qualifiers .3 (absolute
    # and relative to the source sentence). Substance, citations, fits/follows and the question
    # kind are not asked at all, so a supported but hollow sentence ships under it.
    hollow = draft(500)
    hollow["sentences"][0]["text"] = "Budget is the amount stated in the notice."
    decision = Recording()
    result = controlled(decision, ControlPolicy(), hollow)
    assert result["status"] == "FULL"
    assert "qualifiers" in decision.asked and "answered" in decision.asked
    assert not {"substance", "kind", "check"} & set(decision.asked)
    assert (
        ControlPolicy().limits()
        == UNIVERSAL_PROFILE
        == {ANY: {"answered": 0.84, "qualifiers": 0.3, "qualifiers_relative": 0.3}}
    )


def test_refuted_sentence_is_rewritten_from_evidence_and_faces_the_same_gate():
    result, generator = run(ControlPolicy(), draft(999))
    assert result["status"] == "FULL" and result["action"] == "rewrite"
    assert "999" not in result["text"] and result["sentences"][0]["evidence_ids"] == ["a"]
    assert generator.calls.count("control-rewrite-1") == 1


def test_without_recovery_only_verified_sentences_ship_and_missing_items_are_named():
    result, _ = run(ControlPolicy(recover=False), draft(999))
    assert result["status"] == "PARTIAL" and result["text"] == "Deadline is October 1."
    assert result["missing"] == ["budget"]
    strict, _ = run(ControlPolicy(recover=False, partial=False), draft(999))
    assert strict["status"] == "ABSTAIN" and strict["text"] == ""


def test_foreign_citation_and_empty_evidence_are_stopped_by_code():
    foreign = draft(500)
    foreign["sentences"][0]["evidence_ids"] = ["zzz"]
    assert run(ControlPolicy(recover=False), foreign)[0]["reason"] == "citation_id_not_active"
    controller = AnswerController(Decision(), Generator())
    empty = asyncio.run(controller.control(QUESTION, [], draft(500), CallContext.with_timeout(5)))
    assert empty["status"] == "ABSTAIN" and empty["reason"] == "no_active_evidence"


def test_sentence_stating_more_than_its_evidence_is_replaced_by_the_source_text():
    widened = draft(500)
    widened["sentences"][0]["text"] = "Budget is 500 for anyone."
    result, _ = run(ControlPolicy(), widened)
    assert result["status"] == "FULL" and result["action"] == "qualifier_rewrite"
    assert result["sentences"][0]["text"] == "Budget is 500."
    dropped, _ = run(ControlPolicy(recover=False), widened)
    assert dropped["status"] == "PARTIAL" and dropped["text"] == "Deadline is October 1."


class Recording(Decision):
    def __init__(self, *, fits=0.95):
        self.asked, self.fits = [], fits

    async def evaluate(self, request, context):
        self.asked += list(request.questions)
        special = [k for k in request.questions if k in ("kind", "check")]
        rest = {k: q for k, q in request.questions.items() if k not in special}
        plain = request.model_copy(update={"questions": rest})
        batch = await super().evaluate(plain, context) if rest else None
        answers = dict(batch.answers) if batch else {}
        for qid in special:
            if qid == "kind":
                answers[qid] = ChoiceAnswer(
                    choice="comparison", probabilities={"comparison": 0.9, "other": 0.1}
                )
            else:
                asks_kind = set(request.state) == {"question", "answer"}
                answers[qid] = BooleanAnswer(probability_true=self.fits if asks_kind else 0.95)
        return DecisionBatch(answers, self.provider, self.model)


def controlled(decision, policy, value):
    controller = AnswerController(decision, Generator(), policy)
    ctx = CallContext.with_timeout(
        5, max_decision_calls=60, max_generation_calls=9, max_semantic_proposer_calls=9
    )
    return asyncio.run(controller.control(QUESTION, EVIDENCE, value, ctx))


def test_no_profile_can_be_injected_and_the_floor_is_not_part_of_it():
    import pytest

    with pytest.raises(TypeError):
        ControlPolicy(profile={"*": {"answered": 0.5}})  # per-knowledge-base maps are gone
    # The floor is not part of the profile: a refuted claim still cannot ship in full.
    blocked = controlled(Recording(), ControlPolicy(recover=False), draft(999))
    assert blocked["status"] == "PARTIAL"
    strict = controlled(Recording(), ControlPolicy(recover=False, partial=False), draft(999))
    assert strict["status"] == "ABSTAIN"


def test_measure_returns_every_signal_as_an_unlabelled_calibration_record():
    controller = AnswerController(Recording(), Generator())
    ctx = CallContext.with_timeout(
        5, max_decision_calls=60, max_generation_calls=9, max_semantic_proposer_calls=9
    )
    found = asyncio.run(controller.measure(QUESTION, EVIDENCE, draft(500), ctx))
    assert found["group"] == "comparison" and found["eligible"] is True
    assert set(found["scores"]) == set(SIGNALS) and found["scores"]["qualifiers_relative"] == 1.0
    refuted = asyncio.run(controller.measure(QUESTION, EVIDENCE, draft(999), ctx))
    assert refuted["eligible"] is False


def test_sentences_end_without_a_space_only_after_letters_outside_ascii():
    from rvg.control.verify import SENTENCE_BREAK

    korean = "중부지방은 늦을 전망이다.17일 기상청에 따르면 3.5㎜가 내렸다.제주는 맑다."
    assert SENTENCE_BREAK.split(korean) == [
        "중부지방은 늦을 전망이다.",
        "17일 기상청에 따르면 3.5㎜가 내렸다.",
        "제주는 맑다.",
    ]
    english = "It rose 3.5% in 1999.It fell at a.com later. Then it stopped."
    assert SENTENCE_BREAK.split(english) == [
        "It rose 3.5% in 1999.It fell at a.com later.",
        "Then it stopped.",
    ]


def test_a_source_field_adds_one_note_and_its_absence_changes_nothing():
    from rvg.control.verify import SOURCE_NOTE, claim_questions, sourced

    plain = claim_questions("noul", [0], filtered=True)
    assert (
        claim_questions("noul", [0], filtered=True, note=sourced([{"id": "a", "title": "t"}]))
        == plain
    )
    noted = claim_questions("noul", [0], filtered=True, note=sourced([{"id": "a", "origin": "t"}]))
    assert all(q["instructions"].endswith(SOURCE_NOTE) for q in noted.values())


def test_rewrite_leaves_a_sentence_alone_when_its_unit_is_longer_than_a_sentence_may_be():
    """Experiment W-2: a long prose unit used to be shipped whole and broke the draft contract."""
    from rvg.control.repair import SENTENCE_LIMIT, rewrite_extractive

    long_unit = {"id": "a", "title": "", "text": "Budget is 500. " + "x" * SENTENCE_LIMIT}
    draft = {
        "answerable": True,
        "short_answer": "600",
        "sentences": [{"text": "Budget is 600.", "evidence_ids": ["a"]}],
    }
    ctx = CallContext.with_timeout(5, max_generation_calls=3, max_semantic_proposer_calls=3)
    value, report = asyncio.run(
        rewrite_extractive(Generator(), QUESTION, [long_unit], draft, {0}, ctx)
    )
    assert value is None
    assert report["rejected"][0]["reason"] == "unit_longer_than_a_sentence"
