import asyncio

from rvg.contracts import BooleanAnswer, DecisionBatch, ScoreAnswer
from rvg.control import cascade
from rvg.control.scope import anchor_shares, dominant, resolve, restrict
from rvg.control.verify import Asker
from rvg.providers.generation import GenerationResult
from rvg.runtime import CallContext
from rvg.temporal.models import AnswerDraft

QUESTION = {"text": "What was its revenue?", "locale": "en-US"}


def unit(i, title, text="Revenue was 5."):
    return {"id": f"u{i}", "title": title, "text": text}


# ------------------------------------------------------------------------------ scope


def test_anchor_shares_and_dominance_follow_the_fixed_rule():
    units = [unit(1, "A"), unit(2, "A"), unit(3, "B"), unit(4, "A"), unit(5, "")]
    shares = anchor_shares(units)
    assert shares[0][0] == "A" and shares[0][1] > 0.5 and shares[1][0] == "B"
    assert dominant(shares) == "A"
    spread = anchor_shares([unit(1, "A"), unit(2, "B"), unit(3, "C"), unit(4, "D"), unit(5, "E")])
    assert dominant(spread) is None
    assert restrict(units, "A") == [units[0], units[1], units[3]]


def test_resolution_states():
    concentrated = [unit(i, "A") for i in range(5)]
    assert resolve(concentrated, 0.1)["status"] == "RESOLVED"
    spread = [unit(i, t) for i, t in enumerate("ABCDE")]
    assert resolve(spread, 0.1)["status"] == "UNRESOLVED"  # target unnamed, ranking spread
    assert resolve(spread, 0.9)["status"] == "OPEN"  # the question names its target
    assert resolve([unit(i, "") for i in range(5)], 0.1)["status"] == "OPEN"  # no anchors at all
    assert resolve(spread, None)["status"] == "OPEN"


# ---------------------------------------------------------------------------- cascade


class Contained:
    """`self_contained` at a fixed probability; the answerable grade at a fixed level."""

    provider, model, mode = "fake", "fake-jev", "mock"

    def __init__(self, probability, level=4):
        self.probability, self.level = probability, level

    async def evaluate(self, request, context):
        answers = {}
        for k, question in request.questions.items():
            if question.type == "score":
                answers[k] = ScoreAnswer(
                    score=self.level,
                    probabilities={str(i): float(i == self.level) for i in range(5)},
                )
            else:
                answers[k] = BooleanAnswer(probability_true=self.probability)
        return DecisionBatch(answers, self.provider, self.model)


class Writer:
    """Writes when a unit with the answer is present; the draft cites the first such unit.
    `partner_answer` is what it says when asked again on evidence without the answer (the
    consistency partner): None declines, a string answers that."""

    provider, model, mode = "fake", "fake-llm", "mock"

    def __init__(self, partner_answer=None):
        self.seen, self.partner_answer = [], partner_answer

    async def generate_structured(self, request, context):
        evidence = request.payload["evidence"]
        self.seen.append([e["id"] for e in evidence])
        hit = next((e for e in evidence if "42" in e["text"]), None)
        # The consistency partner is written on the first evidence in reverse order.
        partner = bool(evidence) and evidence[0]["id"] == "u3" and self.partner_answer is not None
        if hit is None and not partner:
            value = {"answerable": False, "short_answer": "", "sentences": []}
        else:
            answer = "42" if hit is not None else self.partner_answer
            cited = hit["id"] if hit is not None else evidence[0]["id"]
            value = {
                "answerable": True,
                "short_answer": answer,
                "sentences": [{"text": f"Revenue was {answer}.", "evidence_ids": [cited]}],
            }
        return GenerationResult(AnswerDraft.model_validate(value), {})


class Controller:
    def __init__(self, status="FULL"):
        self.status, self.calls = status, 0

    async def control(self, question, evidence, draft, context):
        self.calls += 1
        return {
            "status": self.status,
            "text": draft.sentences[0].text if draft.sentences else "",
            "sentences": [s.model_dump() for s in draft.sentences],
            "missing": ["revenue"] if self.status != "FULL" else [],
            "action": None,
        }


class Proposer:
    provider, model, mode = "fake", "fake-llm", "mock"

    async def generate_structured(self, request, context):
        return GenerationResult(
            cascade.Queries(queries=("ACME revenue 2023", "What was its revenue?")), {}
        )


def retriever(ranked, by_query=None):
    async def retrieve(text, k):
        if by_query and text in by_query:
            return by_query[text][:k]
        return ranked[:k]

    return retrieve


def run(**kwargs):
    ctx = CallContext.with_timeout(5, max_decision_calls=20, max_generation_calls=9)
    defaults = {
        "writer": Writer(),
        "controller": Controller(),
        "asker": Asker(Contained(0.9), ctx),
        "context": ctx,
        # the stage logic on the writing pool alone; the policy's neighbour window (N6-2,
        # 0.1.0.dev17) is covered below and in test_n6_2_arms.py
        "evidence_window": "none",
        # U8's agreement rule (first evidence, identical text); the policy's since 0.1.0.dev18
        # (U22: widened partner, same-answer judge) is covered in test_u22_consistency.py
        "consistency_partner": "first",
        "consistency_judge": "identical_text",
    }
    return asyncio.run(cascade.run(QUESTION, **(defaults | kwargs)))


def test_first_stage_full_release_stops_the_cascade():
    ranked = [unit(1, "ACME", "Revenue was 42.")] + [unit(i, "ACME") for i in range(2, 21)]
    found = run(retrieve=retriever(ranked))
    assert found["status"] == "FULL" and found["stage"] == 1
    assert found["scope"]["status"] == "RESOLVED" and found["evidence_ids"] == ["u1", "u2", "u3"]


def test_the_writing_pool_takes_the_policys_evidence_window_unless_one_is_given(monkeypatch):
    seen = []
    real = cascade.windowed

    def windowed(pool, k, window="none"):
        seen.append(window)
        return real(pool, k, "none")

    monkeypatch.setattr(cascade, "windowed", windowed)
    ranked = [unit(1, "ACME", "Revenue was 42.")] + [unit(i, "ACME") for i in range(2, 21)]
    ctx = CallContext.with_timeout(5, max_decision_calls=20, max_generation_calls=9)
    asyncio.run(
        cascade.run(
            QUESTION,
            retrieve=retriever(ranked),
            writer=Writer(),
            controller=Controller(),
            asker=Asker(Contained(0.9), ctx),
            context=ctx,
        )
    )
    assert seen == ["neighbors-v1"]
    seen.clear()
    run(retrieve=retriever(ranked))
    assert seen == ["none"]


def test_unresolved_scope_is_a_status_not_a_guess():
    spread = [unit(i, t) for i, t in enumerate("ABCDEFGH", start=1)]
    ctx = CallContext.with_timeout(5, max_decision_calls=20)
    found = run(
        retrieve=retriever(spread),
        asker=Asker(Contained(0.1), ctx),
        context=ctx,
        unresolved_status=True,
    )
    assert found["status"] == "UNRESOLVED_SCOPE" and found["stage"] == 0
    assert found["scope"]["candidates"] == ["A", "B", "C"] and found["evidence_ids"] == []
    # Under the policy (U3: the diversion lost more than it saved) the same question proceeds,
    # with the resolution recorded; and a question that names its target always proceeds.
    ctx2 = CallContext.with_timeout(5, max_decision_calls=20)
    recorded = run(retrieve=retriever(spread), asker=Asker(Contained(0.1), ctx2), context=ctx2)
    assert recorded["status"] == "ABSTAIN" and recorded["scope"]["status"] == "UNRESOLVED"
    assert run(retrieve=retriever(spread))["status"] == "ABSTAIN"


def test_widening_happens_once_and_only_when_it_adds_evidence():
    ranked = [unit(i, "") for i in range(1, 6)] + [unit(6, "", "Revenue was 42.")]
    writer, controller = Writer(), Controller()
    found = run(
        retrieve=retriever(ranked),
        writer=writer,
        controller=controller,
        step="widen",
        consistency={},
    )
    assert found["status"] == "FULL" and found["stage"] == 2
    assert writer.seen[0] == ["u1", "u2", "u3"] and "u6" in writer.seen[1]
    assert controller.calls == 1  # the first stage had no draft, so nothing was judged twice
    short = run(retrieve=retriever(ranked[:3]), step="widen")
    assert short["stage"] == 1 and short["trace"]["second"] == {"skipped": "no_new_evidence"}


def test_gap_queries_add_units_from_new_retrieval_and_skip_the_question_itself():
    ranked = [unit(i, "ACME") for i in range(1, 6)]
    extra = {"ACME revenue 2023": [unit(9, "ACME", "Revenue was 42.")]}
    writer = Writer()
    found = run(retrieve=retriever(ranked, extra), writer=writer, extractor=Proposer(), step="gap")
    assert found["status"] == "FULL" and found["stage"] == 2
    assert found["trace"]["queries"] == ["ACME revenue 2023"]  # the verbatim question is dropped
    assert writer.seen[1] == ["u1", "u2", "u3", "u9"]


def test_a_second_draft_replaces_the_first_only_when_its_status_is_strictly_better():
    ranked = [unit(1, "ACME", "Revenue was 42.")] + [unit(i, "ACME") for i in range(2, 12)]
    partial = Controller("PARTIAL")
    found = run(retrieve=retriever(ranked), controller=partial, step="rewrite")
    assert found["status"] == "PARTIAL" and found["stage"] == 1 and partial.calls == 2
    assert found["trace"]["second"]["status"] == "PARTIAL"


def test_budget_trims_the_tail_but_keeps_the_minimum():
    wide = [unit(i, "T", "x" * 4000) for i in range(10)]
    kept = cascade.budgeted(wide, byte_budget=10_000, min_units=4)
    assert [u["id"] for u in kept] == ["u0", "u1", "u2", "u3"]


def test_the_answerable_gate_skips_writing_when_the_evidence_holds_nothing():
    ranked = [unit(1, "ACME", "Revenue was 42.")] + [unit(i, "ACME") for i in range(2, 12)]
    ctx = CallContext.with_timeout(5, max_decision_calls=20, max_generation_calls=9)
    writer = Writer()
    found = run(
        retrieve=retriever(ranked),
        writer=writer,
        asker=Asker(Contained(0.9, level=1), ctx),  # "same topic, does not hold what is asked"
        context=ctx,
        answerable_gate={"threshold": 0.65, "stages": ["first", "second"]},
    )
    assert found["status"] == "ABSTAIN" and found["reason"] == "evidence_not_answerable"
    assert writer.seen == []  # nothing was written at either stage
    assert found["answerable"] == 0.25


def test_without_the_gate_the_same_evidence_is_written():
    ranked = [unit(1, "ACME", "Revenue was 42.")] + [unit(i, "ACME") for i in range(2, 12)]
    ctx = CallContext.with_timeout(5, max_decision_calls=20, max_generation_calls=9)
    found = run(retrieve=retriever(ranked), asker=Asker(Contained(0.9, level=1), ctx), context=ctx)
    assert found["status"] == "FULL"  # the policy's gate is off


def test_a_widened_release_needs_an_agreeing_partner_draft():
    ranked = [unit(i, "") for i in range(1, 6)] + [unit(6, "", "Revenue was 42.")]
    # The partner (first evidence, reversed) declines: the widened answer is not released.
    declined = Writer()
    found = run(retrieve=retriever(ranked), writer=declined, step="widen")
    assert found["status"] == "ABSTAIN" and found["stage"] == 1
    assert found["trace"]["consistency"] == {"partner_short_answer": None, "agreed": False}
    assert declined.seen[-1] == ["u3", "u2", "u1"]  # the partner saw the first evidence reversed
    # The partner answers something else: still not released.
    other = Writer(partner_answer="41")
    assert run(retrieve=retriever(ranked), writer=other, step="widen")["stage"] == 1
    # The partner agrees (same normalised short answer): released from the second stage.
    same = Writer(partner_answer=" 42 ")
    found = run(retrieve=retriever(ranked), writer=same, step="widen")
    assert found["status"] == "FULL" and found["stage"] == 2
    assert found["trace"]["consistency"]["agreed"] is True
