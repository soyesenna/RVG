import asyncio

import pytest

from rvg.contracts import BooleanAnswer, ChoiceAnswer, DecisionBatch
from rvg.control.compose import compose
from rvg.control.verify import (
    Asker,
    PolicyGateway,
    average,
    borderline,
    claim_questions,
    claim_state,
    judge_claims,
    judge_minimal,
    label,
)
from rvg.errors import ContractError
from rvg.runtime import CallContext

QUESTION = {"text": "What is the budget?", "as_of": "2026-09-20T00:00:00+00:00", "locale": "en-US"}
EVIDENCE = [
    {"id": "a", "title": "T", "text": "Budget is 500.", "source_sha256": "x" * 64},
    {"id": "b", "title": "T", "text": "Deadline is October 1.", "source_sha256": "y" * 64},
]


class Provider:
    provider, model, mode = "fake", "fake-jev", "mock"

    def __init__(self, *, fail_batches=False, support=0.95, refute=0.02):
        self.requests, self.fail_batches = [], fail_batches
        self.support, self.refute = support, refute

    async def evaluate(self, request, context):
        self.requests.append(request)
        if self.fail_batches and len(request.questions) > 1:
            raise ContractError("probability_sum_mismatch", detail={"sum": 0.99})
        answers = {}
        for qid, q in request.questions.items():
            if q.type == "choice":
                rest = (1 - self.support) / (len(q.criteria) - 1)
                answers[qid] = ChoiceAnswer(
                    choice="SUPPORTED",
                    probabilities={
                        k: self.support if k == "SUPPORTED" else rest for k in q.criteria
                    },
                )
            else:
                answers[qid] = BooleanAnswer(
                    probability_true=self.refute if qid.startswith("r") else self.support
                )
        return DecisionBatch(answers, self.provider, self.model)


def asker(provider, **kwargs):
    return Asker(provider, CallContext.with_timeout(5, max_decision_calls=50), **kwargs)


def test_noul_label_requires_support_and_absence_of_refutation():
    assert label({"form": "noul", "s": 0.9, "r": 0.1}) == "SUPPORTED"
    assert label({"form": "noul", "s": 0.9, "r": 0.2}) == "UNCERTAIN"
    assert label({"form": "noul", "s": 0.9, "r": 0.7}) == "CONFLICTING"
    assert label({"form": "noul", "s": 0.05, "r": 0.95}) == "CONTRADICTED"
    assert label({"form": "noul", "s": 0.1, "r": 0.1}) == "UNKNOWN"
    # An uncited unit that refutes the claim blocks an otherwise supported judgement.
    assert label({"form": "noul", "s": 0.99, "r": 0.0, "sweep_r": 0.6}) == "CONFLICTING"
    assert label({"form": "choice", "error": "probability_sum_mismatch"}) == "ERROR"
    assert label({"form": "noul", "no_evidence": True, "error": "no_active_evidence"}) == "UNKNOWN"


def test_choice_label_keeps_the_fixed_margin_gate():
    ok = {
        "form": "choice",
        "probabilities": {"SUPPORTED": 0.9, "UNKNOWN": 0.1, "CONTRADICTED": 0.0},
    }
    diluted = {"form": "choice", "probabilities": {"SUPPORTED": 0.8, "CONFLICTING": 0.2}}
    assert label(ok) == "SUPPORTED" and label(diluted) == "UNCERTAIN"
    assert borderline(diluted) and not borderline(ok)


def test_borderline_average_never_takes_the_maximum():
    low, high = {"form": "noul", "s": 0.8, "r": 0.05}, {"form": "noul", "s": 0.86, "r": 0.05}
    merged = average(low, high)
    assert merged["s"] == pytest.approx(0.83) and label(merged) == "UNCERTAIN"


def test_state_scopes_remove_query_text_and_evidence_metadata():
    full = claim_state("full", QUESTION, EVIDENCE, ["c"])
    assert full["query"]["text"] and "source_sha256" in full["evidence"][0]
    lean = claim_state("lean", QUESTION, EVIDENCE, ["c"])
    # No validity metadata remains, so the state and wording stop asking about as_of at all.
    assert "query" not in lean and set(lean["evidence"][0]) == {"id", "title", "text"}
    assert "query.as_of" not in claim_questions("noul", [0], filtered=True)["s0"]["instructions"]
    assert "query.as_of" in claim_questions("choice", [0])["c0"]["instructions"]
    assert "already filtered" in claim_questions("choice", [0], filtered=True)["c0"]["instructions"]
    assert "source_sha256" in claim_state("no_query", QUESTION, EVIDENCE, ["c"])["evidence"][0]


def test_contract_error_is_split_once_per_question_and_recorded():
    provider = Provider(fail_batches=True)
    client = asker(provider)
    result = asyncio.run(
        judge_claims(client, QUESTION, EVIDENCE, ["one", "two"], form="choice", scope="full")
    )
    assert [label(result[i]) for i in (0, 1)] == ["SUPPORTED", "SUPPORTED"]
    assert all(result[i]["split_recovered"] for i in (0, 1))
    assert [len(r.questions) for r in provider.requests] == [2, 1, 1]
    assert (
        client.events[0]["recovered"] == ["c0", "c1"] and client.events[0]["detail"]["sum"] == 0.99
    )
    disabled = asyncio.run(
        judge_claims(
            asker(Provider(fail_batches=True), split_on_contract_error=False),
            QUESTION,
            EVIDENCE,
            ["one", "two"],
            form="choice",
            scope="full",
        )
    )
    assert label(disabled[0]) == "ERROR"


def test_empty_evidence_is_decided_by_code_without_a_model_call():
    provider = Provider()
    result = asyncio.run(
        judge_claims(asker(provider), QUESTION, [], ["one"], form="noul", scope="lean")
    )
    assert label(result[0]) == "UNKNOWN" and not provider.requests


def test_minimal_scope_shows_only_cited_units_and_sweeps_the_rest():
    provider = Provider(refute=0.4)
    result = asyncio.run(
        judge_minimal(
            asker(provider), QUESTION, EVIDENCE, ["Budget is 500."], [["a"]], form="choice"
        )
    )
    cited, sweep = provider.requests
    assert [e["id"] for e in cited.state["evidence"]] == ["a"]
    assert cited.state["claims"] == ["Budget is 500."]
    assert "query" not in cited.state and "draft" not in cited.state
    assert [e["id"] for e in sweep.state["evidence"]] == ["b"] and list(sweep.questions) == ["r0"]
    assert result[0]["sweep_r"] == 0.4 and label(result[0]) == "UNCERTAIN"
    uncited = asyncio.run(
        judge_minimal(asker(Provider()), QUESTION, EVIDENCE, ["x"], [["missing"]], form="noul")
    )
    assert label(uncited[0]) == "ERROR"


def test_dependency_closure_drops_a_value_whose_condition_is_unverified():
    sentences = [
        "Only approved firms qualify.",
        "Eligible firms get up to 50.",
        "Deadline is Nov 1.",
    ]
    slots = [{"id": f"s{i}", "ask": a} for i, a in enumerate(("eligibility", "cap", "deadline"))]
    relations = {
        "answers": [[0.95, 0.0, 0.0], [0.1, 0.95, 0.0], [0.0, 0.0, 0.97]],
        "depends": [[0.9, 0.0, 0.0], [0.8, 0.9, 0.0], [0.0, 0.0, 0.0]],
    }
    partial = compose(sentences, [1, 2], relations, slots, partial=True)
    assert partial["status"] == "PARTIAL" and partial["sentences"] == [2]
    assert partial["missing_slots"] == [0, 1] and partial["dropped_for_dependency"] == [1]
    assert compose(sentences, [1, 2], relations, slots, partial=False)["status"] == "ABSTAIN"
    full = compose(sentences, [0, 1, 2], relations, slots, partial=True)
    assert full["status"] == "FULL" and full["missing_slots"] == []
    # A single requested item can never be reported as a partial answer.
    single = compose(
        ["Deadline is Nov 1."],
        [],
        {"answers": [[0.97]], "depends": [[0.0]]},
        slots[2:],
        partial=True,
    )
    assert single["status"] == "ABSTAIN" and single["text"] == ""


def test_isolated_scope_gives_each_claim_its_own_state_and_minimal_groups_by_citation():
    provider = Provider()
    asyncio.run(
        judge_claims(
            asker(provider), QUESTION, EVIDENCE, ["one", "two"], form="noul", scope="isolated"
        )
    )
    assert [r.state["claims"] for r in provider.requests] == [["one"], ["two"]]
    assert all(list(r.questions) == ["s0", "r0"] for r in provider.requests)
    grouped = Provider()
    result = asyncio.run(
        judge_minimal(
            asker(grouped),
            QUESTION,
            EVIDENCE,
            ["budget", "deadline", "both"],
            [["a"], ["b"], ["a", "b"]],
            form="noul",
            only=[1],
        )
    )
    # Only the requested claim is asked, but its group state is unchanged for a fair re-query.
    assert list(result) == [1] and grouped.requests[0].state["claims"] == ["deadline"]
    assert [e["id"] for e in grouped.requests[0].state["evidence"]] == ["b"]


def test_upstream_gate_receives_the_same_contract_policy():
    provider = Provider(fail_batches=True)
    gateway = PolicyGateway(asker(provider))
    questions = {f"c{i}": {"type": "boolean", "instructions": "?"} for i in range(2)}
    answers = asyncio.run(gateway.ask("gate", {"pairs": []}, questions))
    assert set(answers) == {"c0", "c1"} and len(provider.requests) == 3
