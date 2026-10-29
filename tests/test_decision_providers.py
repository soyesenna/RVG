import asyncio
import copy
import json

import httpx
import pytest
from pydantic import SecretStr

from rvg.contracts import BooleanQuestion, ChoiceQuestion, DecisionRequest, ScoreQuestion
from rvg.errors import ContractError
from rvg.providers.decision import TypeSafeDecisionProvider, VercelDecisionProvider
from rvg.runtime import CallContext
from rvg.transport import JsonTransport

REQUEST = DecisionRequest(
    state={"ticket": "refunded"},
    questions={
        "yes": BooleanQuestion(instructions="Refunded?"),
        "pick": ChoiceQuestion(
            instructions="Action?", criteria={"refund": "Refund", "ship": "Ship"}
        ),
        "score": ScoreQuestion(instructions="Quality?", criteria=["bad", "good"]),
    },
)
PAYLOAD = {
    "answers": {
        "yes": {"type": "boolean", "probability": 0.97},
        "pick": {
            "type": "choice",
            "choice": "refund",
            "probabilities": {"refund": 0.9, "ship": 0.1},
        },
        "score": {"type": "score", "score": 0.9, "probabilities": {"0": 0.1, "1": 0.9}},
    },
    "usage": {"inputTokens": 12, "outputTokens": 3},
    "providerMetadata": {"gateway": {"cost": "0.002"}},
}


def evaluate(payload, *, native=False, request=REQUEST):
    calls = []

    async def run():
        def handle(req):
            calls.append(req)
            return httpx.Response(200, json=payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            provider = (TypeSafeDecisionProvider if native else VercelDecisionProvider)(
                model="jev-1.13.0" if native else "typesafe-ai/jev",
                api_key=SecretStr("test-key"),
                base_url="https://provider.test",
                transport=JsonTransport(client),
            )
            context = CallContext.with_timeout()
            return await provider.evaluate(request, context), calls, context

    return asyncio.run(run())


def test_vercel_wire_normalization_without_confidence():
    batch, calls, context = evaluate(PAYLOAD)
    req = calls[0]
    assert req.url.path == "/evaluation-model"
    assert req.headers["ai-model-id"] == "typesafe-ai/jev"
    assert req.headers["ai-evaluation-model-specification-version"] == "4"
    body = json.loads(req.content)
    assert "model" not in body
    assert body["questions"]["yes"]["type"] == "boolean"
    assert batch.answers["yes"].probability_true == 0.97
    assert batch.answers["pick"].provider_confidence is None
    assert batch.usage.input_tokens == 12 and batch.usage.cost_usd == 0.002
    assert batch.resolved_model is None
    assert context.decision_calls == 1 and context.http_attempts == 1


def test_typesafe_uses_same_contract():
    payload = copy.deepcopy(PAYLOAD)
    payload["model"] = "jev-1.13.0"
    payload["usage"] = {"input_tokens": 12, "output_tokens": 3}
    payload["answers"]["yes"] = {"type": "noul", "noul": 0.97}
    payload["answers"]["pick"]["confidence"] = 0.8
    batch, calls, _ = evaluate(payload, native=True)
    assert json.loads(calls[0].content)["questions"]["yes"]["type"] == "noul"
    assert calls[0].url.path == "/systemone"
    assert batch.answers["pick"].provider_confidence == 0.8
    assert batch.answers["yes"].probability_true == 0.97
    assert batch.resolved_model == "jev-1.13.0"


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["answers"].pop("yes"),
        lambda p: p["answers"].update(extra={"type": "boolean", "probability": 1}),
        lambda p: p["answers"]["yes"].update(type="noul"),
        lambda p: p["answers"]["yes"].update(probability=True),
        lambda p: p["answers"]["yes"].update(probability=1.1),
        lambda p: p["answers"]["pick"].pop("probabilities"),
        lambda p: p["answers"]["pick"].update(choice="unknown"),
        lambda p: p["answers"]["pick"].update(choice="ship"),
        lambda p: p["answers"]["pick"].update(probabilities={"refund": 0.6, "ship": 0.1}),
        lambda p: p["answers"]["pick"].update(confidence=-1),
        lambda p: p["answers"]["score"].update(score=2),
        lambda p: p["usage"].update(inputTokens=-1),
    ],
)
def test_invalid_answers_fail_closed(change):
    payload = copy.deepcopy(PAYLOAD)
    change(payload)
    with pytest.raises(ContractError):
        evaluate(payload)


def test_nan_in_raw_response():
    from rvg.providers.validation import parse_answers

    payload = copy.deepcopy(PAYLOAD)
    payload["answers"]["yes"]["probability"] = float("nan")
    with pytest.raises(ContractError):
        parse_answers(payload, REQUEST)


def test_rounding_and_missing_usage():
    payload = copy.deepcopy(PAYLOAD)
    payload.pop("usage")
    payload.pop("providerMetadata")
    payload["rounding"] = {"probabilityDecimals": 2}
    payload["answers"]["pick"]["probabilities"] = {"refund": 0.90, "ship": 0.09}
    batch, _, _ = evaluate(payload)
    assert batch.usage.input_tokens is None and batch.usage.cost_usd is None
    payload["rounding"]["probabilityDecimals"] = 0
    with pytest.raises(ContractError, match="rounding_too_coarse"):
        evaluate(payload)


def test_input_cap_before_network():
    request = REQUEST.model_copy(update={"state": "x" * 24001})
    with pytest.raises(ContractError, match="local_byte_limit"):
        evaluate(PAYLOAD, request=request)


def test_many_choice_options_with_declared_rounding():
    request = DecisionRequest(
        state="test",
        questions={
            "next": ChoiceQuestion(
                instructions="Choose", criteria={str(i): str(i) for i in range(18)}
            )
        },
    )
    payload = {
        "answers": {
            "next": {
                "type": "choice",
                "choice": "0",
                "probabilities": {str(i): 0.84 if i == 0 else 0.01 for i in range(18)},
            }
        },
        "rounding": {"probabilityDecimals": 2},
    }
    batch, _, _ = evaluate(payload, request=request)
    assert batch.probability_decimals == 2
    assert batch.answers["next"].probabilities["0"] == 0.84
    payload["answers"]["next"]["probabilities"]["0"] = 0.5
    with pytest.raises(ContractError, match="sum_mismatch"):
        evaluate(payload, request=request)


def test_score_matches_distribution():
    payload = copy.deepcopy(PAYLOAD)
    payload["answers"]["score"]["score"] = 0.1
    with pytest.raises(ContractError, match="score_distribution_mismatch"):
        evaluate(payload)


def test_undeclared_rounding_of_a_score_distribution_is_bounded_not_rejected():
    levels = ScoreQuestion(instructions="Level?", criteria=["a", "b", "c", "d", "e"])
    request = DecisionRequest(state={"ticket": "refunded"}, questions={"level": levels})
    probabilities = {"0": 0.01, "1": 0.02, "2": 0.1, "3": 0.52, "4": 0.34}  # sums to 0.99
    payload = {
        "answers": {"level": {"type": "score", "score": 3.19, "probabilities": probabilities}}
    }
    batch, _, _ = evaluate(payload, request=request)  # mean of the rounded values is 3.13
    assert batch.answers["level"].score == 3.19
    payload["answers"]["level"]["score"] = 2.5
    with pytest.raises(ContractError, match="score_distribution_mismatch"):
        evaluate(payload, request=request)


def test_scalar_state_rejected_before_network():
    with pytest.raises(ContractError, match="invalid_state_type"):
        evaluate(PAYLOAD, request=REQUEST.model_copy(update={"state": True}))


def test_undeclared_two_decimal_rounding_is_bounded_not_rejected():
    from rvg.contracts import ChoiceQuestion, DecisionRequest
    from rvg.errors import ContractError
    from rvg.providers.validation import parse_answers

    criteria = {k: k for k in ("A", "B", "C", "D")}
    request = DecisionRequest(
        state={}, questions={"q": ChoiceQuestion(instructions="?", criteria=criteria)}
    )

    def payload(values):
        probs = dict(zip(criteria, values, strict=True))
        return {"answers": {"q": {"type": "choice", "choice": "A", "probabilities": probs}}}

    # 87 observed failures were all 0.99 over four options with no rounding metadata.
    answer = parse_answers(payload([0.66, 0.05, 0.0, 0.28]), request)["q"]
    assert answer.choice == "A"
    for broken in ([0.66, 0.05, 0.0, 0.2], [0.660001, 0.05, 0.0, 0.28]):
        with pytest.raises(ContractError, match="probability_sum_mismatch"):
            parse_answers(payload(broken), request)
