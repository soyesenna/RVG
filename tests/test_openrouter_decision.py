import asyncio
import copy
import json

import httpx
import pytest
from pydantic import SecretStr
from test_decision_providers import REQUEST

from rvg.errors import ContractError
from rvg.providers.decision import OpenRouterDecisionProvider
from rvg.runtime import CallContext
from rvg.transport import JsonTransport

PAYLOAD = {
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {
        "yes": {"type": "noul", "noul": 0.96},
        "pick": {
            "type": "choice",
            "choice": "refund",
            "confidence": 0.75,
            "probabilities": {"refund": 0.84, "ship": 0.16},
        },
        "score": {
            "type": "score",
            "score": 0.99,
            "confidence": 0.99,
            "probabilities": {"0": 0.01, "1": 0.99},
        },
    },
    "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992},
}


def evaluate(payload):
    calls = []

    async def run():
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenRouterDecisionProvider(
                model="typesafe/jev-1.13",
                api_key=SecretStr("fake"),
                endpoint="https://openrouter.ai/api/alpha/decisions",
                transport=JsonTransport(client),
            )
            context = CallContext.with_timeout()
            return await provider.evaluate(REQUEST, context), context

    result, context = asyncio.run(run())
    return result, context, calls


def test_native_wire_and_model_usage_normalization():
    result, context, calls = evaluate(PAYLOAD)
    request = calls[0]
    assert request.url.path == "/api/alpha/decisions"
    assert request.headers["Authorization"] == "Bearer fake"
    body = json.loads(request.content)
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"] == REQUEST.state
    assert body["questions"]["yes"]["type"] == "noul"
    assert REQUEST.questions["yes"].type == "boolean"
    assert result.answers["yes"].probability_true == 0.96
    assert result.answers["pick"].provider_confidence == 0.75
    assert result.resolved_model == "typesafe/jev-1.13-20260917"
    assert result.usage.input_tokens == 476 and result.usage.cost_usd == 0.000019992
    assert context.decision_calls == 1 and context.http_attempts == 1


@pytest.mark.parametrize(
    "model", [None, "typesafe/jev-1.14", "typesafe/jev-1.13-other", "another/jev-1.13-20260917"]
)
def test_wrong_model_rejected(model):
    payload = copy.deepcopy(PAYLOAD)
    payload["model"] = model
    with pytest.raises(ContractError, match="decision_model_mismatch"):
        evaluate(payload)


def test_unknown_option_and_missing_distribution_rejected():
    for key, value in [("choice", "wrong"), ("probabilities", None)]:
        payload = copy.deepcopy(PAYLOAD)
        payload["answers"]["pick"][key] = value
        with pytest.raises(ContractError):
            evaluate(payload)


def test_unversioned_model_is_not_claimed_immutable():
    payload = copy.deepcopy(PAYLOAD)
    payload["model"] = "typesafe/jev-1.13"
    assert evaluate(payload)[0].resolved_model is None
