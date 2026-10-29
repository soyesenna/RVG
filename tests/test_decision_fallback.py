import asyncio
import copy
import json
import time

import httpx
import pytest
from test_decision_providers import PAYLOAD as VERCEL_PAYLOAD
from test_decision_providers import REQUEST
from test_openrouter_decision import PAYLOAD as OPENROUTER_PAYLOAD_113

from rvg.config import Settings
from rvg.errors import BudgetExceeded, ConfigurationError, ContractError, ProviderError
from rvg.factory import decision_provider
from rvg.gateway import DecisionGateway
from rvg.runtime import CallContext
from rvg.transport import JsonTransport

# The fallback answers with the default OpenRouter model (jev-1.14 since 0.1.0.dev11; the dated
# name is the one OpenRouter returned on 2026-10-17).
OPENROUTER_PAYLOAD = copy.deepcopy(OPENROUTER_PAYLOAD_113) | {"model": "typesafe/jev-1.14-20261014"}


def settings(**values):
    # The default provider is OpenRouter since 0.1.0.dev11; the gateway path is chosen explicitly.
    return Settings(
        _env_file=None,
        deepjev_mode="live",
        ai_gateway_api_key="vercel-secret",
        openrouter_api_key="router-secret",
        **({"jev_provider": "vercel_gateway"} | values),
    )


def exercise(handler, *, config=None, context=None, max_attempts=3):
    calls = []

    async def run():
        def wrapped(request):
            calls.append(request)
            return handler(request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(wrapped)) as client:
            ctx = context or CallContext.with_timeout()
            provider = decision_provider(
                config or settings(), JsonTransport(client, max_attempts=max_attempts)
            )
            return await provider.evaluate(REQUEST, ctx), ctx

    return asyncio.run(run()), calls


def test_first_429_immediately_switches_same_request_and_budget(monkeypatch):
    async def no_sleep(_):
        pytest.fail("Vercel 429 must immediately fall back")

    monkeypatch.setattr("rvg.transport.asyncio.sleep", no_sleep)

    def handler(request):
        if request.url.host == "ai-gateway.vercel.sh":
            assert request.headers["Authorization"] == "Bearer vercel-secret"
            return httpx.Response(429, headers={"Retry-After": "999"})
        assert request.url.path == "/api/alpha/decisions"
        assert request.headers["Authorization"] == "Bearer router-secret"
        return httpx.Response(200, json=OPENROUTER_PAYLOAD)

    (batch, ctx), calls = exercise(handler, context=CallContext.with_timeout(max_decision_calls=1))
    assert len(calls) == 2 and ctx.http_attempts == 2 and ctx.decision_calls == 1
    primary, secondary = [json.loads(c.content) for c in calls]
    assert primary["state"] == secondary["state"] == REQUEST.state
    assert set(primary["questions"]) == set(secondary["questions"])
    assert primary["questions"]["yes"]["type"] == "boolean"
    assert secondary["questions"]["yes"]["type"] == "noul"
    assert secondary["model"] == "typesafe/jev-1.14"
    assert batch.provider == "openrouter" and batch.fallback_from == "vercel_gateway"
    assert batch.fallback_reason == "http_429"
    assert ctx.fallbacks[0]["status"] == "succeeded"
    assert [a["provider"] for a in ctx.attempts] == ["vercel_gateway", "openrouter"]
    assert (
        ctx.unknown_cost_calls == 1 and ctx.reported_cost_usd == OPENROUTER_PAYLOAD["usage"]["cost"]
    )
    assert "secret" not in json.dumps(ctx.attempts + ctx.fallbacks)


def test_primary_success_skips_fallback():
    (batch, ctx), calls = exercise(lambda _: httpx.Response(200, json=VERCEL_PAYLOAD))
    assert len(calls) == 1 and batch.provider == "vercel_gateway"
    assert batch.fallback_from is None and not ctx.fallbacks


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 500, 503])
def test_other_http_errors_do_not_fallback(status):
    def handler(request):
        assert request.url.host == "ai-gateway.vercel.sh"
        return httpx.Response(status, text="private error response")

    with pytest.raises(ProviderError) as exc:
        exercise(handler, max_attempts=1)
    assert exc.value.status == status
    assert "private" not in str(exc.value)


def test_malformed_response_timeout_and_network_do_not_fallback():
    for failure in ("invalid", "timeout", "network"):
        ctx = CallContext.with_timeout()

        def handler(request, kind=failure):
            assert request.url.host == "ai-gateway.vercel.sh"
            if kind == "timeout":
                raise httpx.ReadTimeout("secret")
            if kind == "network":
                raise httpx.ConnectError("secret")
            return httpx.Response(200, json={"answers": {}})

        with pytest.raises(ProviderError):
            exercise(handler, context=ctx)
        assert ctx.http_attempts == 1 and not ctx.fallbacks


def test_fallback_error_does_not_bounce_or_publish_invalid_answers():
    for status in (401, 429):
        ctx = CallContext.with_timeout()

        def handler(request, code=status):
            return httpx.Response(429 if request.url.host == "ai-gateway.vercel.sh" else code)

        with pytest.raises(ProviderError) as exc:
            exercise(handler, context=ctx, max_attempts=1)
        assert exc.value.status == status
        assert ctx.http_attempts == 2 and ctx.decision_calls == 1
        assert ctx.fallbacks[0]["status"] == "failed"
    invalid = copy.deepcopy(OPENROUTER_PAYLOAD)
    invalid["answers"]["pick"]["choice"] = "outside"

    def handler(request):
        return (
            httpx.Response(429)
            if request.url.host == "ai-gateway.vercel.sh"
            else httpx.Response(200, json=invalid)
        )

    with pytest.raises(ContractError):
        exercise(handler)


def test_fallback_cannot_reset_deadline_attempt_or_cost_limit():
    for limit in ("deadline", "attempts", "cost"):
        ctx = CallContext.with_timeout(max_http_attempts=1 if limit == "attempts" else 24)

        def handler(request, kind=limit, budget=ctx):
            assert request.url.host == "ai-gateway.vercel.sh"
            if kind == "deadline":
                budget.deadline = time.monotonic() - 1
            if kind == "cost":
                budget.max_cost_usd = 0.01
                budget.reported_cost_usd = 0.01
            return httpx.Response(429)

        with pytest.raises(BudgetExceeded):
            exercise(handler, context=ctx)
        assert ctx.http_attempts == 1 and ctx.decision_calls == 1
        assert ctx.fallbacks[0]["status"] == "failed"


def test_can_disable_fallback_or_select_openrouter_directly():
    with pytest.raises(ProviderError) as exc:
        exercise(
            lambda _: httpx.Response(429),
            config=settings(jev_fallback_enabled=False),
            max_attempts=1,
        )
    assert exc.value.status == 429
    (batch, ctx), calls = exercise(
        lambda _: httpx.Response(200, json=OPENROUTER_PAYLOAD),
        config=settings(jev_provider="openrouter"),
    )
    assert len(calls) == 1 and batch.provider == "openrouter" and not ctx.fallbacks


def test_missing_fallback_key_detected_before_any_call():
    config = Settings(
        _env_file=None,
        deepjev_mode="live",
        jev_provider="vercel_gateway",
        ai_gateway_api_key="fake",
    )
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        decision_provider(config, None)
    config.jev_fallback_enabled = False
    assert decision_provider(config, None).provider == "vercel_gateway"


def test_fallback_cache_and_trace_preserve_actual_provider():
    async def run():
        def handler(request):
            if request.url.host == "ai-gateway.vercel.sh":
                return httpx.Response(429)
            return httpx.Response(200, json=OPENROUTER_PAYLOAD)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            ctx = CallContext.with_timeout(max_decision_calls=1)
            gateway = DecisionGateway(decision_provider(settings(), JsonTransport(client)), ctx)
            questions = {key: q.model_dump() for key, q in REQUEST.questions.items()}
            await gateway.ask("first", REQUEST.state, questions)
            await gateway.ask("cached", REQUEST.state, questions)
            assert ctx.decision_calls == 1 and ctx.http_attempts == 2
            assert len(ctx.fallbacks) == 1
            assert gateway.trace[-1]["cached"]
            assert gateway.trace[-1]["provider"] == "openrouter"
            assert gateway.trace[-1]["model_requested"] == "typesafe/jev-1.14"
            assert gateway.trace[-1]["fallback_reason"] == "http_429"

    asyncio.run(run())
