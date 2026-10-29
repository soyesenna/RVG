from __future__ import annotations

import re
from abc import ABC, abstractmethod

from pydantic import SecretStr

from ..contracts import DecisionBatch, DecisionCapabilities, DecisionRequest
from ..errors import ContractError
from ..runtime import CallContext
from ..transport import JsonTransport
from .validation import parse_answers, parse_usage, validate_request


class HttpDecisionProvider(ABC):
    """Reserve one logical decision; routing can use multiple HTTP attempts."""

    mode = "live"

    async def evaluate(self, request: DecisionRequest, context: CallContext) -> DecisionBatch:
        validate_request(request)
        context.begin_call("decision")
        return await self._evaluate(request, context)

    @abstractmethod
    async def _evaluate(self, request: DecisionRequest, context: CallContext) -> DecisionBatch:
        """Execute an already reserved logical decision."""
        ...


class VercelDecisionProvider(HttpDecisionProvider):
    provider = "vercel_gateway"
    mode = "live"

    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr,
        base_url: str,
        transport: JsonTransport,
        fail_fast_on_429: bool = False,
    ):
        self.model, self._key = model, api_key
        self.base_url, self.transport = base_url, transport
        self.fail_fast_on_429 = fail_fast_on_429

    def capabilities(self) -> DecisionCapabilities:
        return DecisionCapabilities()

    async def _evaluate(self, request: DecisionRequest, context: CallContext) -> DecisionBatch:
        payload = await self.transport.post(
            self.base_url + "/evaluation-model",
            headers={
                "Authorization": "Bearer " + self._key.get_secret_value(),
                "ai-gateway-protocol-version": "0.0.1",
                "ai-gateway-auth-method": "api-key",
                "ai-evaluation-model-specification-version": "4",
                "ai-model-id": self.model,
            },
            body=request.model_dump(mode="json"),
            context=context,
            provider=self.provider,
            model=self.model,
            no_retry_statuses=frozenset({429}) if self.fail_fast_on_429 else frozenset(),
        )
        usage = parse_usage(payload, camel_case=True)
        context.record_cost(usage.cost_usd)
        answers = parse_answers(payload, request)
        response_model = payload.get("model")
        # Upstream warning text may echo private state. Only record presence.
        warnings = ("provider_warnings_present",) if payload.get("warnings") else ()
        return DecisionBatch(
            answers,
            self.provider,
            self.model,
            response_model if isinstance(response_model, str) else None,
            usage=usage,
            warnings=warnings,
            probability_decimals=payload.get("rounding", {}).get("probabilityDecimals"),
            score_decimals=payload.get("rounding", {}).get("scoreDecimals"),
        )


class TypeSafeDecisionProvider(HttpDecisionProvider):
    provider = "typesafe_direct"
    mode = "live"

    def __init__(self, *, model: str, api_key: SecretStr, base_url: str, transport: JsonTransport):
        self.model, self._key = model, api_key
        self.base_url, self.transport = base_url, transport

    def capabilities(self) -> DecisionCapabilities:
        return DecisionCapabilities(provider_confidence=True)

    async def _evaluate(self, request: DecisionRequest, context: CallContext) -> DecisionBatch:
        questions = request.model_dump(mode="json")["questions"]
        for q in questions.values():
            if q["type"] == "boolean":
                q["type"] = "noul"
        payload = await self.transport.post(
            self.base_url + "/systemone",
            headers={"Authorization": "Bearer " + self._key.get_secret_value()},
            body={"model": self.model, "state": request.state, "questions": questions},
            context=context,
            provider=self.provider,
            model=self.model,
        )
        usage = parse_usage(payload)
        context.record_cost(usage.cost_usd)
        answers = parse_answers(payload, request, native=True)
        response_model = payload.get("model")
        # TypeSafe documents a versioned response model; preserve but don't infer one.
        version = (
            response_model
            if isinstance(response_model, str)
            and response_model.startswith("jev-")
            and response_model[4:5].isdigit()
            else None
        )
        return DecisionBatch(
            answers,
            self.provider,
            self.model,
            response_model if isinstance(response_model, str) else None,
            resolved_model=version,
            usage=usage,
        )


class OpenRouterDecisionProvider(HttpDecisionProvider):
    """OpenRouter Decisions alpha API; not the /api/v1 chat endpoint."""

    provider = "openrouter"

    def __init__(self, *, model: str, api_key: SecretStr, endpoint: str, transport: JsonTransport):
        self.model, self._key = model, api_key
        self.endpoint, self.transport = endpoint, transport

    def capabilities(self) -> DecisionCapabilities:
        return DecisionCapabilities(provider_confidence=True)

    async def _evaluate(self, request: DecisionRequest, context: CallContext) -> DecisionBatch:
        questions = request.model_dump(mode="json")["questions"]
        for question in questions.values():
            if question["type"] == "boolean":
                question["type"] = "noul"
        payload = await self.transport.post(
            self.endpoint,
            headers={"Authorization": "Bearer " + self._key.get_secret_value()},
            body={"model": self.model, "state": request.state, "questions": questions},
            context=context,
            provider=self.provider,
            model=self.model,
        )
        usage = parse_usage(payload)
        context.record_cost(usage.cost_usd)
        response_model = payload.get("model")
        # The documented response uses a date-versioned model ID. Match only the
        # requested family/version; a different Jev release must not pass silently.
        versioned = isinstance(response_model, str) and re.fullmatch(
            re.escape(self.model) + r"-\d{8}", response_model
        )
        if response_model != self.model and not versioned:
            raise ContractError("decision_model_mismatch")
        answers = parse_answers(payload, request, native=True)
        return DecisionBatch(
            answers,
            self.provider,
            self.model,
            response_model,
            resolved_model=response_model if versioned else None,
            usage=usage,
            probability_decimals=payload.get("rounding", {}).get("probabilityDecimals"),
            score_decimals=payload.get("rounding", {}).get("scoreDecimals"),
        )
