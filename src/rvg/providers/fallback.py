from dataclasses import replace

from ..contracts import DecisionBatch, DecisionCapabilities, DecisionRequest
from ..errors import ProviderError
from ..runtime import CallContext
from .decision import HttpDecisionProvider, OpenRouterDecisionProvider, VercelDecisionProvider


class RateLimitFallbackProvider(HttpDecisionProvider):
    """One logical decision; only a primary Vercel HTTP 429 changes route."""

    def __init__(self, primary: VercelDecisionProvider, fallback: OpenRouterDecisionProvider):
        self.primary, self.fallback = primary, fallback
        self.provider, self.model = primary.provider, primary.model

    def capabilities(self) -> DecisionCapabilities:
        return DecisionCapabilities()

    async def _evaluate(self, request: DecisionRequest, context: CallContext) -> DecisionBatch:
        try:
            return await self.primary._evaluate(request, context)
        except ProviderError as exc:
            if exc.status != 429:
                raise
        # No recursive routing and no counter rollback. Both requests share the
        # original deadline, cost ledger and HTTP-attempt cap reserved by evaluate().
        event = {
            "from_provider": self.primary.provider,
            "from_model": self.primary.model,
            "to_provider": self.fallback.provider,
            "to_model": self.fallback.model,
            "http_status": 429,
            "status": "started",
        }
        context.fallbacks.append(event)
        try:
            context.remaining()
            result = await self.fallback._evaluate(request, context)
        except ProviderError as exc:
            event.update(status="failed", error=exc.code, fallback_http_status=exc.status)
            raise
        event["status"] = "succeeded"
        return replace(result, fallback_from=self.primary.provider, fallback_reason="http_429")
