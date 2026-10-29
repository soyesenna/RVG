from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass

from .contracts import DecisionProvider, DecisionRequest
from .providers.validation import parse_answers
from .runtime import CallContext
from .universal.trace import CURRENT_REPLAY, CURRENT_STAGE, node


@dataclass(frozen=True)
class ChoicePolicy:
    kind: str = "probability_margin"
    min_probability: float = 0.60
    min_margin: float = 0.15
    min_provider_confidence: float = 0.60

    def accepts(self, answer: dict) -> bool:
        if self.kind == "provider_confidence":
            value = answer.get("provider_confidence")
            return value is not None and value >= self.min_provider_confidence
        if self.kind != "probability_margin":
            raise ValueError("unknown_choice_policy")
        probabilities = answer["probabilities"]
        selected = probabilities[answer["choice"]]
        other = max(p for option, p in probabilities.items() if option != answer["choice"])
        return selected >= self.min_probability and selected - other >= self.min_margin


class DecisionGateway:
    """Run-scoped cache; no source text or credentials in the trace."""

    def __init__(self, provider: DecisionProvider, context: CallContext):
        self.provider, self.context = provider, context
        self.cache = {}
        self.trace = []
        self._inflight = {}  # replay only: a recorded cache hit waiting for its twin

    async def ask(self, stage: str, state: dict, questions: dict) -> dict:
        self.context.remaining()
        request = DecisionRequest(state=state, questions=questions)
        raw = json.dumps(
            {
                "provider": self.provider.provider,
                "model": self.provider.model,
                "request": request.model_dump(mode="json"),
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        key = hashlib.sha256(raw).hexdigest()
        started = time.monotonic()
        # Observation only: a trace node per ask (a run-cache hit included); decisions unchanged.
        with node("decision", raw.decode(), stage=stage, gateway_key=key) as n:
            cached = key in self.cache
            oracle = CURRENT_REPLAY.get()
            recorded = oracle.cache_hit(n.site) if oracle is not None and n.site else None
            if recorded is not None:
                # Replay: whether the live run found this request in the run cache depended on
                # its timing; the replay takes that decision from the record.
                cached = recorded
                if cached and key not in self.cache:
                    await self._inflight.setdefault(key, asyncio.get_running_loop().create_future())
            n.set(cached=cached)
            if cached:
                result = self.cache[key]
            else:
                # The stage travels with the call so a recorded exchange names what asked it.
                token = CURRENT_STAGE.set(stage)
                try:
                    result = await self.provider.evaluate(request, self.context)
                finally:
                    CURRENT_STAGE.reset(token)
                # Check injected custom providers too, not just the built-in HTTP adapters.
                wire = {}
                for qid, answer in result.answers.items():
                    data = answer.model_dump()
                    if data["type"] == "boolean":
                        data["probability"] = data.pop("probability_true")
                    else:
                        data["confidence"] = data.pop("provider_confidence")
                    wire[qid] = data
                rounding = (
                    {}
                    if result.probability_decimals is None
                    else {"probabilityDecimals": result.probability_decimals}
                )
                if result.score_decimals is not None:
                    rounding["scoreDecimals"] = result.score_decimals
                parse_answers({"answers": wire, "rounding": rounding}, request)
                self.cache[key] = result
                waiting = self._inflight.pop(key, None)
                if waiting is not None and not waiting.done():
                    waiting.set_result(None)
            n.set(
                provider_model=result.response_model,
                provider_version=result.resolved_model,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                reported_cost_usd=None if cached else result.usage.cost_usd,
            )
            n.output({q: a.model_dump() for q, a in result.answers.items()})
        answers = {qid: answer.model_dump() for qid, answer in result.answers.items()}
        self.trace.append(
            {
                "stage": stage,
                "request_hash": key,
                "cached": cached,
                "provider": result.provider,
                "model_requested": result.requested_model,
                "model_response": result.response_model,
                "model_resolved": result.resolved_model,
                "fallback_from": result.fallback_from,
                "fallback_reason": result.fallback_reason,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                "answers": answers,
                "usage": asdict(result.usage),
                "warnings": result.warnings,
                "rounding": {
                    "probability_decimals": result.probability_decimals,
                    "score_decimals": result.score_decimals,
                },
            }
        )
        return answers
