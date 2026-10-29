"""A deterministic stand-in for the three OpenRouter endpoints the product path calls (chat
completions for the writer/extractor, the Jev decisions API, embeddings), served through
`httpx.MockTransport` so the real providers, transport and parsers run unchanged.

Answers depend only on request content. `faults` injects one HTTP-level failure per
(endpoint, call number): an int is a status code, "malformed" a probability distribution that
does not sum to one, "timeout" a transport timeout."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re

import httpx
from pydantic import SecretStr

from ..contracts import EmbeddingSpace
from ..providers.decision import OpenRouterDecisionProvider
from ..providers.embedding import OpenRouterEmbeddingProvider
from ..providers.generation import OpenRouterGenerationProvider
from ..providers.mock import extractive_draft, structured_answer

BASE = "https://openrouter.test/api/v1"
DECISIONS = "https://openrouter.test/api/alpha/decisions"
JEV, QWEN, EMBED = "typesafe/jev-1.14", "qwen/qwen3.8-27b", "google/gemini-embedding-2"
DIMS = 8
RELATIONS = {"REPLACES", "CONFLICTS", "UNRELATED", "UNKNOWN"}


def space(dims=DIMS):
    return EmbeddingSpace(
        provider="openrouter",
        model=EMBED,
        base_url=BASE,
        dimensions=dims,
        normalize=True,
        similarity_metric="cosine",
        encoding_format="float",
        query_prefix="",
        document_prefix="",
        query_input_type=None,
        document_input_type=None,
        version=1,
    )


def vector(text, dims=DIMS):
    digest = hashlib.sha256(text.encode()).digest()
    return [(b - 127.5) / 127.5 for b in digest[:dims]]


# The writer's rule is the mock writer's (`rvg.providers.mock.extractive_draft`): the first
# evidence unit that states a number, cited.
default_writer = extractive_draft


def spread(options, top, p=0.94):
    rest = [o for o in options if o != top]
    probabilities = {o: round((1 - p) / len(rest), 6) for o in rest}
    probabilities[top] = round(1 - sum(probabilities.values()), 6)
    return probabilities


class FakeOpenRouter:
    """`latency` makes every response wait like a network call (deterministic per call number),
    so concurrent requests are in flight together as in a live run; `sampling` makes boolean
    judgements differ between identical requests (a model's sampling), as a live Jev may."""

    def __init__(
        self,
        *,
        writer=default_writer,
        relation="UNRELATED",
        faults=None,
        latency=0.0,
        sampling=False,
    ):
        self.writer, self.relation = writer, relation
        self.faults = dict(faults or {})
        self.latency, self.sampling = latency, sampling
        self.counts = {}
        self.bodies = []
        self.decisions = 0

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        n = len(self.bodies)
        response = self.handler(request)
        if self.latency:
            await asyncio.sleep(self.latency * (1 + (n * 7919) % 5) / 5)
        return response

    def _fault(self, endpoint):
        n = self.counts.get(endpoint, 0)
        self.counts[endpoint] = n + 1
        return self.faults.get((endpoint, n))

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append((str(request.url), body))
        url = str(request.url)
        endpoint = (
            "embeddings"
            if url.endswith("/embeddings")
            else "chat"
            if url.endswith("/chat/completions")
            else "decisions"
        )
        fault = self._fault(endpoint)
        if fault == "timeout":
            raise httpx.ReadTimeout("fake_timeout", request=request)
        if isinstance(fault, int):
            return httpx.Response(fault, json={"error": "injected"})
        if endpoint == "embeddings":
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "data": [
                        {"index": i, "embedding": vector(t, body["dimensions"])}
                        for i, t in enumerate(texts)
                    ],
                    "usage": {"prompt_tokens": 3, "cost": 0.0},
                },
            )
        if endpoint == "chat":
            return self._chat(body)
        return self._decide(body, malformed=fault == "malformed")

    def _chat(self, body):
        schema = body["response_format"]["json_schema"]["name"]
        payload = json.loads(body["messages"][-1]["content"])
        value = structured_answer(schema, payload, writer=self.writer)
        if value is None:
            return httpx.Response(400, json={"error": "unknown_schema:" + schema})
        return httpx.Response(
            200,
            json={
                "id": "gen-fake",
                "model": body["model"],
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(value)},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001},
            },
        )

    def _decide(self, body, *, malformed=False):
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "choice":
                names = list(q["criteria"])
                p = 0.94
                if set(names) == RELATIONS:
                    # `relation` is a choice, or a callable (request state, question id) ->
                    # (choice, probability) for injected judgements.
                    top = self.relation
                    if callable(self.relation):
                        top, p = self.relation(body["state"], qid)
                else:
                    top = "SUPPORTED" if "SUPPORTED" in names else names[0]
                probabilities = spread(names, top, p)
                if malformed:
                    probabilities[top] = 0.5
                answers[qid] = {"type": "choice", "choice": top, "probabilities": probabilities}
            elif q["type"] == "score":
                top = len(q["criteria"]) - 1
                answers[qid] = {
                    "type": "score",
                    "score": top,
                    "probabilities": {str(i): float(i == top) for i in range(len(q["criteria"]))},
                }
            elif re.fullmatch(r"[abd]\d+_\d+", qid):
                i, j = map(int, qid[1:].split("_"))
                p = 0.95 if qid[0] in "ab" and i == j else 0.02
                answers[qid] = {"type": "noul", "noul": p}
            else:
                p = 0.95
                if self.sampling:  # identical requests get different samples
                    self.decisions += 1
                    p = [0.95, 0.62, 0.81, 0.4][self.decisions % 4]
                answers[qid] = {"type": "noul", "noul": p}
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": 20, "output_tokens": 1, "cost": 0.0002},
                "rounding": {"probabilityDecimals": 6},
            },
        )


def providers(transport, dims=DIMS):
    key = SecretStr("test-key-not-real")
    return {
        "decision": OpenRouterDecisionProvider(
            model=JEV, api_key=key, endpoint=DECISIONS, transport=transport
        ),
        "writer": OpenRouterGenerationProvider(
            model=QWEN, api_key=key, base_url=BASE, transport=transport
        ),
        "extractor": OpenRouterGenerationProvider(
            model=QWEN, api_key=key, base_url=BASE, transport=transport
        ),
        "embedding": OpenRouterEmbeddingProvider(
            space=space(dims), api_key=key, transport=transport
        ),
    }
