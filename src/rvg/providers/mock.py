from __future__ import annotations

import hashlib
import json
import math
import re

from ..contracts import (
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionBatch,
    DecisionCapabilities,
    DecisionRequest,
    EmbeddingBatch,
    EmbeddingSpace,
    ScoreAnswer,
    Usage,
)
from ..errors import ContractError
from ..runtime import CallContext
from .validation import validate_request


def extractive_draft(payload):
    """The mock writer's draft: the first evidence unit that states a number, cited, its first
    number as the short answer; not answerable when no unit has one. Deterministic, no model."""
    for unit in payload["evidence"]:
        found = re.search(r"\d+", unit["text"])
        if found:
            return {
                "answerable": True,
                "short_answer": found.group(),
                "sentences": [{"text": unit["text"], "evidence_ids": [unit["id"]]}],
            }
    return {"answerable": False, "short_answer": "", "sentences": []}


def structured_answer(schema_name, payload, writer=extractive_draft):
    """What the mock writer and extractor answer for the product path's structured outputs (the
    same rules as the HTTP-level fake `rvg.testing.fake_openrouter`), or None for another
    schema."""
    if schema_name == "temporal_answer_v1":
        return writer(payload)
    if schema_name == "control_slots_v1":
        return {"slots": [{"id": "s0", "ask": payload["question"][:400]}]}
    if schema_name == "control_rewrites_v1":
        return {"rewrites": []}
    if schema_name == "gap_queries_v1":
        return {"queries": []}
    return None


def structured_text(schema_name, payload):
    answer = structured_answer(schema_name, payload)
    return None if answer is None else json.dumps(answer)


class MockDecisionProvider:
    """Mechanical wiring double; never an inference-quality benchmark."""

    provider = "mock"
    model = "mock-heuristic-not-jev"
    mode = "mock"

    def capabilities(self):
        return DecisionCapabilities(provider_confidence=True)

    async def evaluate(self, request: DecisionRequest, context: CallContext) -> DecisionBatch:
        validate_request(request)
        context.begin_call("decision")
        state = request.state
        answers = {}
        for qid, q in request.questions.items():
            if isinstance(q, BooleanQuestion):
                if isinstance(state, dict) and "candidates" in state:
                    candidate = state["candidates"][int(qid.rsplit("_", 1)[1])]
                    if qid.startswith("relevant"):
                        terms = lambda value: set(re.findall(r"[a-z0-9]+|[가-힣]+", value.lower()))
                        p = (
                            0.96
                            if terms(state["query"])
                            & terms(
                                candidate.get("quote", candidate.get("text", ""))
                                + candidate.get("object", "")
                            )
                            else 0.08
                        )
                    else:
                        p = (
                            0.97
                            if candidate["subject"] in candidate["quote"]
                            and candidate["object"] in candidate["quote"]
                            else 0.1
                        )
                else:
                    p = 0.02 if qid == "shipped" else 0.97
                answers[qid] = BooleanAnswer(probability_true=p)
            elif isinstance(q, ChoiceQuestion):
                options = list(q.criteria)
                chosen = {"intent": "explain", "style": "separate", "action": "refund"}.get(qid)
                if chosen not in options:
                    chosen = next((o for o in options if o.startswith("c")), options[0])
                answers[qid] = ChoiceAnswer(
                    choice=chosen,
                    probabilities={
                        o: 0.96 if o == chosen else 0.04 / (len(options) - 1) for o in options
                    },
                    provider_confidence=0.9,
                )
            else:
                answers[qid] = ScoreAnswer(
                    score=0,
                    probabilities={str(i): float(i == 0) for i in range(len(q.criteria))},
                    provider_confidence=0.9,
                )
        return DecisionBatch(
            answers, self.provider, self.model, response_model=self.model, usage=Usage(0, 0, 0)
        )


class MockEmbeddingProvider:
    mode = "mock"

    def __init__(self, space: EmbeddingSpace):
        self.space = space

    def describe(self):
        return self.space

    async def embed(self, texts, *, purpose, context: CallContext):
        if (
            purpose not in {"query", "document"}
            or not texts
            or any(not isinstance(t, str) or not t.strip() for t in texts)
        ):
            raise ContractError("invalid_mock_embedding_input")
        context.begin_call("embedding")
        vectors = []
        prefix = self.space.query_prefix if purpose == "query" else self.space.document_prefix
        for text in texts:
            raw = hashlib.shake_256((prefix + text).encode()).digest(self.space.dimensions)
            values = [(b - 127.5) / 127.5 for b in raw]
            norm = math.hypot(*values)
            vectors.append(tuple(v / norm if self.space.normalize else v for v in values))
        return EmbeddingBatch(
            tuple(vectors), self.space, ("mock-not-embedding-model",), (Usage(0, 0, 0),), "mock"
        )
