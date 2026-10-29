from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class BooleanQuestion(Contract):
    type: Literal["boolean"] = "boolean"
    instructions: str = Field(min_length=1)


class ChoiceQuestion(Contract):
    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] = Field(min_length=2, max_length=255)


class ScoreQuestion(Contract):
    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    criteria: list[str] = Field(min_length=2, max_length=10)


Question = BooleanQuestion | ChoiceQuestion | ScoreQuestion


class DecisionRequest(Contract):
    state: JsonValue
    questions: dict[str, Question] = Field(min_length=1)


class BooleanAnswer(Contract):
    type: Literal["boolean"] = "boolean"
    probability_true: float


class ChoiceAnswer(Contract):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    provider_confidence: float | None = None


class ScoreAnswer(Contract):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, float]
    provider_confidence: float | None = None


Answer = BooleanAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class DecisionCapabilities:
    question_types: tuple[str, ...] = ("boolean", "choice", "score")
    max_choice_options: int = 255
    provider_confidence: bool = False


@dataclass(frozen=True)
class DecisionBatch:
    answers: dict[str, Answer]
    provider: str
    requested_model: str
    response_model: str | None = None
    # A returned alias is not an immutable weights revision.
    resolved_model: str | None = None
    usage: Usage = field(default_factory=Usage)
    warnings: tuple[str, ...] = ()
    probability_decimals: int | None = None
    score_decimals: int | None = None
    fallback_from: str | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True)
class EmbeddingSpace:
    provider: str
    model: str
    base_url: str
    dimensions: int
    normalize: bool
    similarity_metric: str
    encoding_format: str
    query_prefix: str
    document_prefix: str
    query_input_type: str | None
    document_input_type: str | None
    version: int

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    space: EmbeddingSpace
    response_models: tuple[str, ...]
    usage: tuple[Usage, ...]
    input_limit_check: str = "utf8_bytes_conservative"


class DecisionProvider(Protocol):
    model: str
    provider: str
    mode: str

    def capabilities(self) -> DecisionCapabilities: ...

    async def evaluate(self, request: DecisionRequest, context: CallContext) -> DecisionBatch: ...


class EmbeddingProvider(Protocol):
    mode: str

    def describe(self) -> EmbeddingSpace: ...

    async def embed(
        self, texts: list[str], *, purpose: Literal["query", "document"], context: CallContext
    ) -> EmbeddingBatch: ...


# Imported last to keep context independent of provider data structures.
from .runtime import CallContext
