from __future__ import annotations

import math
from typing import Literal

from pydantic import SecretStr

from ..contracts import EmbeddingBatch, EmbeddingSpace
from ..errors import ContractError
from ..runtime import CallContext
from ..transport import JsonTransport
from ..universal.trace import node
from .validation import parse_usage

MODEL_ALIASES = {
    "google/gemini-embedding-2": {"google/gemini-embedding-2", "gemini-embedding-2"},
}


class OpenRouterEmbeddingProvider:
    mode = "live"

    def __init__(
        self,
        *,
        space: EmbeddingSpace,
        api_key: SecretStr,
        transport: JsonTransport,
        batch_size: int = 1,
        max_input_tokens: int = 8192,
        upstream_only: tuple[str, ...] = (),
    ):
        self.space, self._key, self.transport = space, api_key, transport
        self.batch_size, self.max_input_tokens = batch_size, max_input_tokens
        self.upstream_only = upstream_only

    def describe(self) -> EmbeddingSpace:
        return self.space

    async def embed(
        self, texts: list[str], *, purpose: Literal["query", "document"], context: CallContext
    ) -> EmbeddingBatch:
        if purpose not in {"query", "document"}:
            raise ContractError("unsupported_embedding_purpose")
        if not isinstance(texts, list) or not texts:
            raise ContractError("empty_embedding_batch")
        prefix = self.space.query_prefix if purpose == "query" else self.space.document_prefix
        formatted = []
        # Validate every item before spending on the first batch. This is a deliberately
        # conservative byte precheck, NOT an exact model-token count. No truncation.
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ContractError("empty_embedding_input")
            value = prefix + text
            if len(value.encode("utf-8")) > self.max_input_tokens:
                raise ContractError("embedding_input_exceeds_conservative_byte_cap")
            formatted.append(value)
        input_type = (
            self.space.query_input_type if purpose == "query" else self.space.document_input_type
        )
        vectors, usages, response_models = [], [], []
        for start in range(0, len(formatted), self.batch_size):
            chunk = formatted[start : start + self.batch_size]
            body = {
                "model": self.space.model,
                "input": chunk[0] if len(chunk) == 1 else chunk,
                "dimensions": self.space.dimensions,
                "encoding_format": self.space.encoding_format,
            }
            if input_type is not None:
                body["input_type"] = input_type
            if self.upstream_only:  # same model, one upstream: a drained route cannot answer 429
                body["provider"] = {"only": list(self.upstream_only)}
            context.begin_call("embedding")
            # Observation only: one trace node per embedding request.
            with node(
                "embedding:" + purpose,
                {"space": self.space.fingerprint, "input": body["input"]},
                provider_model=self.space.model,
            ):
                payload = await self.transport.post(
                    self.space.base_url + "/embeddings",
                    headers={"Authorization": "Bearer " + self._key.get_secret_value()},
                    body=body,
                    context=context,
                    provider=self.space.provider,
                    model=self.space.model,
                )
            usage = parse_usage(payload, embeddings=True)
            context.record_cost(usage.cost_usd)
            model = payload.get("model")
            if not isinstance(model, str) or model not in MODEL_ALIASES.get(
                self.space.model, {self.space.model}
            ):
                raise ContractError("embedding_model_mismatch")
            rows = payload.get("data")
            if not isinstance(rows, list) or len(rows) != len(chunk):
                raise ContractError("embedding_count_mismatch")
            ordered = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ContractError("invalid_embedding_row")
                index = row.get("index")
                if type(index) is not int or not 0 <= index < len(chunk) or index in ordered:
                    raise ContractError("embedding_index_mismatch")
                vector = row.get("embedding")
                if not isinstance(vector, list) or len(vector) != self.space.dimensions:
                    raise ContractError("embedding_dimension_mismatch")
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                    raise ContractError("invalid_embedding_number")
                norm = math.hypot(*vector)
                if not math.isfinite(norm) or norm == 0:
                    raise ContractError("invalid_embedding_norm")
                ordered[index] = tuple(
                    v / norm if self.space.normalize else float(v) for v in vector
                )
            vectors.extend(ordered[index] for index in range(len(chunk)))
            usages.append(usage)
            response_models.append(model)
        # No partial vector batch is published after an error in a later request.
        return EmbeddingBatch(tuple(vectors), self.space, tuple(response_models), tuple(usages))
