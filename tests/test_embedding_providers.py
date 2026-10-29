import asyncio
import json
import math
from dataclasses import replace

import httpx
import pytest

from rvg.config import Settings
from rvg.contracts import BooleanQuestion, DecisionRequest
from rvg.errors import ConfigurationError, ContractError
from rvg.factory import decision_provider, embedding_provider
from rvg.runtime import CallContext
from rvg.transport import JsonTransport


def settings(**kwargs):
    return Settings(
        _env_file=None,
        deepjev_mode="live",
        openrouter_api_key="test-key",
        embedding_dimensions=128,
        **kwargs,
    )


def row(index=0, vector=None):
    return {"index": index, "embedding": vector if vector is not None else [1.0] + [0.0] * 127}


def embed(handler, *, config=None, texts=None, purpose="query"):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = embedding_provider(config or settings(), JsonTransport(client))
            context = CallContext.with_timeout()
            result = await provider.embed(texts or ["질문"], purpose=purpose, context=context)
            return result, context

    return asyncio.run(run())


def test_wire_prefix_dimension_alias_and_normalization():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gemini-embedding-2",
                "data": [row(vector=[2.0] * 128)],
                "usage": {"prompt_tokens": 8},
            },
        )

    batch, ctx = embed(handler)
    assert calls == [
        {
            "model": "google/gemini-embedding-2",
            "input": "task: search result | query: 질문",
            "dimensions": 128,
            "encoding_format": "float",
        }
    ]
    assert math.isclose(math.hypot(*batch.vectors[0]), 1)
    assert batch.response_models == ("gemini-embedding-2",)
    assert batch.usage[0].cost_usd is None and ctx.embedding_calls == 1
    embed(handler, purpose="document")
    assert calls[-1]["input"] == "title: none | text: 질문"
    embed(handler, config=settings(embedding_upstream_only="google-ai-studio, "))
    assert calls[-1]["provider"] == {"only": ["google-ai-studio"]}


def test_batch_reorders_response_and_separates_requests():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        data = (
            [row(1, [0.0, 1.0] + [0.0] * 126), row(0)]
            if isinstance(body["input"], list)
            else [row()]
        )
        return httpx.Response(200, json={"model": "gemini-embedding-2", "data": data})

    batch, ctx = embed(handler, config=settings(embedding_batch_size=2), texts=["a", "b", "c"])
    assert len(calls) == 2 and ctx.embedding_calls == 2
    assert batch.vectors[0][0] == 1 and batch.vectors[1][1] == 1
    assert len(batch.vectors) == 3


@pytest.mark.parametrize(
    "data,model",
    [
        ([], "gemini-embedding-2"),
        ([row(1)], "gemini-embedding-2"),
        ([row(True)], "gemini-embedding-2"),
        ([row(vector=[1.0])], "gemini-embedding-2"),
        ([row(vector=[0.0] * 128)], "gemini-embedding-2"),
        ([row(vector=[True] * 128)], "gemini-embedding-2"),
        ([row()], "gemini-embedding-001"),
        ([row()], "other/gemini-embedding-2"),
    ],
)
def test_bad_embeddings(data, model):
    with pytest.raises(ContractError):
        embed(lambda _: httpx.Response(200, json={"model": model, "data": data}))


def test_nonfinite_vector_and_aggregated_batch():
    for invalid in [float("nan"), float("inf")]:
        payload = {"model": "gemini-embedding-2", "data": [row(vector=[invalid] * 128)]}
        with pytest.raises(ContractError):
            embed(lambda _, data=payload: httpx.Response(200, content=json.dumps(data)))
    with pytest.raises(ContractError, match="count_mismatch"):
        embed(
            lambda _: httpx.Response(200, json={"model": "gemini-embedding-2", "data": [row()]}),
            config=settings(embedding_batch_size=2),
            texts=["a", "b"],
        )
    with pytest.raises(ContractError, match="index_mismatch"):
        embed(
            lambda _: httpx.Response(
                200, json={"model": "gemini-embedding-2", "data": [row(), row()]}
            ),
            config=settings(embedding_batch_size=2),
            texts=["a", "b"],
        )


def test_all_input_validation_precedes_network():
    for texts in [["ok", ""], ["한" * 3000]]:
        with pytest.raises(ContractError):
            embed(lambda _: pytest.fail("network called"), texts=texts)


def test_space_changes_for_model_dimensions_prefix_and_version():
    async def run():
        async with httpx.AsyncClient() as client:
            space = embedding_provider(settings(), JsonTransport(client)).describe()
            for change in [
                {"model": "different"},
                {"dimensions": 768},
                {"query_prefix": "different"},
                {"version": 3},
                {"normalize": False},
            ]:
                assert replace(space, **change).fingerprint != space.fingerprint

    asyncio.run(run())


def test_mock_needs_no_keys_or_network():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: pytest.fail("network called"))
        ) as client:
            config = Settings(_env_file=None)
            transport = JsonTransport(client)
            ctx = CallContext.with_timeout()
            decision = await decision_provider(config, transport).evaluate(
                DecisionRequest(
                    state="test", questions={"q": BooleanQuestion(instructions="test?")}
                ),
                ctx,
            )
            embedding = await embedding_provider(config, transport).embed(
                ["test"], purpose="query", context=ctx
            )
            assert decision.provider == "mock" and embedding.space.provider == "mock"
            assert ctx.http_attempts == 0

    asyncio.run(run())


def test_factory_requires_only_selected_key():
    config = settings()
    assert embedding_provider(config, None).mode == "live"
    with pytest.raises(ConfigurationError, match="AI_GATEWAY_API_KEY"):
        decision_provider(settings(jev_provider="vercel_gateway"), None)
    with pytest.raises(ConfigurationError, match="input_type"):
        embedding_provider(settings(embedding_query_input_type="search_query"), None)
