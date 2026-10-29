import asyncio
import json
import time

import httpx
import pytest

from rvg.config import load_settings
from rvg.errors import BudgetExceeded, ConfigurationError, ContractError, ProviderError
from rvg.runtime import CallContext
from rvg.transport import JsonTransport


def test_env_precedence_and_prefix_secret_safety(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "JEV_MODEL=file-model\nAI_GATEWAY_API_KEY=secret-value\n"
        'EMBEDDING_QUERY_PREFIX="task: search result | query: "\n'
        "RUN_MAX_COST_USD=\n"
    )
    monkeypatch.setenv("JEV_MODEL", "process-model")
    settings = load_settings(tmp_path)
    assert settings.jev_model == "process-model"
    assert settings.embedding_query_prefix.endswith(": ")
    assert settings.run_max_cost_usd is None
    assert "secret-value" not in repr(settings)
    assert "secret-value" not in settings.model_dump_json()
    assert load_settings(tmp_path, jev_model="cli-model").jev_model == "cli-model"
    assert settings.resolve_path(settings.deepjev_db_path) == tmp_path / "data/deepjev.sqlite3"


@pytest.mark.parametrize(
    "setting",
    [
        "JEV_PROVIDER=nope",
        "HTTP_TIMEOUT_SECONDS=nan",
        "EMBEDDING_DIMENSIONS=-1",
        "OPENROUTER_BASE_URL=https://user:secret@host/v1",
    ],
)
def test_bad_settings_redacted(tmp_path, setting):
    (tmp_path / ".env").write_text(setting)
    with pytest.raises(ConfigurationError) as exc:
        load_settings(tmp_path)
    assert "secret" not in str(exc.value)
    assert setting not in str(exc.value)


def run_transport(handler, *, context=None, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await JsonTransport(client, **kwargs).post(
                "https://provider.test/endpoint",
                headers={"Authorization": "Bearer secret"},
                body={"input": "private-input"},
                context=context or CallContext.with_timeout(),
            )

    return asyncio.run(run())


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 302])
def test_permanent_error_and_redirect_not_retried(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status, text="secret private-input", headers={"Location": "https://elsewhere.test"}
        )

    with pytest.raises(ProviderError) as exc:
        run_transport(handler)
    assert exc.value.status == status
    assert "secret" not in str(exc.value)
    assert len(calls) == 1


def test_retry_accounting(monkeypatch):
    waits = []

    async def sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr("rvg.transport.asyncio.sleep", sleep)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"ok": True})

    context = CallContext.with_timeout()
    assert run_transport(handler, context=context) == {"ok": True}
    assert waits == [2]
    assert context.http_attempts == 2
    assert [a["http_status"] for a in context.attempts] == [429, 200]


def test_ambiguous_timeout_never_retried():
    context = CallContext.with_timeout()

    def handler(request):
        raise httpx.ReadTimeout("secret", request=request)

    with pytest.raises(ProviderError, match="ambiguous_timeout"):
        run_transport(handler, context=context)
    assert context.http_attempts == 1


def test_deadline_attempt_limit_retry_wait():
    for context in [
        CallContext(deadline=time.monotonic() - 1),
        CallContext.with_timeout(max_http_attempts=0),
    ]:
        with pytest.raises(BudgetExceeded):
            run_transport(lambda _: pytest.fail("network called"), context=context)
    with pytest.raises(BudgetExceeded, match="retry_wait"):
        run_transport(lambda _: httpx.Response(429, headers={"Retry-After": "100"}))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="bad"),
        httpx.Response(200, json=[]),
        httpx.Response(200, content=b"x" * 101),
    ],
)
def test_invalid_response(response):
    with pytest.raises(ContractError):
        run_transport(lambda _: response, max_response_bytes=100)


def test_http_deadline_covers_stream():
    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(0.1)
            yield json.dumps({"ok": True}).encode()

    with pytest.raises(ProviderError, match="ambiguous_timeout"):
        run_transport(lambda _: httpx.Response(200, stream=SlowStream()), timeout=0.01)
