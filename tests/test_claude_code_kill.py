"""D-CLI-KILL (parent docs/rvi/FIX_CLI_KILL_PROTOCOL.md): a timeout or a cancellation of a Claude
Code CLI call ends its process without replacing the exception being raised — a timeout becomes
`ProviderError("ambiguous_timeout")` in the transport, a cancellation stays a cancellation — and
leaves no process behind (killed and reaped). The race of the defect (the process ends between
the timeout and the kill, so `kill()` raises `ProcessLookupError`) is made deterministic by letting
the process finish before the timeout or cancellation is raised. Offline: a fake `claude`."""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import sys

import httpx
import pytest
from test_rvi03_scoped_fence import QUESTION, SPACE, Q, World

from rvg.errors import ProviderError
from rvg.providers import claude_code
from rvg.providers.generation import ClaudeCodeGenerationProvider
from rvg.runtime import CallContext
from rvg.temporal.access import TrustedPrincipal
from rvg.temporal.models import TemporalQuestion
from rvg.temporal.retriever import TemporalRetriever
from rvg.testing.fake_openrouter import FakeOpenRouter, providers
from rvg.transport import JsonTransport
from rvg.universal.trace import CURRENT_REQUEST

FAKE = r"""#!{python}
import json, os, sys, time
state = os.environ["FAKE_STATE"]
with open(os.path.join(state, "pid"), "w") as f:
    f.write(str(os.getpid()))
mode = open(os.path.join(state, "mode")).read().strip()
if mode == "hang":
    time.sleep(60)
model = sys.argv[sys.argv.index("--model") + 1]
print(json.dumps({{
    "type": "result", "subtype": "success", "is_error": False, "api_error_status": None,
    "usage": {{"input_tokens": 1, "output_tokens": 1}}, "total_cost_usd": 0.0,
    "modelUsage": {{model: {{"inputTokens": 1, "outputTokens": 1}}}},
    "structured_output": {{"answerable": True, "short_answer": "500",
                          "sentences": [{{"text": "The budget of project P is 500.",
                                         "evidence_ids": []}}]}},
    "result": "",
}}))
"""

BODY = json.dumps(
    {"model": "claude-haiku-5-5", "system": "S", "input": "{}", "json_schema": {"type": "object"}}
).encode()


@pytest.fixture
def cli(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    binary = tmp_path / "claude"
    binary.write_text(FAKE.format(python=sys.executable))
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_STATE", str(state))

    def mode(value):
        (state / "mode").write_text(value)

    def pid():
        path = state / "pid"
        return int(path.read_text()) if path.exists() else None

    mode("ok")
    return binary, mode, pid


def gone(pid):
    """No process (running or zombie) with this pid is left: `kill(pid, 0)` fails."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


class Racing:
    """`asyncio` for `claude_code` only, whose `wait_for` lets the process finish and then raises
    `raised`: the timeout or cancellation arrives after the process ended (the defect's race)."""

    def __init__(self, raised):
        self.raised = raised

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def wait_for(self, awaitable, timeout):
        await awaitable
        raise self.raised


def transport(binary, timeout=5.0):
    return JsonTransport(None, claude_code_binary=str(binary), claude_code_timeout=timeout)


def post(t):
    async def go():
        token = CURRENT_REQUEST.set("r1")
        try:
            ctx = CallContext.with_timeout(60, max_http_attempts=10, max_generation_calls=10)
            return await t.post(claude_code.URL, headers={}, body=json.loads(BODY), context=ctx)
        finally:
            CURRENT_REQUEST.reset(token)

    return asyncio.run(go())


# --- the race of the defect ----------------------------------------------------------------------


def test_a_timeout_after_the_process_ended_is_a_provider_error(cli, monkeypatch):
    binary, _, pid = cli
    monkeypatch.setattr(claude_code, "asyncio", Racing(TimeoutError()))
    with pytest.raises(ProviderError) as err:  # before the fix: ProcessLookupError from kill()
        post(transport(binary))
    assert err.value.code == "ambiguous_timeout"
    assert gone(pid())


def test_a_cancellation_after_the_process_ended_stays_a_cancellation(cli, monkeypatch):
    binary, _, pid = cli
    monkeypatch.setattr(claude_code, "asyncio", Racing(asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):  # before the fix: ProcessLookupError
        asyncio.run(claude_code.send(BODY, binary=str(binary), timeout=5, max_bytes=10**6))
    assert gone(pid())


def test_a_kill_that_finds_the_process_gone_is_absorbed(cli, monkeypatch):
    """`returncode` is still None but the process has just ended: `kill()` raises."""
    binary, mode, pid = cli
    mode("hang")
    real_kill = asyncio.subprocess.Process.kill

    def late(self):
        real_kill(self)  # it ends now ...
        raise ProcessLookupError  # ... and the kill reports it gone

    monkeypatch.setattr(asyncio.subprocess.Process, "kill", late)
    with pytest.raises(ProviderError) as err:
        post(transport(binary, timeout=0.5))
    assert err.value.code == "ambiguous_timeout" and gone(pid())


# --- timeout and cancellation of a running process: no process left -----------------------------


def test_a_hanging_call_times_out_and_leaves_no_process(cli):
    binary, mode, pid = cli
    mode("hang")
    with pytest.raises(ProviderError) as err:
        post(transport(binary, timeout=0.5))
    assert err.value.code == "ambiguous_timeout"
    assert gone(pid())  # killed and reaped: no orphan, no zombie


def test_a_cancelled_call_cancels_and_leaves_no_process(cli):
    binary, mode, pid = cli
    mode("hang")

    async def go():
        task = asyncio.create_task(
            claude_code.send(BODY, binary=str(binary), timeout=30, max_bytes=10**6)
        )
        while pid() is None:
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pid()

    assert gone(asyncio.run(go()))


def test_an_outer_timeout_boundary_still_sees_a_timeout(cli):
    """The caller's `asyncio.timeout` cancels the call; the cancellation is not turned into
    another exception, so the boundary turns it into `TimeoutError` as asyncio intends."""
    binary, mode, pid = cli
    mode("hang")

    async def go():
        async with asyncio.timeout(0.5):
            await claude_code.send(BODY, binary=str(binary), timeout=30, max_bytes=10**6)

    with pytest.raises(TimeoutError):
        asyncio.run(go())
    assert gone(pid())


# --- the engine: an ERROR result, never an exception ---------------------------------------------


def test_the_engine_answers_error_when_the_writer_call_races_its_timeout(cli, monkeypatch):
    from rvg.universal.engine import UniversalEngine

    binary, _, _ = cli
    w = World()
    monkeypatch.setattr(claude_code, "asyncio", Racing(TimeoutError()))

    async def go():
        fake = FakeOpenRouter()
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(JsonTransport(client, timeout=5, max_attempts=2))
            writer = ClaudeCodeGenerationProvider(
                model="claude-haiku-5-5", transport=transport(binary)
            )
            engine = UniversalEngine(
                retriever=TemporalRetriever(w.storage, _Fixed(), principal=_ALICE),
                decision=p["decision"],
                writer=writer,
                extractor=p["extractor"],
            )
            q = TemporalQuestion(
                text=QUESTION, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
            )
            ctx = CallContext.with_timeout(
                60,
                max_decision_calls=400,
                max_generation_calls=24,
                max_embedding_calls=40,
                max_semantic_proposer_calls=16,
                max_surface_proposer_calls=8,
                max_http_attempts=800,
            )
            return await engine.answer("t", q, ctx)

    result = asyncio.run(go())  # before the fix: ProcessLookupError out of answer()
    assert result["status"] == "ERROR"
    codes = json.dumps(result.get("error") or {}) + str(result.get("reason"))
    assert re.search(r"ambiguous_timeout|stage_failed", codes)


_ALICE = TrustedPrincipal("alice", groups=("staff",))


class _Fixed:
    mode = "fixture"

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        from rvg.contracts import EmbeddingBatch

        return EmbeddingBatch(tuple(tuple(Q) for _ in texts), SPACE, ("f",), ())
