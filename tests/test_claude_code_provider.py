"""U21 Claude Code CLI generation (parent docs/UNIVERSAL_U21_PROTOCOL.md §8.1), offline: a fake
`claude` executable records how it was started and answers like the CLI's JSON output."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys

import pytest

from rvg.contracts import Contract
from rvg.errors import ContractError, ProviderError
from rvg.providers import claude_code
from rvg.providers.generation import (
    ClaudeCodeGenerationProvider,
    GenerationOptions,
    GenerationRequest,
)
from rvg.runtime import CallContext
from rvg.transport import JsonTransport
from rvg.universal.replay import RecordingTransport, ReplayTransport, Tape
from rvg.universal.trace import CURRENT_REQUEST

FAKE = r"""#!{python}
import json, os, sys, time
args = sys.argv[1:]
state = os.environ["FAKE_STATE"]
mode = open(os.path.join(state, "mode")).read().strip()
calls = os.path.join(state, "calls")
n = len(open(calls).read().splitlines()) if os.path.exists(calls) else 0
seen = {{
    "args": args,
    "cwd_entries": os.listdir(os.getcwd()),
    "cwd": os.getcwd(),
    "env_claudecode": [k for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT") if k in os.environ],
    "stdin": sys.stdin.read(),
}}
with open(calls, "a") as f:
    f.write(json.dumps(seen) + "\n")
model = args[args.index("--model") + 1]
if mode == "sleep":
    time.sleep(5)
if mode == "limit" and n == 0:
    print(json.dumps({{"type": "result", "subtype": "success", "is_error": True,
                      "api_error_status": 429, "result": "rate limit"}}))
    sys.exit(1)
if mode == "garbage":
    print("not json")
    sys.exit(1)
answer_model = "claude-sonnet-5-5" if mode == "mismatch" else model
print(json.dumps({{
    "type": "result", "subtype": "success", "is_error": False, "api_error_status": None,
    "session_id": "s-%d" % n, "duration_ms": 1234, "stop_reason": "tool_use",
    "total_cost_usd": 0.0003,
    "usage": {{"input_tokens": 2, "output_tokens": 30, "cache_read_input_tokens": 2400,
              "cache_creation_input_tokens": 0, "output_tokens_details": {{"thinking_tokens": 7}}}},
    "modelUsage": {{answer_model: {{"inputTokens": 2, "outputTokens": 30}}}},
    "structured_output": {{"answer": "Paris", "confident": True}},
    "result": "",
}}))
"""


class Answer(Contract):
    answer: str
    confident: bool


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    binary = tmp_path / "claude"
    binary.write_text(FAKE.format(python=sys.executable))
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_STATE", str(state))
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")

    def set_mode(mode):
        (state / "mode").write_text(mode)

    def calls():
        path = state / "calls"
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    set_mode("ok")
    return binary, set_mode, calls


def request(system="Answer in English."):
    return GenerationRequest(
        role="surface",
        system=system,
        payload={"question": "Capital of France?"},
        response_type=Answer,
        schema_name="answer_v1",
        prompt_version="test",
        options=GenerationOptions(temperature=0, max_tokens=256),
    )


def run(coro):
    token = CURRENT_REQUEST.set("r1")
    try:
        return asyncio.run(coro)
    finally:
        CURRENT_REQUEST.reset(token)


def context():
    return CallContext.with_timeout(60, max_http_attempts=10, max_generation_calls=10)


def provider(binary, transport_cls=JsonTransport, **options):
    transport = transport_cls(None, claude_code_binary=str(binary), **options)
    return ClaudeCodeGenerationProvider(model="claude-haiku-5-5", transport=transport), transport


def test_the_command_line_carries_every_isolation_flag():
    body = {"model": "m", "system": "S", "input": "{}", "json_schema": {"type": "object"}}
    args = claude_code.argv("claude", body)
    assert args[:3] == ["claude", "-p", "{}"]
    for flag, value in [
        ("--model", "m"),
        ("--output-format", "json"),
        ("--tools", ""),
        ("--setting-sources", ""),
        ("--system-prompt", "S"),
        ("--json-schema", '{"type":"object"}'),
    ]:
        assert args[args.index(flag) + 1] == value
    assert "--strict-mcp-config" in args and "--no-session-persistence" in args
    assert "--bare" not in args  # it would turn the subscription login off


def test_a_call_runs_in_a_new_empty_folder_without_the_session_variables(fake_cli):
    binary, _, calls = fake_cli
    gen, _ = provider(binary)
    result = run(gen.generate_structured(request(), context()))
    assert result.value == Answer(answer="Paris", confident=True)
    (seen,) = calls()
    assert seen["cwd_entries"] == [] and seen["env_claudecode"] == [] and seen["stdin"] == ""
    assert not os.path.exists(seen["cwd"])  # removed after the call
    assert json.loads(seen["args"][seen["args"].index("-p") + 1]) == {
        "question": "Capital of France?"
    }


def test_the_metadata_is_the_subscription_path_and_the_meter_gets_zero(fake_cli):
    binary, _, _ = fake_cli
    gen, _ = provider(binary)
    ctx = context()
    result = run(gen.generate_structured(request(), ctx))
    meta = result.metadata
    assert meta["provider"] == "claude_code" and meta["billing"] == "subscription"
    assert meta["response_model"] == "claude-haiku-5-5" and meta["cost_usd"] == 0.0
    assert meta["nominal_cost_usd"] == 0.0003 and meta["options_applied"] is False
    assert meta["cached_tokens"] == 2400 and meta["reasoning_tokens"] == 7
    assert ctx.reported_cost_usd == 0.0 and ctx.unknown_cost_calls == 0
    assert ctx.generations[-1]["status"] == "SCHEMA_VALID"


def test_a_usage_limit_is_retried_like_http_429(fake_cli, monkeypatch):
    binary, set_mode, calls = fake_cli
    set_mode("limit")

    async def no_wait(self, seconds):
        return None

    monkeypatch.setattr(JsonTransport, "pause", no_wait)
    gen, _ = provider(binary, max_attempts=3)
    ctx = context()
    result = run(gen.generate_structured(request(), ctx))
    assert result.value.answer == "Paris" and len(calls()) == 2
    assert [a["http_status"] for a in ctx.attempts] == [429, 200]


def test_an_unreadable_answer_is_an_upstream_error_after_the_retries(fake_cli, monkeypatch):
    binary, set_mode, _ = fake_cli
    set_mode("garbage")

    async def no_wait(self, seconds):
        return None

    monkeypatch.setattr(JsonTransport, "pause", no_wait)
    gen, _ = provider(binary, max_attempts=2)
    with pytest.raises(ProviderError) as err:
        run(gen.generate_structured(request(), context()))
    assert err.value.code == "upstream_http_error"


def test_another_answering_model_is_a_mismatch(fake_cli):
    binary, set_mode, _ = fake_cli
    set_mode("mismatch")
    gen, _ = provider(binary)
    ctx = context()
    with pytest.raises(ContractError) as err:
        run(gen.generate_structured(request(), ctx))
    assert err.value.code == "generation_response_model_mismatch"


def test_a_slow_call_times_out_and_the_process_is_killed(fake_cli):
    binary, set_mode, _ = fake_cli
    set_mode("sleep")
    gen, _ = provider(binary, claude_code_timeout=0.5)
    with pytest.raises(ProviderError) as err:
        run(gen.generate_structured(request(), context()))
    assert err.value.code == "ambiguous_timeout"


def test_a_recorded_call_replays_without_starting_a_process(fake_cli, tmp_path):
    binary, _, calls = fake_cli
    tape = Tape()
    gen, _ = provider(
        binary, transport_cls=lambda client, **o: RecordingTransport(client, tape, **o)
    )
    recorded = run(gen.generate_structured(request(), context()))
    assert len(calls()) == 1 and len(tape.entries) == 1
    entry = tape.entries[0]
    assert entry["url"] == claude_code.URL and entry["headers"] == {}
    assert entry["body"]["model"] == "claude-haiku-5-5"
    replay = ReplayTransport(tape, claude_code_binary=str(tmp_path / "missing"))
    again = run(
        ClaudeCodeGenerationProvider(
            model="claude-haiku-5-5", transport=replay
        ).generate_structured(request(), context())
    )
    assert again.value == recorded.value and len(calls()) == 1
    assert replay.unused("r1") == []


def test_a_different_system_prompt_is_a_different_tape_key(fake_cli):
    binary, _, _ = fake_cli
    tape = Tape()
    gen, _ = provider(
        binary, transport_cls=lambda client, **o: RecordingTransport(client, tape, **o)
    )
    run(gen.generate_structured(request("A"), context()))
    run(gen.generate_structured(request("B"), context()))
    assert len({e["key"] for e in tape.entries}) == 2


def test_classify_maps_the_cli_outcomes():
    ok = {"subtype": "success", "is_error": False}
    assert claude_code.classify(0, json.dumps(ok).encode()) == (200, ok)
    limited = {"subtype": "success", "is_error": True, "api_error_status": 529}
    assert claude_code.classify(1, json.dumps(limited).encode()) == (429, None)
    words = {"subtype": "error", "is_error": True, "result": "Claude AI usage limit reached"}
    assert claude_code.classify(1, json.dumps(words).encode()) == (429, None)
    other = {"subtype": "error_during_execution", "is_error": True, "result": "boom"}
    assert claude_code.classify(1, json.dumps(other).encode()) == (502, None)
    assert claude_code.classify(0, b"[]") == (502, None)
    unfinished = {"subtype": "error_max_turns", "is_error": False}
    assert claude_code.classify(0, json.dumps(unfinished).encode()) == (200, unfinished)
