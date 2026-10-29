"""PERF-1 (parent docs/rvi/PERF_1_PROTOCOL.md §12): a CLI call returns once the result is printed
and the process's own shutdown is reaped in the background (G2); with a prewarm pool, a call whose
model, system prompt and schema have come before is given to a process started ahead of it with
the same flags, which reads its one input from stdin and ends (G1). Timeouts and cancellations
keep D-CLI-KILL's meaning and no process is left behind. Offline: a fake `claude`."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time

import pytest

from rvg.providers import claude_code

FAKE = r"""#!{python}
import json, os, sys, time
args = sys.argv[1:]
state = os.environ["FAKE_STATE"]
mode = open(os.path.join(state, "mode")).read().strip()
stream = "--input-format" in args
record = {{"pid": os.getpid(), "args": args, "stream": stream, "started": time.time()}}
if stream:
    line = sys.stdin.readline()
    record["input"] = json.loads(line)["message"]["content"] if line.strip() else None
    record["input_at"] = time.time()
else:
    record["input"] = args[args.index("-p") + 1]
with open(os.path.join(state, "calls"), "a") as f:
    f.write(json.dumps(record) + "\n")
if mode == "hang" or (mode == "hang_stream" and stream):
    time.sleep(60)
model = args[args.index("--model") + 1]
result = {{"type": "result", "subtype": "success", "is_error": False, "api_error_status": None,
           "session_id": "s", "duration_ms": 10, "total_cost_usd": 0.0,
           "usage": {{"input_tokens": 1, "output_tokens": 1}},
           "modelUsage": {{model: {{"inputTokens": 1, "outputTokens": 1}}}},
           "structured_output": {{"echo": record["input"]}}, "result": ""}}
if mode == "fail":
    result.update(is_error=True, api_error_status=429, result="rate limit")
if stream:
    print(json.dumps({{"type": "system", "subtype": "init"}}), flush=True)
print(json.dumps(result), flush=True)
if mode == "linger":
    time.sleep(float(open(os.path.join(state, "linger")).read()))
sys.exit(1 if mode == "fail" else 0)
"""


@pytest.fixture
def fake(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    binary = tmp_path / "claude"
    binary.write_text(FAKE.format(python=sys.executable))
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_STATE", str(state))

    def mode(name, linger=0.0):
        (state / "mode").write_text(name)
        (state / "linger").write_text(str(linger))

    def calls():
        path = state / "calls"
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    mode("ok")
    return str(binary), mode, calls


@pytest.fixture
def spawned(monkeypatch):
    """The ids of the processes the provider started, seen from the parent: a test that a process
    was killed must not need the fake to have started far enough to write its record. Under load
    it may not within a one-second timeout, and it is killed all the same (0.1.0, parent PUB-4:
    the record was missing and the test failed with an `IndexError`)."""
    pids = []
    create = asyncio.create_subprocess_exec

    async def recorded(*args, **kwargs):
        process = await create(*args, **kwargs)
        pids.append(process.pid)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorded)
    return pids


def body(text="q", schema=None, system="S"):
    return json.dumps(
        {
            "model": "claude-haiku-5-5",
            "system": system,
            "input": text,
            "json_schema": schema or {"type": "object"},
        }
    ).encode()


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers signal 0; ps tells it apart
    out = os.popen(f"ps -o stat= -p {pid}").read().strip()
    return bool(out) and not out.startswith("Z")


async def send(binary, encoded, **kw):
    kw.setdefault("timeout", 20)
    return await claude_code.send(encoded, binary=binary, max_bytes=1_000_000, **kw)


# --- G2: the result returns before the process ends --------------------------------------------


def test_a_result_returns_before_the_process_ends_and_is_reaped(fake):
    binary, mode, calls = fake
    mode("linger", 2.0)

    async def go():
        t = time.monotonic()
        status, _, data = await send(binary, body("a"))
        took = time.monotonic() - t
        pid = calls()[0]["pid"]
        assert status == 200 and data["structured_output"] == {"echo": "a"}
        assert took < 1.5 and alive(pid)  # returned while the process was still ending
        await asyncio.sleep(2.5)
        assert not alive(pid)  # reaped once it ended by itself
        return pid

    asyncio.run(go())


def test_the_loop_shutting_down_kills_a_process_still_ending(fake):
    binary, mode, calls = fake
    mode("linger", 30.0)
    asyncio.run(send(binary, body("a")))
    time.sleep(0.5)
    assert not alive(calls()[0]["pid"])


def test_a_failed_result_still_waits_for_the_exit_and_is_classified_as_before(fake):
    binary, mode, _ = fake
    mode("fail")
    status, _, data = asyncio.run(send(binary, body("a")))
    assert (status, data) == (429, None)


def test_a_timeout_keeps_its_meaning_and_kills_the_process(fake, spawned):
    binary, mode, calls = fake
    mode("hang")
    with pytest.raises(TimeoutError):
        asyncio.run(send(binary, body("a"), timeout=1))
    time.sleep(0.3)
    assert len(spawned) == 1 and not alive(spawned[0])
    assert all(r["pid"] in spawned for r in calls())  # the fake's own record, if it got that far


def test_a_cancellation_stays_a_cancellation_and_kills_the_process(fake, spawned):
    binary, mode, calls = fake
    mode("hang")

    async def go():
        task = asyncio.ensure_future(send(binary, body("a")))
        await asyncio.sleep(1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    time.sleep(0.3)
    assert len(spawned) == 1 and not alive(spawned[0])
    assert all(r["pid"] in spawned for r in calls())


# --- G1: a recurring call takes a process started ahead of it ----------------------------------


def test_a_recurring_call_takes_a_prestarted_process_with_the_same_flags(fake):
    binary, _, calls = fake

    async def go():
        outs = []
        for i in range(3):
            outs.append(await send(binary, body(f"q{i}"), prewarm=1))
            await asyncio.sleep(0.8)  # the refill starts in the background
        await claude_code.close_prewarm()
        return outs

    outs = asyncio.run(go())
    assert [o[2]["structured_output"]["echo"] for o in outs] == ["q0", "q1", "q2"]
    rows = calls()
    stream = [r for r in rows if r["stream"] and r["input"] is not None]
    assert [r["input"] for r in stream] == ["q2"]  # the third call ran on the pre-started process
    cold = [r for r in rows if not r["stream"]]
    flags = [
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
    ]
    for r in cold + stream:
        joined = r["args"]
        assert all(f in joined for f in flags)
    s = stream[0]["args"]
    assert s[s.index("--input-format") + 1] == "stream-json" and "q2" not in s
    assert (
        s[s.index("--json-schema") + 1]
        == cold[0]["args"][cold[0]["args"].index("--json-schema") + 1]
    )
    assert stream[0]["input_at"] > stream[0]["started"]


def test_a_call_seen_once_is_never_prestarted(fake):
    binary, _, calls = fake

    async def go():
        for i in range(3):  # three different schemas (a writer's evidence enum differs per call)
            await send(binary, body("q", schema={"enum": [i]}), prewarm=2)
            await asyncio.sleep(0.5)
        await claude_code.close_prewarm()

    asyncio.run(go())
    assert not any(r["stream"] for r in calls())


def test_the_pool_stays_within_its_size(fake):
    binary, _, calls = fake

    async def go():
        for i in range(6):
            await send(binary, body("q", schema={"k": i % 2}), prewarm=1)
            await asyncio.sleep(0.5)
        pool = claude_code._POOLS[id(asyncio.get_running_loop())]
        size = pool.size()
        await claude_code.close_prewarm()
        return size

    assert asyncio.run(go()) <= 1
    started = [r for r in calls() if r["stream"]]
    assert len(started) <= 4  # never more than one waiting at a time


def test_a_stale_prestarted_process_is_replaced_and_ended(fake):
    binary, _, calls = fake

    async def go():
        for _ in range(2):
            await send(binary, body("q"), prewarm=1)
        await asyncio.sleep(0.8)
        idle = [
            e[0].pid
            for v in claude_code._POOLS[id(asyncio.get_running_loop())].idle.values()
            for e in v
        ]
        out = await send(binary, body("late"), prewarm=1, prewarm_max_age=0)
        await asyncio.sleep(0.5)
        await claude_code.close_prewarm()
        return idle, out

    idle, out = asyncio.run(go())
    assert out[2]["structured_output"] == {"echo": "late"}
    assert idle and not any(alive(pid) for pid in idle)
    assert not any(r["stream"] and r["input"] == "late" for r in calls())


def test_a_timeout_on_a_prestarted_process_keeps_its_meaning(fake):
    binary, mode, calls = fake

    mode("hang_stream")  # cold calls answer; the pre-started process hangs after its input

    async def go():
        for _ in range(2):
            await send(binary, body("q"), prewarm=1)
        await asyncio.sleep(0.8)
        with pytest.raises(TimeoutError):
            await send(binary, body("q"), prewarm=1, timeout=1)
        await claude_code.close_prewarm()

    asyncio.run(go())
    time.sleep(0.3)
    hung = [r for r in calls() if r["stream"] and r["input"] == "q"]
    assert hung and not alive(hung[0]["pid"])


def test_closing_and_a_new_loop_end_the_idle_processes(fake):
    binary, _, _ = fake

    async def first():
        for _ in range(2):
            await send(binary, body("q"), prewarm=1)
        await asyncio.sleep(0.8)
        return [
            e[0].pid
            for v in claude_code._POOLS[id(asyncio.get_running_loop())].idle.values()
            for e in v
        ]

    idle = asyncio.run(first())  # the loop ends with a process waiting for its input
    assert idle

    async def second():
        await send(binary, body("other"), prewarm=1)  # a new loop retires the closed loop's pool
        await claude_code.close_prewarm()

    asyncio.run(second())
    time.sleep(0.5)
    assert not any(alive(pid) for pid in idle)


def test_the_transport_prestarts_by_default_and_zero_turns_it_off(fake):
    from rvg.runtime import CallContext
    from rvg.transport import JsonTransport

    binary, _, calls = fake
    assert JsonTransport(None).claude_code_prewarm == claude_code.PREWARM == 2

    async def go(prewarm):
        transport = JsonTransport(None, claude_code_binary=binary, claude_code_prewarm=prewarm)
        for i in range(3):
            await transport.send(
                claude_code.URL, {}, body(f"p{prewarm}-{i}"), CallContext.with_timeout(30)
            )
            await asyncio.sleep(0.8)
        await claude_code.close_prewarm()

    asyncio.run(go(0))
    assert not any(r["stream"] for r in calls())  # 0: every call starts its own process
    asyncio.run(go(2))
    assert [r["input"] for r in calls() if r["stream"] and r["input"]] == ["p2-2"]
