"""Structured generation through the Claude Code CLI in non-interactive mode (`claude -p`), the
subscription path of parent contract U21 (docs/UNIVERSAL_U21_PROTOCOL.md §8.1).

`JsonTransport.send` hands every request whose URL has the `claude-code` scheme to `send` below,
so the recording and replay transports (tapes keyed by the canonical request body) work unchanged
and a replay never starts a process. The request body is the canonical
`{model, system, input, json_schema, options}`; the reply is the CLI's JSON result object.

Isolation (all measured on Claude Code 2.1.302): no tools, no setting sources (otherwise the
user's global CLAUDE.md enters the context), no MCP servers, no session file, stdin closed, a new
empty working directory per call and the parent Claude Code session variables removed. `HOME` is
kept because the subscription login lives there; `--bare` would disable it.

PERF-1 (parent docs/rvi/PERF_1_PROTOCOL.md §12): a call returns as soon as the CLI has printed a
successful result; the process's own shutdown (about a second) is reaped in the background. With a
prewarm pool (`prewarm` > 0), a call whose model, system prompt and schema have come before is
given to a process started ahead of it with the same flags, which reads its one input from stdin
(stream-json) and ends after it — one process per request, no session shared. The model, the
instruction, the input and the tape key are unchanged; a replay still never starts a process."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import os
import shutil
import signal
import tempfile
from collections import Counter

from ..errors import ContractError

SCHEME = "claude-code://"
URL = SCHEME + "print"
DROPPED_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
# Signals of a usage limit or an overloaded service: retried like HTTP 429.
LIMIT_STATUSES = {429, 529}
LIMIT_WORDS = ("rate limit", "rate_limit", "overloaded", "usage limit", "too many requests")


def argv(binary: str, body: dict) -> list[str]:
    return [
        binary,
        "-p",
        body["input"],
        "--model",
        body["model"],
        "--output-format",
        "json",
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--system-prompt",
        body["system"],
        "--json-schema",
        json.dumps(body["json_schema"], ensure_ascii=False, separators=(",", ":")),
    ]


def environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in DROPPED_ENV}


def classify(returncode: int, stdout: bytes) -> tuple[int, dict | None]:
    """(status, data): 200 with the result object, 429 for limits, 502 for anything else."""
    try:
        data = json.loads(stdout)
    except (ValueError, UnicodeError):
        data = None
    if not isinstance(data, dict):
        return 502, None
    if returncode == 0 and data.get("subtype") == "success" and data.get("is_error") is False:
        return 200, data
    status = data.get("api_error_status")
    text = str(data.get("result") or "").lower()
    if status in LIMIT_STATUSES or any(word in text for word in LIMIT_WORDS):
        return 429, None
    if returncode == 0 and data.get("is_error") is False:
        # A finished call that is not a success (e.g. max turns): the provider rejects it.
        return 200, data
    return 502, None


async def stop(process) -> None:
    """Kill the process unless it has already ended, and reap it. A process that ends between
    the check and the kill (`ProcessLookupError`) is already stopped; any other failure while
    stopping is left to the operating system rather than raised over the caller's exception."""
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    with contextlib.suppress(Exception):  # never hide the timeout or cancellation being raised
        await process.wait()


# PERF-1: how long a process may take to end after its result before it is killed, and how long a
# pre-started process may wait for its input before it is replaced.
EXIT_GRACE = 30.0
PREWARM_MAX_AGE = 300.0
# PERF-1: processes kept started ahead of recurring calls (the default of `JsonTransport`); each
# idle one holds about 270 MB (A′, Claude Code 2.1.306). 0: every call starts its own process.
PREWARM = 2
_REAPERS: set[asyncio.Task] = set()
_POOLS: dict[int, _Pool] = {}
_IDLE_PIDS: set[int] = set()


def stream_argv(binary: str, body: dict) -> list[str]:
    """The pre-started form of `argv`: the same flags, the input read from stdin (stream-json)."""
    args = argv(binary, body)
    i = args.index("-p")
    rest = args[i + 2 :]
    j = rest.index("--output-format")
    rest = rest[:j] + rest[j + 2 :]
    return [
        *args[:i],
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        *rest,
    ]


def _final(data) -> bool:
    return (
        isinstance(data, dict)
        and data.get("subtype") == "success"
        and data.get("is_error") is False
    )


async def _drain(stream) -> None:
    with contextlib.suppress(Exception):
        while await stream.read(65536):
            pass


async def _read_json(process, max_bytes: int) -> tuple[bytes, bool]:
    """(stdout, early): stops at a complete successful result object (early), else at EOF."""
    out = bytearray()
    while chunk := await process.stdout.read(65536):
        out.extend(chunk)
        if len(out) > max_bytes:
            raise ContractError("response_too_large")
        try:
            data = json.loads(out)
        except (ValueError, UnicodeError):
            continue
        if _final(data):
            return bytes(out), True
    return bytes(out), False


async def _read_stream(process, max_bytes: int) -> tuple[bytes, bool]:
    """(the `result` event, early): early when it is successful; the last one seen at EOF."""
    pending, total, last = b"", 0, b""
    while chunk := await process.stdout.read(65536):
        total += len(chunk)
        if total > max_bytes:
            raise ContractError("response_too_large")
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                last = line
                if _final(event):
                    return line, True
    return last, False


def _reap_later(process, workdir: str, drain) -> None:
    """End a process that has already given its result: wait for it to exit by itself, kill it
    after `EXIT_GRACE`, and kill it at once if the loop shuts down; never leave it unreaped."""

    async def reap():
        try:
            await asyncio.wait_for(process.wait(), EXIT_GRACE)
        except TimeoutError:
            await stop(process)
        except asyncio.CancelledError:
            # the loop is shutting down: kill now, and never let the reaping hold the shutdown
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(process.wait(), 5)
            raise
        finally:
            drain.cancel()
            shutil.rmtree(workdir, ignore_errors=True)

    task = asyncio.get_running_loop().create_task(reap())
    _REAPERS.add(task)
    task.add_done_callback(_REAPERS.discard)


class _Pool:
    """The pre-started processes of one event loop, by (binary, model, system prompt, schema)."""

    def __init__(self, loop):
        self.loop = loop
        self.idle: dict[tuple, list] = {}
        self.seen: Counter = Counter()
        self.spawning = 0

    def size(self) -> int:
        return sum(len(v) for v in self.idle.values()) + self.spawning

    async def end(self) -> None:
        for entries in self.idle.values():
            for process, workdir, _, drain in entries:
                _IDLE_PIDS.discard(process.pid)
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), 5)
                drain.cancel()
                shutil.rmtree(workdir, ignore_errors=True)
        self.idle.clear()


async def _guard(pool: _Pool):
    """Ends the pool's processes when its loop shuts down: `asyncio.run` closes the async
    generators it started (`shutdown_asyncgens`) before the loop closes."""
    try:
        yield
    finally:
        if _POOLS.get(id(pool.loop)) is pool:
            del _POOLS[id(pool.loop)]
        await pool.end()


def _kill_pid(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)
    _IDLE_PIDS.discard(pid)


@atexit.register
def _kill_idle() -> None:
    for pid in list(_IDLE_PIDS):
        _kill_pid(pid)


async def _pool() -> _Pool:
    loop = asyncio.get_running_loop()
    for key, pool in list(_POOLS.items()):
        if pool.loop is not loop and pool.loop.is_closed():
            for entries in pool.idle.values():
                for process, workdir, _, _ in entries:
                    _kill_pid(process.pid)
                    shutil.rmtree(workdir, ignore_errors=True)
            del _POOLS[key]
    pool = _POOLS.get(id(loop))
    if pool is None or pool.loop is not loop:
        pool = _POOLS[id(loop)] = _Pool(loop)
        pool.guard = _guard(pool)  # held here; the loop tracks it once it has started
        await pool.guard.__anext__()
    return pool


def _key(binary: str, body: dict) -> tuple:
    return (binary, body["model"], body["system"], json.dumps(body["json_schema"], sort_keys=True))


def _retire(entry) -> None:
    """End a pre-started process that will not be used (stale or already ended)."""
    process, workdir, _, drain = entry
    _IDLE_PIDS.discard(process.pid)
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        _reap_later(process, workdir, drain)
    else:
        drain.cancel()
        shutil.rmtree(workdir, ignore_errors=True)


def _take(pool: _Pool, key: tuple, max_age: float):
    """A live pre-started process for `key` (the oldest fitting one), retiring stale ones."""
    now = pool.loop.time()
    for k, entries in list(pool.idle.items()):
        keep = []
        for entry in entries:
            if entry[0].returncode is not None or now - entry[2] > max_age:
                _retire(entry)
            else:
                keep.append(entry)
        pool.idle[k] = keep
    entries = pool.idle.get(key) or []
    if entries:
        entry = entries.pop(0)
        _IDLE_PIDS.discard(entry[0].pid)
        return entry
    return None


def _refill(pool: _Pool, key: tuple, binary: str, body: dict, limit: int) -> None:
    """Start one process for `key` ahead of its next call when the key has come before and the
    pool is below `limit` processes (idle and starting)."""
    if pool.seen[key] < 2 or pool.size() >= limit:
        return
    pool.spawning += 1

    async def start():
        workdir = tempfile.mkdtemp(prefix="rvg-claude-code-")
        try:
            process = await asyncio.create_subprocess_exec(
                *stream_argv(binary, body),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=environment(),
            )
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        finally:
            pool.spawning -= 1
        if _POOLS.get(id(pool.loop)) is not pool:  # the pool was closed while this one started
            await stop(process)
            shutil.rmtree(workdir, ignore_errors=True)
            return
        _IDLE_PIDS.add(process.pid)
        drain = asyncio.ensure_future(_drain(process.stderr))
        pool.idle.setdefault(key, []).append((process, workdir, pool.loop.time(), drain))

    task = pool.loop.create_task(start())
    _REAPERS.add(task)
    task.add_done_callback(_REAPERS.discard)


async def close_prewarm() -> None:
    """End the running loop's pre-started processes now (they also end with the loop)."""
    pool = _POOLS.pop(id(asyncio.get_running_loop()), None)
    if pool is not None:
        await pool.end()


async def send(
    encoded: bytes,
    *,
    binary: str,
    timeout: float,
    max_bytes: int,
    prewarm: int = 0,
    prewarm_max_age: float = PREWARM_MAX_AGE,
) -> tuple[int, None, dict | None]:
    """One CLI call: (status, Retry-After, data) like `JsonTransport.send`."""
    try:
        body = json.loads(encoded)
    except ValueError:
        raise ContractError("invalid_request_json") from None
    warm = None
    if prewarm > 0:
        pool, key = await _pool(), _key(binary, body)
        warm = _take(pool, key, prewarm_max_age)
        pool.seen[key] += 1
        _refill(pool, key, binary, body, prewarm)
    if warm is None:
        workdir = tempfile.mkdtemp(prefix="rvg-claude-code-")
        try:
            process = await asyncio.create_subprocess_exec(
                *argv(binary, body),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=environment(),
            )
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        drain, read = asyncio.ensure_future(_drain(process.stderr)), _read_json
    else:
        process, workdir, _, drain = warm
        read = _read_stream
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    try:
        if warm is not None:
            message = {"type": "user", "message": {"role": "user", "content": body["input"]}}
            process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
            await asyncio.wait_for(process.stdin.drain(), max(0.0, deadline - loop.time()))
            process.stdin.close()
        stdout, early = await asyncio.wait_for(
            read(process, max_bytes), max(0.0, deadline - loop.time())
        )
        if not early:
            await asyncio.wait_for(process.wait(), max(0.0, deadline - loop.time()))
    except (TimeoutError, asyncio.CancelledError, ContractError, OSError):
        # The original exception goes on: a timeout becomes `ProviderError` in
        # `JsonTransport.post`; a cancellation stays a cancellation (the caller's). Ending
        # the process must not replace it (D-CLI-KILL, parent docs/rvi/FIX_CLI_KILL_PROTOCOL.md).
        await stop(process)
        drain.cancel()
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    if early:
        _reap_later(process, workdir, drain)
        return (*_classified(0, stdout),)
    drain.cancel()
    shutil.rmtree(workdir, ignore_errors=True)
    return (*_classified(process.returncode, stdout),)


def _classified(returncode, stdout) -> tuple[int, None, dict | None]:
    status, data = classify(returncode, stdout)
    return status, None, data
