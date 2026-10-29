from __future__ import annotations

import asyncio
import email.utils
import json
import math
import random
import time
from datetime import UTC, datetime

import httpx

from .errors import BudgetExceeded, ContractError, ProviderError
from .providers import claude_code
from .runtime import CallContext

RETRYABLE = {429, 500, 502, 503, 504, 529}


class JsonTransport:
    """One owner of HTTP retries; bounded reads and sanitized errors."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        timeout: float = 20,
        max_attempts: int = 3,
        max_response_bytes: int = 4_000_000,
        claude_code_binary: str = "claude",
        claude_code_slots: int = 8,
        claude_code_timeout: float = 90,
        claude_code_prewarm: int = claude_code.PREWARM,
    ):
        self.client, self.timeout = client, timeout
        self.max_attempts, self.max_response_bytes = max_attempts, max_response_bytes
        # Requests to `claude-code://` run the Claude Code CLI (providers/claude_code.py) instead of
        # HTTP; at most `claude_code_slots` processes at a time.
        self.claude_code_binary, self.claude_code_timeout = claude_code_binary, claude_code_timeout
        self.claude_code_slots, self._claude_code_gate = claude_code_slots, None
        # PERF-1: processes started ahead of a recurring CLI call (0: none, every call starts one)
        self.claude_code_prewarm = claude_code_prewarm

    def open_call(self, url: str, headers: dict[str, str], body: dict):
        """One logical request (all of its attempts); subclasses that record or replay HTTP
        exchanges return their handle for it. The live transport needs none."""
        return

    async def send(self, url, headers, encoded, context, *, call=None, attempt=0):
        """One HTTP attempt: (status, Retry-After, parsed JSON object for a 2xx else None).
        Timeouts, network and response-contract errors propagate to `post`, which owns the
        retry policy and the cost/attempt accounting."""
        if url.startswith(claude_code.SCHEME):
            if self._claude_code_gate is None:
                self._claude_code_gate = asyncio.Semaphore(self.claude_code_slots)
            async with self._claude_code_gate:
                return await claude_code.send(
                    encoded,
                    binary=self.claude_code_binary,
                    timeout=min(self.claude_code_timeout, context.remaining()),
                    max_bytes=self.max_response_bytes,
                    prewarm=self.claude_code_prewarm,
                )
        async with asyncio.timeout(min(self.timeout, context.remaining())):
            async with self.client.stream(
                "POST",
                url,
                headers={**headers, "Content-Type": "application/json"},
                content=encoded,
                timeout=min(self.timeout, context.remaining()),
                follow_redirects=False,
            ) as response:
                status = response.status_code
                retry_after = response.headers.get("Retry-After")
                if not 200 <= status < 300:
                    return status, retry_after, None
                raw = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    raw.extend(chunk)
                    if len(raw) > self.max_response_bytes:
                        raise ContractError("response_too_large")
                try:
                    data = json.loads(raw)
                except (ValueError, UnicodeError):
                    raise ContractError("invalid_response_json") from None
                if not isinstance(data, dict):
                    raise ContractError("invalid_response_shape")
                return status, retry_after, data

    async def pause(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: dict,
        context: CallContext,
        no_retry_statuses: frozenset[int] = frozenset(),
        provider: str | None = None,
        model: str | None = None,
    ) -> dict:
        try:
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        except (ValueError, TypeError):
            raise ContractError("invalid_request_json") from None
        call = self.open_call(url, headers, body)
        for attempt in range(self.max_attempts):
            context.begin_attempt()
            started, status, retry_after, error = time.monotonic(), None, None, None
            try:
                status, retry_after, data = await self.send(
                    url, headers, encoded, context, call=call, attempt=attempt
                )
                if data is not None:
                    return data
            except (httpx.TimeoutException, TimeoutError):
                error = "ambiguous_timeout"
                context.record_cost(None)
                raise ProviderError(error) from None
            except httpx.HTTPError:
                error = "network_error"
                context.record_cost(None)
                raise ProviderError(error) from None
            except ContractError:
                error = "invalid_response"
                context.record_cost(None)
                raise
            finally:
                context.attempts.append(
                    {
                        "provider": provider,
                        "model": model,
                        "http_status": status,
                        "error": error,
                        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                    }
                )
            context.record_cost(None)
            if (
                status in no_retry_statuses
                or status not in RETRYABLE
                or attempt + 1 == self.max_attempts
            ):
                raise ProviderError("upstream_http_error", status=status)
            delay = min(2**attempt + random.random(), 8)
            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    try:
                        dt = email.utils.parsedate_to_datetime(retry_after)
                        wait = (dt - datetime.now(UTC)).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        wait = delay
                if not math.isfinite(wait):
                    wait = delay
                delay = max(delay, wait)
            if delay > 30 or delay >= context.remaining():
                raise BudgetExceeded("retry_wait_exceeds_deadline")
            await self.pause(delay)
        raise ProviderError("retry_exhausted")
