"""Record and replay every provider exchange of an answer at the HTTP-attempt level (N0).

`RecordingTransport` is the live transport plus a tape: each logical request (all its attempts,
including retried 429/5xx, timeouts and malformed bodies) is stored with its canonical request
hash, the runtime request it belongs to and, for decision calls, the controller stage that asked
it. `ReplayTransport` answers the same requests from the tape only: the request bytes must hash
to a recorded key, identical requests are served in the order they were made within their
runtime request, and a missing exchange raises `ReplayMiss` — it never falls back to a live
model. Because the whole provider stack (request building, response parsing, validation, cost
and attempt accounting) runs again on replay, a replayed answer differs from the recorded one
only if the code, the policy or the stored sources differ.

Secrets never enter a tape: authorization headers are dropped before hashing or storing."""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from datetime import UTC, datetime

import httpx

from ..errors import ContractError, ProviderError
from ..transport import JsonTransport
from .trace import CURRENT_PARENT, CURRENT_REQUEST, CURRENT_STAGE, CURRENT_TRACE, canonical_hash

SECRET_HEADER_PARTS = ("authorization", "key", "token", "secret", "cookie")


class ReplayMiss(ProviderError):
    pass


def public_headers(headers):
    return {
        k: v
        for k, v in sorted(headers.items())
        if not any(part in k.lower() for part in SECRET_HEADER_PARTS)
    }


def exchange_key(url, headers, body):
    return canonical_hash({"url": url, "headers": public_headers(headers), "body": body})


class Tape:
    """Logical requests in the order they were opened. `calls[(request_id, key)]` lists the
    identical requests of one runtime request in call order."""

    def __init__(self, entries=()):
        self.entries, self.calls, self.sites = [], {}, {}
        for entry in entries:
            self._index(entry)

    def _index(self, entry):
        self.entries.append(entry)
        self.calls.setdefault((entry["request_id"], entry["key"]), []).append(entry)
        if entry.get("site"):
            self.sites.setdefault((entry["request_id"], entry["site"], entry["key"]), []).append(
                entry
            )

    def of(self, request_id):
        return [e for e in self.entries if e["request_id"] == request_id]

    def ambiguous(self, request_id):
        """Keys of a request with several occurrences whose recorded outcomes differ. Replay
        serves occurrences in the order the requests are opened; when identical requests from
        different call sites race, that order — and so which outcome each site gets — can depend
        on scheduling, so these are reported rather than assumed deterministic."""
        out = []
        for (rid, key), calls in self.calls.items():
            if rid != request_id or len(calls) < 2:
                continue
            outcomes = {
                canonical_hash([(a["status"], a.get("error"), a["data"]) for a in c["attempts"]])
                for c in calls
            }
            if len(outcomes) > 1:
                out.append(
                    {"key": key, "occurrences": len(calls), "stages": [c["stage"] for c in calls]}
                )
        return out

    def open(self, url, headers, body):
        request_id, key = CURRENT_REQUEST.get(), exchange_key(url, headers, body)
        trace = CURRENT_TRACE.get()
        entry = {
            "request_id": request_id,
            "key": key,
            "call": len(self.calls.get((request_id, key), [])),
            "stage": CURRENT_STAGE.get(),
            # The trace node (decision / generation / embedding) that made this request.
            "parent": CURRENT_PARENT.get(),
            # Its call site, which does not depend on scheduling (see RequestTrace.open).
            "site": trace.site_of(CURRENT_PARENT.get()) if trace is not None else None,
            "url": url,
            "headers": public_headers(headers),
            "body": body,
            "opened_at": datetime.now(UTC).isoformat(),
            "attempts": [],
        }
        self._index(entry)
        return entry


class RecordingTransport(JsonTransport):
    def __init__(self, client: httpx.AsyncClient, tape: Tape | None = None, **options):
        super().__init__(client, **options)
        self.tape = tape if tape is not None else Tape()

    def open_call(self, url, headers, body):
        return self.tape.open(url, headers, body)

    async def send(self, url, headers, encoded, context, *, call=None, attempt=0):
        started = time.monotonic()
        outcome = {
            "attempt": attempt,
            "attempt_kind": "original" if attempt == 0 else "transport_retry",
            "status": None,
            "retry_after": None,
            "data": None,
        }
        try:
            status, retry_after, data = await super().send(
                url, headers, encoded, context, call=call, attempt=attempt
            )
            outcome.update(status=status, retry_after=retry_after, data=data)
            return status, retry_after, data
        except (httpx.TimeoutException, TimeoutError):
            outcome["error"] = "timeout"
            raise
        except httpx.HTTPError:
            outcome["error"] = "network"
            raise
        except ContractError as exc:
            outcome["error"] = "contract:" + exc.code
            raise
        finally:
            outcome["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
            if call is not None:
                call["attempts"].append(outcome)


class ReplayTransport(JsonTransport):
    """Serves recorded exchanges. `jitter_seed` delays every completion by a small pseudo-random
    time so concurrent calls finish in a different order than they were recorded (N0-C)."""

    def __init__(self, tape: Tape, *, jitter_seed=None, max_jitter_s=0.004, **options):
        super().__init__(None, **options)
        self.tape, self.jitter_seed, self.max_jitter_s = tape, jitter_seed, max_jitter_s
        self.served, self.taken, self.site_fallbacks = {}, set(), 0

    def open_call(self, url, headers, body):
        """The recorded occurrence for this request: the one made from the same call site if the
        tape has sites, otherwise (or if that site's occurrences are used up) the next unused one
        in opening order. Never a live call, never a near match."""
        request_id, key = CURRENT_REQUEST.get(), exchange_key(url, headers, body)
        trace = CURRENT_TRACE.get()
        site = trace.site_of(CURRENT_PARENT.get()) if trace is not None else None
        for entry in self.tape.sites.get((request_id, site, key), []):
            if id(entry) not in self.taken:
                self.taken.add(id(entry))
                return entry
        for entry in self.tape.calls.get((request_id, key), []):
            if id(entry) not in self.taken:
                self.taken.add(id(entry))
                if entry.get("site"):
                    self.site_fallbacks += 1
                return entry
        recorded = len(self.tape.calls.get((request_id, key), []))
        raise ReplayMiss("replay_miss", detail={"recorded": recorded})

    async def send(self, url, headers, encoded, context, *, call=None, attempt=0):
        if call is None or attempt >= len(call["attempts"]):
            raise ReplayMiss("replay_attempt_missing", detail={"attempt": attempt})
        self.served[(call["request_id"], call["key"], call["call"])] = attempt + 1
        outcome = call["attempts"][attempt]
        # A live call always waits on the network: yield once so concurrent requests are in
        # flight together here too (without it a replayed answer can hit a run cache that the
        # live run could not have, N0 R1-replay).
        await asyncio.sleep(0)
        if self.jitter_seed is not None:
            seed = f"{self.jitter_seed}|{call['request_id']}|{call['key']}|{call['call']}|{attempt}"
            rng = random.Random(hashlib.sha256(seed.encode()).digest())
            await asyncio.sleep(rng.random() * self.max_jitter_s)
        error = outcome.get("error")
        if error == "timeout":
            raise TimeoutError()
        if error == "network":
            raise httpx.NetworkError("replayed_network_error")
        if error and error.startswith("contract:"):
            raise ContractError(error.split(":", 1)[1])
        return outcome["status"], outcome["retry_after"], outcome["data"]

    async def pause(self, seconds):
        return None

    def unused(self, request_id):
        """Recorded exchanges of a request that the replay never asked for (the code now makes
        fewer or different calls)."""
        return [
            {"key": e["key"], "call": e["call"], "stage": e["stage"]}
            for e in self.tape.of(request_id)
            if self.served.get((request_id, e["key"], e["call"]), 0) < len(e["attempts"])
        ]
