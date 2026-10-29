"""Canonical hashing and the request-scoped trace of an answer (N0).

A request is hashed from its canonical JSON bytes: map keys sorted, list order kept (the order of
evidence is part of what a model was shown), compact separators, no NaN. Hashing never changes
what a model receives.

Context variables carry, into every task an answer spawns, the runtime request id, the decision
stage that asked a Jev question, the enclosing trace node (the parent of whatever is opened
next) and the request's `RequestTrace`. Product code only *emits* nodes; with no active trace
(the product outside an evaluation) emitting is a no-op, so tracing never changes a decision."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from contextvars import ContextVar
from datetime import UTC, datetime

CURRENT_REQUEST: ContextVar[str | None] = ContextVar("deepjev_request_id", default=None)
CURRENT_STAGE: ContextVar[str | None] = ContextVar("deepjev_decision_stage", default=None)
CURRENT_PARENT: ContextVar[str | None] = ContextVar("deepjev_trace_parent", default=None)
# Bumped whenever the structured-output parser's acceptance rules change.
PARSER_VERSION = "structured-json-strict-1"


def canonical_bytes(value) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def canonical_hash(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _now():
    return datetime.now(UTC).isoformat()


class RequestTrace:
    """Append-only nodes of one runtime request (the `trace_node` schema plus the operation's
    own fields). Node ids are per-request counters: they differ between a live run and its replay
    and are never compared; nodes are compared by (operation, input_hash, status, output_hash)."""

    def __init__(self, *, run_id, request_id, policy_hash, attempt_kind="original"):
        self.run_id, self.request_id, self.policy_hash = run_id, request_id, policy_hash
        self.attempt_kind = attempt_kind
        self.nodes, self._ids = [], itertools.count()
        self._by_id, self._siblings = {}, {}

    def site_of(self, node_id):
        node = self._by_id.get(node_id)
        return node["site"] if node else None

    def open(self, operation, input_value, *, attempt_kind=None, **fields):
        parent = CURRENT_PARENT.get()
        input_hash = canonical_hash(input_value)
        # A call site that does not depend on scheduling: the parent's site, the operation and
        # its input, and the rank among earlier siblings with that same operation and input.
        base = (self._by_id.get(parent) or {}).get("site", "")
        twin = (parent, operation, input_hash)
        rank = self._siblings.get(twin, 0)
        self._siblings[twin] = rank + 1
        node = {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "node_id": f"n{next(self._ids)}",
            "parent_ids": [parent] if parent else [],
            "site": f"{base}/{operation}:{input_hash[:12]}#{rank}",
            "operation": operation,
            "attempt_kind": attempt_kind or self.attempt_kind,
            "policy_hash": self.policy_hash,
            "input_hash": input_hash,
            "output_hash": None,
            "started_at": _now(),
            "ended_at": None,
            "status": "OK",
            "_t0": time.monotonic(),
            **fields,
        }
        self.nodes.append(node)
        self._by_id[node["node_id"]] = node
        return node

    def close(self, node, *, status="OK", output_value=None, **fields):
        node["status"] = status
        if output_value is not None:
            node["output_hash"] = canonical_hash(output_value)
        node["ended_at"] = _now()
        node["duration_ms"] = round((time.monotonic() - node.pop("_t0")) * 1000, 2)
        node.update(fields)

    def export(self):
        return [{k: v for k, v in n.items() if k != "_t0"} for n in self.nodes]


CURRENT_TRACE: ContextVar[RequestTrace | None] = ContextVar("deepjev_request_trace", default=None)


class ReplayOracle:
    """What a live run decided on its own timing, for a replay to decide the same: whether a
    decision call was served from the run cache. Only a replay sets one."""

    def __init__(self, nodes):
        self.cached = {n["site"]: bool(n.get("cached")) for n in nodes if "site" in n}

    def cache_hit(self, site):
        return self.cached.get(site)


CURRENT_REPLAY: ContextVar[ReplayOracle | None] = ContextVar("deepjev_replay_oracle", default=None)
# 0.1.0 (parent PUB-2): where an evaluation collects the draft values a result that releases
# nothing does not show (`withhold_drafts`). Only `rvg.testing.records.answer_one` sets one.
HELD_DRAFTS: ContextVar[list | None] = ContextVar("rvg_held_drafts", default=None)


def status_of(exc):
    """The trace status of a failed operation."""
    code = getattr(exc, "code", "")
    if code in {"ambiguous_timeout", "deadline_exceeded", "retry_wait_exceeds_deadline"}:
        return "TIMEOUT"
    if code.startswith("generation_schema") or code in {"probability_sum_mismatch"}:
        return "SCHEMA_ERROR"
    return "ERROR"


class node:
    """`with node(op, input, **fields) as n:` opens a node under the current parent, makes it the
    parent of everything inside, and closes it OK or with the failure's status. A no-op without an
    active trace."""

    def __init__(self, operation, input_value, **fields):
        self.operation, self.input_value, self.fields = operation, input_value, fields
        self.trace, self.value, self.token = CURRENT_TRACE.get(), None, None

    def __enter__(self):
        if self.trace is not None:
            self.value = self.trace.open(self.operation, self.input_value, **self.fields)
            self.token = CURRENT_PARENT.set(self.value["node_id"])
        return self

    @property
    def site(self):
        return self.value["site"] if self.value is not None else None

    def set(self, **fields):
        if self.value is not None:
            self.value.update(fields)

    def output(self, value):
        """The node's output (hashed at close). A callable is only called while tracing, so
        building an observation never runs outside an evaluation."""
        if self.value is not None:
            self.value["_output"] = value() if callable(value) else value

    def __exit__(self, kind, exc, tb):
        if self.trace is None:
            return False
        CURRENT_PARENT.reset(self.token)
        out = self.value.pop("_output", None)
        if exc is None:
            self.trace.close(self.value, output_value=out)
        else:
            self.trace.close(
                self.value,
                status=status_of(exc),
                error={"type": type(exc).__name__, "code": getattr(exc, "code", str(exc)[:80])},
            )
        return False
