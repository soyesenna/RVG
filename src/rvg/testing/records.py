"""One request record per runtime request of an evaluation run (N0).

`answer_one` runs `UniversalEngine.answer` exactly as the product does, tagging every provider
exchange with the request id; the record keeps the engine's result, the call usage and, when the
run records, the request's tape. `manifest` reads the record into the fields an audit needs
(policy hash, knowledge snapshot, retrieval ranks and scores, the evidence the writer saw, raw
writer and Jev responses, the gate's decisions, the status reason and the citations).
A result that releases nothing shows no draft value (0.1.0, `engine.withhold_drafts`); the
values removed are kept beside it as the record's `held_draft`, for evaluations that read a held
draft. `signature` and `numbers` are what two runs of the same request are compared on: every
decision field exactly, and the pre-registered floating-point fields (retrieval scores, signal
scores) with a tolerance."""

from __future__ import annotations

import json
import math
import time

from ..temporal.models import TemporalQuestion
from ..universal import universal_hash
from ..universal.trace import (
    CURRENT_REPLAY,
    CURRENT_REQUEST,
    CURRENT_TRACE,
    HELD_DRAFTS,
    RequestTrace,
    canonical_hash,
)

RESULT_KEYS = (
    "status",
    "reason",
    "abstain_reason",
    "error",
    "stage",
    "short_answer",
    "text",
    "sentences",
    "missing",
    "action",
    "evidence_ids",
    "scope",
    "assessment",
    "assessment_before_recovery",
    "trace",
    "audit",
    "snapshot",
    "config_hash",
    "as_of",
    "known_at",
    "arm",
    "release",
    "blocked_documents",
    "retried_after_change",
    "storage",
    "conflict_signal",
    "answer_kind",
    "conflict_rewrite",
    "conflict_rewrite_of",
)
USAGE_KEYS = (
    "decision_calls",
    "generation_calls",
    "embedding_calls",
    "http_attempts",
    "reported_cost_usd",
    "unknown_cost_calls",
)
STATUSES = ("FULL", "PARTIAL", "ABSTAIN", "UNRESOLVED_SCOPE", "ERROR")
# The only floating-point fields a replay may reproduce within a tolerance instead of exactly.
NUMERIC = ("assessment", "assessment_before_recovery", "audit", "scope")


def usage(context):
    return {key: getattr(context, key) for key in USAGE_KEYS}


# Engine questions carry only these fields of an evaluation example; references, gold ids,
# expected statuses and dataset names never reach a runtime request.
RUNTIME_FIELDS = ("question", "locale", "as_of", "known_at")
# Results whose outcome depended on the wall clock (a CallContext deadline, not a recorded
# transport timeout): a replay with a frozen clock cannot re-decide them.
CLOCK_REASONS = {"deadline_exceeded", "retry_wait_exceeds_deadline"}


def runtime_question(example):
    """The engine question of an evaluation example, built from `RUNTIME_FIELDS` only."""
    return TemporalQuestion(
        text=example["question"],
        as_of=example["as_of"],
        known_at=example.get("known_at") or example["as_of"],
        locale=example["locale"],
    )


async def answer_one(
    engine,
    tenant,
    question,
    *,
    request_id,
    context,
    tape=None,
    scope=None,
    run_id="run",
    oracle=None,
):
    """The product call for one runtime request. `scope` records how the evaluation restricted
    the searchable documents (e.g. a gold-removed stress condition); it is evaluation data and
    reaches the engine only through the retriever the caller built."""
    trace = RequestTrace(run_id=run_id, request_id=request_id, policy_hash=universal_hash())
    held = []
    tokens = (
        CURRENT_REQUEST.set(request_id),
        CURRENT_TRACE.set(trace),
        CURRENT_REPLAY.set(oracle),  # a replay's record of timing-dependent decisions
        HELD_DRAFTS.set(held),  # 0.1.0: the draft values a held result does not show
    )
    started = time.monotonic()
    try:
        found = await engine.answer(tenant, question, context)
    finally:
        HELD_DRAFTS.reset(tokens[3])
        CURRENT_REPLAY.reset(tokens[2])
        CURRENT_TRACE.reset(tokens[1])
        CURRENT_REQUEST.reset(tokens[0])
    result = {key: found.get(key) for key in RESULT_KEYS}
    if "handoff" in found:  # PROD-1: the opt-in handoff packet, only when the engine gave one
        result["handoff"] = found["handoff"]
    record = {
        "request_id": request_id,
        "run_id": run_id,
        "tenant": tenant,
        "question": question.model_dump(mode="json"),
        "scope": scope,
        "result": result,
        "usage": usage(context),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "clock_dependent": bool(
            {result.get("reason"), (result.get("error") or {}).get("code")} & CLOCK_REASONS
        ),
        "trace_nodes": trace.export(),
    }
    if held:  # 0.1.0: evaluation data beside the product result, never inside it
        record["held_draft"] = held[-1]
    if tape is not None:
        record["tape"] = tape.of(request_id)
        record["ambiguous_occurrences"] = tape.ambiguous(request_id)
    return record


def node_multiset(record):
    """Trace nodes as a comparable multiset: (operation, input, status, output, cache hit)."""
    return sorted(
        (
            n["operation"],
            n["input_hash"],
            n["status"],
            n.get("output_hash") or "",
            bool(n.get("cached")),
        )
        for n in record.get("trace_nodes") or []
    )


def _payload(entry):
    try:
        return json.loads(entry["body"]["messages"][-1]["content"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _final(entry):
    attempts = entry.get("attempts") or []
    return attempts[-1] if attempts else {}


def _schema(entry):
    return ((entry["body"].get("response_format") or {}).get("json_schema") or {}).get("name")


def exchanges(tape):
    """The tape of one request split by provider role, in the order the calls were opened."""
    out = {"writer": [], "extractor": [], "decision": [], "embedding": []}
    for entry in tape or []:
        final = _final(entry)
        data = final.get("data") or {}
        base = {
            "key": entry["key"],
            "call": entry["call"],
            "attempts": len(entry.get("attempts") or []),
            "http_status": final.get("status"),
            "error": final.get("error"),
            "cost_usd": (data.get("usage") or {}).get("cost"),
        }
        url = entry["url"]
        if url.endswith("/chat/completions"):
            payload = _payload(entry) or {}
            choice = (data.get("choices") or [{}])[0] if data else {}
            item = base | {
                "schema": _schema(entry),
                "system_sha256": canonical_hash(entry["body"]["messages"][0]["content"]),
                "raw": (choice.get("message") or {}).get("content"),
                "finish_reason": choice.get("finish_reason"),
                "model": data.get("model"),
            }
            if _schema(entry) == "temporal_answer_v1":
                evidence = payload.get("evidence") or []
                item["evidence"] = [
                    {"id": e.get("id"), "text_sha256": canonical_hash(e.get("text"))}
                    for e in evidence
                ]
                out["writer"].append(item)
            else:
                out["extractor"].append(item)
        elif url.endswith("/embeddings"):
            texts = entry["body"].get("input")
            texts = texts if isinstance(texts, list) else [texts]
            out["embedding"].append(base | {"inputs_sha256": [canonical_hash(t) for t in texts]})
        else:
            questions = entry["body"].get("questions") or {}
            out["decision"].append(
                base
                | {
                    "stage": entry.get("stage"),
                    "questions": sorted(questions),
                    "answers": data.get("answers"),
                    "model": data.get("model"),
                }
            )
    return out


def manifest(record, *, store_digest=None):
    """The audit fields of one request record."""
    result = record["result"]
    retrieval = [
        {
            "query": call.get("query"),
            "ranking": (call.get("retrieval") or {}).get("ranking"),
            "documents": (call.get("retrieval") or {}).get("documents"),
            "pending": (call.get("retrieval") or {}).get("pending"),
            "gate": {
                "active": sorted(
                    unit
                    for unit, status in ((call.get("gate") or {}).get("statuses") or {}).items()
                    if status == "ELIGIBLE"
                ),
                "reasons": (call.get("gate") or {}).get("reasons"),
                "conflicts": (call.get("gate") or {}).get("conflicts"),
                "unresolved": (call.get("gate") or {}).get("unresolved"),
                "complete": (call.get("gate") or {}).get("complete_within_local_snapshot"),
            },
        }
        for call in result.get("audit") or []
    ]
    return {
        "request_id": record["request_id"],
        "config_hash": result.get("config_hash"),
        "knowledge": {
            "tenant": record["tenant"],
            "event_seq": result.get("snapshot"),
            "store_digest": store_digest,
        },
        "question": record["question"],
        "scope": record.get("scope"),
        "retrieval": retrieval,
        "stages": result.get("trace"),
        "exchanges": exchanges(record.get("tape")),
        "decision": {
            "status": result.get("status"),
            "reason": result.get("reason"),
            "stage": result.get("stage"),
            "action": result.get("action"),
            "assessment": result.get("assessment"),
            "assessment_before_recovery": result.get("assessment_before_recovery"),
        },
        "answer": {
            "short_answer": result.get("short_answer"),
            "text": result.get("text"),
            "citations": [s.get("evidence_ids") for s in result.get("sentences") or []],
            "missing": result.get("missing"),
        },
        "usage": record.get("usage"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "trace_nodes": record.get("trace_nodes"),
        "clock_dependent": record.get("clock_dependent"),
        "ambiguous_occurrences": record.get("ambiguous_occurrences"),
    }


def _strip_numbers(value):
    """The decision skeleton: every float replaced by a marker (floats are compared apart)."""
    if isinstance(value, float):
        return "<float>"
    if isinstance(value, dict):
        return {k: _strip_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_numbers(v) for v in value]
    return value


def signature(record):
    """Everything a replay must reproduce exactly: public status and its reason, the stage and
    recovery action, the answer and its citations, the evidence of every stage, the retrieval
    order and the gate's decisions. Floats are left to `numbers`."""
    result = record["result"]
    return _strip_numbers(
        {
            key: result.get(key)
            for key in (
                "status",
                "reason",
                "abstain_reason",
                "error",
                "stage",
                "action",
                "short_answer",
                "text",
                "sentences",
                "missing",
                "evidence_ids",
                "scope",
                "trace",
                "assessment",
                "assessment_before_recovery",
                "audit",
            )
        }
    )


def numbers(value, path=""):
    """(path, float) pairs of the pre-registered numeric fields."""
    if isinstance(value, float):
        yield path, value
    elif isinstance(value, dict):
        for k in sorted(value):
            yield from numbers(value[k], f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from numbers(v, f"{path}[{i}]")


def contract_view(record):
    """(the record as the 0.1.0 public result contract shows it, the draft values that rule
    withholds — None when it withholds nothing). A recording older than 0.1.0 kept a held draft's
    values in its result; read through this view it compares field by field with a newer record,
    and the values withheld compare with the newer record's `held_draft` (parent PUB-2)."""
    from ..universal.engine import withhold_drafts

    if record["result"].get("status") in ("FULL", "PARTIAL"):
        return record, None
    held = []
    token = HELD_DRAFTS.set(held)
    try:
        result = withhold_drafts(record["result"])
    finally:
        HELD_DRAFTS.reset(token)
    return {**record, "result": result}, (held[-1] if held else None)


def compare(a, b, *, tolerance=1e-12):
    """Compare two records of the same request: (equal signature, max |float difference|,
    names of the top-level fields that differ)."""
    sa, sb = signature(a), signature(b)
    fields = sorted(k for k in sa if sa.get(k) != sb.get(k))
    na = dict(numbers({k: a["result"].get(k) for k in NUMERIC}))
    nb = dict(numbers({k: b["result"].get(k) for k in NUMERIC}))
    if set(na) != set(nb):
        worst = math.inf
    else:
        worst = max((abs(na[p] - nb[p]) for p in na), default=0.0)
    nodes_equal = node_multiset(a) == node_multiset(b)
    return {
        "equal": not fields and worst <= tolerance and nodes_equal,
        "fields": fields,
        "max_float_difference": worst,
        "trace_nodes_equal": nodes_equal,
    }
