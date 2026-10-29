"""`universal-query`: the product path over a time-scoped store, one policy for every store;
`ingest`: documents, their access and source events from a JSON Lines file into that store, so a
knowledge base can be built and asked from the command line alone (0.1.0, parent PUB-2). The
command line's caller is trusted to name the principal (`--principal`, `--group`)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from ..factory import decision_provider, embedding_provider, generation_provider
from ..runtime import CallContext
from ..storage import open_storage
from ..temporal.access import TrustedPrincipal
from ..temporal.models import TemporalQuestion
from ..temporal.store import TemporalRetriever, TemporalStore
from .config import universal, warn_dimensions
from .engine import UniversalEngine

EVENT_TYPES = ("revision", "revoke", "access")


class InputError(ValueError):
    """A command-line input the command refuses; its message quotes no input content."""


SHOWN_REJECTIONS = 20


def _store_arguments(command):
    command.add_argument(
        "--store",
        type=Path,
        default=Path("data/temporal.sqlite3"),
        help="local SQLite store (default: data/temporal.sqlite3 under --root)",
    )
    command.add_argument(
        "--storage",
        type=Path,
        help="deployment storage configuration (storage.json: SQLite or PostgreSQL state, vector "
        "index); overrides --store",
    )
    command.add_argument("--tenant", default="demo", help="tenant (default: demo)")


def add_commands(commands):
    query = commands.add_parser(
        "universal-query",
        help="answer a question: retrieve, check validity, scope, write, control (one policy)",
    )
    query.add_argument("text", help="the question")
    query.add_argument("--as-of", help="ISO 8601 time the answer must hold at (default: now)")
    query.add_argument("--known-at", help="ISO 8601 knowledge cutoff (default: --as-of)")
    query.add_argument(
        "--locale", default="en-US", help="answer language (BCP-47); it does not select a policy"
    )
    _store_arguments(query)
    query.add_argument(
        "--principal",
        help="the asker's principal id; documents with an access list are seen only by the "
        "principals and groups it allows (default: no principal)",
    )
    query.add_argument(
        "--group",
        action="append",
        help="a group of the principal (repeatable; needs --principal)",
    )
    query.add_argument(
        "--allowed-document-id",
        action="append",
        help="restrict retrieval to this document id (repeatable)",
    )
    query.add_argument(
        "--include-audit", action="store_true", help="keep the per-search audit in the output"
    )
    query.add_argument(
        "--handoff-packet",
        action="store_true",
        help="attach the handoff packet (result.handoff) to a result that is not released",
    )
    query.add_argument("--out", type=Path, help="also write the JSON output to this file")


def add_ingest_command(commands):
    """`ingest` (0.1.0) beside `universal-query`; separate, since a host command line (e.g. a
    research harness) may have its own `ingest`."""
    ingest = commands.add_parser(
        "ingest",
        help="store documents, their access lists and source events from a JSON Lines file",
        description="Each line is one JSON object: a document (the TemporalDocument fields), "
        '{"document": {...}, "access": {"allow": [...], "deny": [...]}}, or an event '
        '{"type": "revision" | "revoke" | "access", ...} as RuntimeSession.apply_event takes it. '
        "Lines are applied in order; then every stored revision without a vector is embedded.",
    )
    ingest.add_argument("file", type=Path, help="the JSON Lines file")
    _store_arguments(ingest)
    ingest.add_argument(
        "--no-vectors", action="store_true", help="store only; embed later (pending revisions)"
    )
    ingest.add_argument(
        "--deadline-seconds",
        type=float,
        default=3600,
        help="time limit of the whole ingest (default: 3600)",
    )
    ingest.add_argument("--out", type=Path, help="also write the JSON output to this file")


def principal_of(args):
    if args.group and not args.principal:
        raise InputError("--group needs --principal")
    if not args.principal:
        return None
    return TrustedPrincipal(args.principal, groups=tuple(args.group or ()))


def open_store(args, settings):
    if args.storage is not None:
        return open_storage(settings.resolve_path(args.storage), root=settings.project_root)
    return TemporalStore(settings.resolve_path(args.store))


async def execute(args, settings, transport):
    if args.command == "ingest":
        return await ingest(args, settings, transport)
    principal = principal_of(args)
    store = open_store(args, settings)
    try:
        embedding = (
            embedding_provider(settings, transport)
            if universal()["retrieval"]["strategy"] != "lexical"
            else None
        )
        warn_dimensions(embedding)
        retriever = TemporalRetriever(
            store, embedding, allowed_ids=args.allowed_document_id, principal=principal
        )
        as_of = args.as_of or datetime.now(UTC)
        question = TemporalQuestion(
            text=args.text, as_of=as_of, known_at=args.known_at or as_of, locale=args.locale
        )
        engine = UniversalEngine(
            retriever=retriever,
            decision=decision_provider(settings, transport),
            writer=generation_provider(settings, transport, role="surface"),
            extractor=generation_provider(settings, transport, role="semantic"),
            handoff_packet=getattr(args, "handoff_packet", False),
        )
        context = settings.context()
        result = await engine.answer(args.tenant, question, context)
        result.update(
            decision_calls=context.decision_calls,
            generation_calls=context.generation_calls,
            embedding_calls=context.embedding_calls,
            reported_cost_usd=context.reported_cost_usd,
        )
        if not args.include_audit:
            result.pop("audit", None)
        return result
    finally:
        store.close()


def read_lines(path):
    """(line number, object) of every non-empty line; a line that is not a JSON object is an
    error naming its number only (never its content)."""
    out = []
    with open(path, encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                raise InputError(f"line {number}: not JSON") from None
            if not isinstance(value, dict):
                raise InputError(f"line {number}: not a JSON object")
            out.append((number, value))
    return out


async def ingest(args, settings, transport):
    from ..session import OperationError, RequestContext, RuntimeSession

    lines = read_lines(settings.resolve_path(args.file))
    store = open_store(args, settings)
    try:
        session = RuntimeSession.open(store, settings, transport=transport)
        call = CallContext.with_timeout(
            args.deadline_seconds,
            max_embedding_calls=10**9,
            max_http_attempts=10**9,
            max_decision_calls=0,
            max_generation_calls=0,
        )
        context = RequestContext(tenant=args.tenant, call=call)
        documents, events, rejected = Counter(), Counter(), []
        batch = []

        async def flush():
            if not batch:
                return
            done = await session.ingest(
                [value for _, value in batch], context, prepare_vectors=False
            )
            for (number, _), row in zip(batch, done["results"], strict=True):
                documents[row["outcome"]] += 1
                if row["outcome"] in ("REJECTED", "ERROR"):
                    rejected.append({"line": number, **row})
            batch.clear()

        for number, value in lines:
            if value.get("type") in EVENT_TYPES:
                await flush()
                try:
                    out = await session.apply_event(value, context, prepare_vectors=False)
                    events[out.get("outcome") or "APPLIED"] += 1
                except OperationError as exc:
                    events["REJECTED"] += 1
                    rejected.append({"line": number, "outcome": "REJECTED", "reason": exc.code})
            else:
                batch.append((number, value))
        await flush()
        vectors = None
        if not args.no_vectors:
            prepared = await session.ingest([], context, prepare_vectors=True)
            vectors = prepared["vectors"]
        await session.aclose()
        return {
            "operation": "ingest",
            "tenant": args.tenant,
            "lines": len(lines),
            "documents": dict(documents),
            "events": dict(events),
            "rejected": rejected[:SHOWN_REJECTIONS],
            "rejected_total": len(rejected),
            "vectors": vectors,
            "knowledge_seq": getattr(store, "state", store).snapshot(args.tenant),
            "embedding_calls": call.embedding_calls,
            "reported_cost_usd": call.reported_cost_usd,
        }
    finally:
        store.close()
