"""A runtime session: the public way to run RVG as one system (SEP-04).

`open_session` assembles what an application or a benchmark adapter needs from the existing
objects — a deployment (`open_storage`), the providers (`decision_provider`,
`generation_provider`, `embedding_provider`), `TemporalRetriever` and `UniversalEngine` — and
exposes neutral operations over them:

- `ingest(documents, context)` and `apply_event(event, context)`: commit revisions, revocations and
  whole-document ACLs to the knowledge state, and say what is actually known about readiness
  (the commit, the vectors, what the next query's index view contains);
- `answer(request, context)`: `UniversalEngine.answer`, called once; its result is returned
  unchanged inside an envelope that adds the request id and the call counters;
- `health(level, context)`: `liveness` checks local state only; `readiness` adds short read-only
  backend reads. Neither calls a model, migrates, warms an index or writes;
- `export_artifact_lineage(reference, context)`: the recorded released evidence of an answer given
  by this session, filtered by the permissions and revocations in force now;
- `capabilities()`, `describe_runtime()` and `aclose()`.

Nothing here decides an answer, re-ranks, re-verifies, rewrites or falls back to another provider.
A principal is a `TrustedPrincipal` from the caller's authentication context, never a string from
a request. Failures of an operation are `OperationError`; an answer that fails inside the engine is
the engine's own `ERROR` status."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata as md
import json
import re
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from . import __version__
from .config import Settings, load_settings
from .contracts import Contract
from .errors import ConfigurationError, ContractError, ProviderError
from .factory import decision_provider, embedding_provider, generation_provider
from .prompts import manifest_sha256, verify_all
from .runtime import CallContext
from .storage import Storage, as_storage, open_storage
from .storage.identity import input_hash
from .storage.segments import segment_revision
from .temporal.access import DENY_ALL, RESERVED_PREFIX, TrustedPrincipal
from .temporal.models import TemporalDocument, TemporalQuestion
from .temporal.retriever import TemporalRetriever
from .transport import JsonTransport
from .universal import universal, universal_hash
from .universal.config import warn_dimensions
from .universal.engine import UniversalEngine

API_VERSION = "rvg-runtime-api/0.1"
OBSERVATION_VERSION = "rvg-observation/0.1"
READINESS_TIMEOUT_SECONDS = 5.0
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
_UNIT_ID = re.compile(r"^(?P<document>[\w-]+)@(?P<revision>\d+)#(?P<unit>[\w-]+)$")


class OperationError(RuntimeError):
    """An operation that could not be done, as a code a caller can act on. `code` is one of
    UNSUPPORTED_CAPABILITY, INVALID_INPUT, FORBIDDEN_INPUT, NOT_READY, BACKEND_ERROR,
    PROVIDER_ERROR, TIMEOUT, INTERNAL_ERROR; `native_code` says which rule or failure it was.
    Messages never carry request text, document text, credentials or connection strings."""

    def __init__(self, operation, code, native_code=None, *, retryable=None):
        self.operation, self.code, self.native_code = operation, code, native_code
        self.retryable = retryable
        super().__init__(f"{operation}: {code}" + (f" ({native_code})" if native_code else ""))

    def to_dict(self):
        return {
            "api_version": API_VERSION,
            "operation": self.operation,
            "error": {
                "code": self.code,
                "native_code": self.native_code,
                "retryable": self.retryable,
            },
        }


@dataclass(frozen=True)
class RequestContext:
    """Who asks and in which tenant, from the caller's trusted context. `selected_document_ids`
    is a real user's own document selection (it only narrows what may be retrieved); `call` is the
    request's budget and deadline (default: the session settings' budget)."""

    tenant: str
    principal: TrustedPrincipal | None = None
    request_id: str | None = None
    selected_document_ids: tuple[str, ...] | None = None
    call: CallContext | None = None
    # RVI-07A: the source system that issued an event, from the trusted context (never from the
    # event body); with the tenant and the event id it keys the event-id ledger. Default: the
    # principal's id, else "default".
    issuer: str | None = None


class DocumentAccess(Contract):
    """A document's whole ACL: `allow` names principals or groups that may see it (None: no allow
    restriction; an empty tuple: nobody — explicit deny-all), `deny` names those that may not
    (deny wins). Applies to every revision of the document id."""

    allow: tuple[str, ...] | None = None
    deny: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _no_reserved_names(self):
        if any(n.startswith(RESERVED_PREFIX) for n in (*(self.allow or ()), *self.deny)):
            raise ValueError("reserved_principal_name")
        return self

    def entries(self):
        if self.allow is not None and not self.allow:
            out = {DENY_ALL: "allow"}  # RVI-07A: an allow set no principal can match
        else:
            out = {name: "allow" for name in self.allow or ()}
        out.update({name: "deny" for name in self.deny})
        return out


UNCHANGED = "unchanged"


class IngestItem(Contract):
    """A revision to store, and what to do with its document's ACL first: leave it (`unchanged`),
    clear it (None: visible to the tenant) or replace it."""

    document: TemporalDocument
    access: DocumentAccess | None | Literal["unchanged"] = UNCHANGED


class RevisionEvent(Contract):
    type: Literal["revision"] = "revision"
    event_id: str | None = None
    document: TemporalDocument
    access: DocumentAccess | None | Literal["unchanged"] = UNCHANGED


class RevokeEvent(Contract):
    type: Literal["revoke"] = "revoke"
    event_id: str | None = None
    document_id: str = Field(min_length=1, max_length=160, pattern=r"^[\w-]+$")


class AccessEvent(Contract):
    type: Literal["access"] = "access"
    event_id: str | None = None
    document_id: str = Field(min_length=1, max_length=160, pattern=r"^[\w-]+$")
    access: DocumentAccess | None


class _Event(Contract):
    event: Annotated[RevisionEvent | RevokeEvent | AccessEvent, Field(discriminator="type")]


def _now():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _readiness(state, reason, observed_at=None):
    return {"state": state, "observed_at": observed_at, "reason": reason}


def _validation_code(exc: ValidationError):
    """The first failed rule of a validation error, without the input value."""
    error = exc.errors(include_input=False)[0]
    where = ".".join(str(p) for p in error["loc"])
    if error["type"] == "value_error":
        rule = str(error.get("ctx", {}).get("error", "value_error"))
        return f"{where}:{rule}" if where else rule
    return f"{where}:{error['type']}" if where else error["type"]


def _is_forbidden(exc: ValidationError):
    return any(e["type"] == "extra_forbidden" for e in exc.errors(include_input=False))


def _sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def _backend_errors():
    """Failures a state store or vector service can raise: SQLite, PostgreSQL (when installed),
    HTTP to a vector service, the runtime's own contract errors and OS errors."""
    import sqlite3

    import httpx

    errors = [sqlite3.Error, httpx.HTTPError, ProviderError, OSError, RuntimeError, KeyError]
    try:
        import psycopg

        errors.append(psycopg.Error)
    except ImportError:
        pass
    return tuple(errors)


def _schemas_sha256():
    root = files("rvg") / "resources" / "schemas"
    listing = {}

    def walk(node, prefix=""):
        for child in sorted(node.iterdir(), key=lambda c: c.name):
            if child.is_dir():
                walk(child, prefix + child.name + "/")
            elif child.name.endswith(".json"):
                listing[prefix + child.name] = _sha(child.read_bytes())

    walk(root)
    return _sha(json.dumps(listing, sort_keys=True)), listing


def _provider(value, role):
    if value is None:
        return {"role": role, "kind": None, "model_id": None, "network": "NONE"}
    kind = type(value).__name__
    model = getattr(value, "model", None)
    primary = getattr(value, "primary", None)
    if model is None and primary is not None:
        model = getattr(primary, "model", None)
    return {
        "role": role,
        "kind": kind,
        "model_id": model,
        "network": "NONE" if kind.startswith("Mock") else "EXTERNAL_API",
    }


class RuntimeSession:
    """Owns only what it opened: the storage it opened from a configuration and the HTTP client it
    created. Storage objects and providers handed in stay the caller's to close."""

    def __init__(
        self,
        storage,
        *,
        decision,
        writer,
        extractor,
        embedding=None,
        settings: Settings | None = None,
        owns_storage=False,
        client=None,
        transport_origin="caller",
        max_documents=24,
        lineage_capacity=1024,
        handoff_packet=False,
        decision_cache=None,
    ):
        """`handoff_packet` (PROD-1, opt-in, off by default): every answer of this session that
        is not released carries `result.handoff` (`rvg.universal.handoff`). `decision_cache`
        (PROD-2, opt-in, off by default): a `rvg.universal.cache.CacheMode` every answer of this
        session looks up and/or records."""
        self.storage = as_storage(storage)
        self.handoff_packet = bool(handoff_packet)
        self.decision_cache = decision_cache
        self.decision, self.writer, self.extractor = decision, writer, extractor
        self.embedding = embedding
        self.settings = settings
        self.max_documents = max_documents
        self._owns_storage, self._client = owns_storage, client
        self._transport_origin = transport_origin
        self._lineage: OrderedDict[str, dict] = OrderedDict()
        self._lineage_capacity = lineage_capacity
        self.closed = False

    # --- lifecycle -----------------------------------------------------------------------------
    @classmethod
    def open(
        cls,
        storage,
        settings: Settings | None = None,
        *,
        root=None,
        environ=None,
        transport: JsonTransport | None = None,
        providers: Mapping | None = None,
        max_documents=24,
        handoff_packet=False,
        decision_cache=None,
    ):
        """Open a session. `storage` is an open `Storage` (or a store that carries one) or a
        storage configuration (`storage.json` path, dict or `StorageConfig`), which the session
        then opens and owns. Providers come from `providers` (caller-built, e.g. record/replay)
        or from `settings` through the product factories, over `transport` or over an HTTP
        client the session creates and owns. A library caller passes `settings`; without it the
        session calls `load_settings(root or the current directory)`, which also reads that
        directory's `.env` — a convenience for standalone use."""
        settings = settings or load_settings(Path(root) if root else Path.cwd())
        owned = not (isinstance(storage, Storage) or hasattr(storage, "storage"))
        opened = open_storage(storage, root=root, environ=environ) if owned else as_storage(storage)
        client, origin = None, "caller"
        try:
            if providers is None:
                if transport is None:
                    import httpx

                    client = httpx.AsyncClient(trust_env=False)
                    transport = JsonTransport(
                        client,
                        timeout=settings.http_timeout_seconds,
                        max_attempts=settings.http_max_attempts,
                    )
                    origin = "session"
                lexical = universal()["retrieval"]["strategy"] == "lexical"
                providers = {
                    "decision": decision_provider(settings, transport),
                    "writer": generation_provider(settings, transport, role="surface"),
                    "extractor": generation_provider(settings, transport, role="semantic"),
                    "embedding": None if lexical else embedding_provider(settings, transport),
                }
            session = cls(
                opened,
                decision=providers["decision"],
                writer=providers["writer"],
                extractor=providers["extractor"],
                embedding=providers.get("embedding"),
                settings=settings,
                owns_storage=owned,
                client=client,
                transport_origin=origin,
                max_documents=max_documents,
                handoff_packet=handoff_packet,
                decision_cache=decision_cache,
            )
            warn_dimensions(session.embedding)
            return session
        except BaseException:
            if owned:
                opened.close()
            # A client that sent nothing holds no connection; it is dropped with the failure.
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        """Close what this session opened; idempotent. Later operations fail NOT_READY."""
        if self.closed:
            return
        self.closed = True
        try:
            if self._owns_storage:
                await asyncio.to_thread(self.storage.close)
        finally:
            if self._client is not None:
                await self._client.aclose()

    def _check(self, operation):
        if self.closed:
            raise OperationError(operation, "NOT_READY", "session_closed")

    def new_call(self, deadline_seconds=None) -> CallContext:
        """A request budget from the session settings (`Settings.context`), optionally with
        another deadline; the limits are the settings' own."""
        if self.settings is None:
            raise OperationError("answer", "INVALID_INPUT", "call_context_required")
        call = self.settings.context()
        if deadline_seconds is not None:
            call.deadline = call.clock() + deadline_seconds
        return call

    def _context(self, operation, context):
        if not isinstance(context, RequestContext):
            raise OperationError(operation, "INVALID_INPUT", "request_context_required")
        if not isinstance(context.tenant, str) or not context.tenant.strip():
            raise OperationError(operation, "INVALID_INPUT", "tenant_required")
        if context.principal is not None and not isinstance(context.principal, TrustedPrincipal):
            raise OperationError(
                operation, "FORBIDDEN_INPUT", "principal_must_be_trusted_principal"
            )
        if context.request_id is not None and not _REQUEST_ID.match(context.request_id):
            raise OperationError(operation, "INVALID_INPUT", "request_id_format")
        if context.issuer is not None and (
            not isinstance(context.issuer, str) or not _REQUEST_ID.match(context.issuer)
        ):
            raise OperationError(operation, "INVALID_INPUT", "issuer_format")
        return context

    # --- knowledge -----------------------------------------------------------------------------
    async def ingest(self, documents: Iterable, context: RequestContext, *, prepare_vectors=True):
        """Store revisions (each item — its ACL, when given, and its revision — in one
        transaction: RVI-07A) and, when asked, embed every revision of the tenant that has no
        vector yet."""
        self._check("ingest")
        context = self._context("ingest", context)
        return await self._ingest(documents, context, prepare_vectors)

    async def _ingest(self, documents, context, prepare_vectors, ledger=None):
        results, stored = [], []
        for raw in documents:
            try:
                item = self._item(raw)
            except ValidationError as exc:
                ident = raw.get("document", raw) if isinstance(raw, Mapping) else {}
                results.append(
                    {
                        "document_id": ident.get("id") if isinstance(ident, Mapping) else None,
                        "revision": ident.get("revision") if isinstance(ident, Mapping) else None,
                        "outcome": "REJECTED",
                        "reason": _validation_code(exc),
                        "event_seq": None,
                        "access": None,
                    }
                )
                continue
            row = await self._store(context.tenant, item.document, item.access, ledger)
            results.append(row)
            if row["outcome"] in ("STORED", "DUPLICATE"):
                stored.append(item.document)
        committed_at = _now() if stored else None
        vectors, readiness = await self._prepare(context, stored, prepare_vectors)
        readiness["durable_commit"] = (
            _readiness("READY", "state_store_transaction_committed", committed_at)
            if stored
            else _readiness("NOT_APPLICABLE", "nothing_stored")
        )
        return {
            "api_version": API_VERSION,
            "operation": "ingest",
            "tenant": context.tenant,
            "results": results,
            "knowledge_seq": await asyncio.to_thread(self.storage.state.snapshot, context.tenant),
            "vectors": vectors,
            "readiness": readiness,
        }

    @staticmethod
    def _item(raw):
        if isinstance(raw, IngestItem):
            return raw
        if isinstance(raw, TemporalDocument):
            return IngestItem(document=raw)
        if isinstance(raw, Mapping) and "document" in raw:
            return IngestItem.model_validate(raw)
        return IngestItem(document=TemporalDocument.model_validate(raw))

    async def _store(self, tenant, document, access, ledger=None):
        state = self.storage.state
        row = {
            "document_id": document.id,
            "revision": document.revision,
            "outcome": None,
            "reason": None,
            "event_seq": None,
            "access": None,
        }
        if not hasattr(state, "commit_event"):
            return await self._store_in_two_steps(tenant, document, access, row)
        try:
            done = await asyncio.to_thread(
                state.commit_event,
                tenant,
                document=document,
                access={} if access in (UNCHANGED, None) else access.entries(),
                keep_access=access == UNCHANGED,
                ledger=ledger,
            )
        except ContractError as exc:
            if exc.code == "event_id_conflict":
                raise OperationError("apply_event", "INVALID_INPUT", "event_id_conflict") from None
            return row | {"outcome": "ERROR", "reason": "backend:" + type(exc).__name__}
        except ValueError as exc:  # nothing was written: the check runs before any change
            return row | {"outcome": "REJECTED", "reason": str(exc)}
        except _backend_errors() as exc:  # the transaction rolled back as a whole
            return row | {"outcome": "ERROR", "reason": "backend:" + type(exc).__name__}
        row["access"] = done["access"]
        if done.get("ledger") == "REPLAYED":
            return row | {
                "outcome": "DUPLICATE",
                "reason": "event_id_replayed",
                "event_seq": done["event_seq"],
            }
        if done["event_seq"] is not None:
            return row | {"outcome": "STORED", "event_seq": done["event_seq"]}
        return row | {"outcome": "DUPLICATE", "reason": "identical_revision_already_stored"}

    async def _store_in_two_steps(self, tenant, document, access, row):
        """A state store without `commit_event` (a caller's own port): the ACL, then the
        revision, as before RVI-07A — not atomic, and without an event-id ledger."""
        state = self.storage.state
        key = (document.id, document.revision)
        stored = (await asyncio.to_thread(state.bodies, tenant, [key])).get(key)
        if stored is not None and stored.model_dump_json() != document.model_dump_json():
            # Checked before the ACL, so a refused revision changes nothing.
            return row | {"outcome": "REJECTED", "reason": "immutable_temporal_revision_changed"}
        if access != UNCHANGED:
            try:
                row["access"] = await self._replace_access(tenant, document.id, access, "ingest")
            except OperationError as exc:
                return row | {"outcome": "REJECTED", "reason": "unsupported:" + exc.native_code}
            except ValueError as exc:
                return row | {"outcome": "REJECTED", "reason": str(exc)}
        before = await asyncio.to_thread(state.snapshot, tenant)
        try:
            await asyncio.to_thread(state.upsert, tenant, document)
        except ValueError as exc:
            return row | {"outcome": "REJECTED", "reason": str(exc)}
        except _backend_errors() as exc:  # the row says what failed, nothing else
            return row | {"outcome": "ERROR", "reason": "backend:" + type(exc).__name__}
        events = await asyncio.to_thread(state.events, tenant, before)
        mine = [
            e["seq"]
            for e in events
            if e["kind"] == "insert"
            and e["document_id"] == document.id
            and e["revision"] == document.revision
        ]
        if mine:
            return row | {"outcome": "STORED", "event_seq": mine[0]}
        return row | {"outcome": "DUPLICATE", "reason": "identical_revision_already_stored"}

    async def _replace_access(self, tenant, document_id, access, operation):
        state = self.storage.state
        if not hasattr(state, "replace_access"):
            raise OperationError(operation, "UNSUPPORTED_CAPABILITY", "acl_replace_not_available")
        entries = {} if access is None else access.entries()
        return await asyncio.to_thread(state.replace_access, tenant, document_id, entries)

    async def _prepare(self, context, stored, prepare_vectors):
        if self.embedding is None:
            vectors = {"requested": prepare_vectors, "space": None, "status": "NO_EMBEDDING"}
            readiness = {
                "embedding_ready": _readiness("NOT_APPLICABLE", "lexical_retrieval"),
                "index_visible": _readiness(
                    "READY", "lexical_search_reads_committed_state", _now()
                ),
            }
            return vectors, readiness
        space = self.embedding.describe().fingerprint
        vectors = {"requested": prepare_vectors, "space": space, "status": "NOT_REQUESTED"}
        if prepare_vectors:
            retriever = TemporalRetriever(self.storage, self.embedding)
            call = context.call or self.new_call()
            try:
                report = await retriever.index(context.tenant, call)
                vectors |= {"status": "DONE", "report": report}
            except ProviderError as exc:
                vectors |= {"status": "FAILED", "error": exc.code}
        pending = await asyncio.to_thread(self._pending, context.tenant, stored, space)
        observed = _now()
        kind = self.storage.describe()["index"]
        if not stored:
            ready = _readiness("NOT_APPLICABLE", "nothing_stored")
            return vectors | {"pending": 0}, {"embedding_ready": ready, "index_visible": ready}
        if pending:
            reason = "vectors_pending:" + str(pending)
            return vectors | {"pending": pending}, {
                "embedding_ready": _readiness("NOT_READY", reason, observed),
                # Revisions without a vector are still searched, lexically (the pending path).
                "index_visible": _readiness("NOT_READY", reason + ";lexical_fallback", observed),
            }
        return vectors | {"pending": 0}, {
            "embedding_ready": _readiness("READY", "vectors_stored", observed),
            # Every query builds its index view from the committed vectors and generations.
            "index_visible": _readiness("READY", "query_time_view:" + kind, observed),
        }

    def _pending(self, tenant, documents, space):
        from .storage.segments import catalog_entries

        hashes = set()
        for document in documents:
            try:
                hashes.update(input_hash(text) for *_, text in catalog_entries(document))
            except ProviderError:
                continue  # an unsupported segment is never embedded (listed by the catalog)
        have = self.storage.blobs.has_vectors(tenant, space, hashes)
        return len(hashes - set(have))

    async def apply_event(self, event, context: RequestContext, *, prepare_vectors=True):
        """A source event the caller's trusted context authorised: a new revision, a revocation of
        every revision of a document, or a document's whole ACL."""
        self._check("apply_event")
        context = self._context("apply_event", context)
        try:
            event = (
                _Event.model_validate({"event": event}).event
                if isinstance(event, Mapping)
                else event
            )
        except ValidationError as exc:
            code = "FORBIDDEN_INPUT" if _is_forbidden(exc) else "INVALID_INPUT"
            raise OperationError("apply_event", code, _validation_code(exc)) from None
        if not isinstance(event, (RevisionEvent, RevokeEvent, AccessEvent)):
            raise OperationError("apply_event", "UNSUPPORTED_CAPABILITY", "unknown_event")
        tenant, state = context.tenant, self.storage.state
        out = {
            "api_version": API_VERSION,
            "operation": "apply_event",
            "event_id": event.event_id,
            "type": event.type,
        }
        ledger = self._ledger(context, event)
        if isinstance(event, RevisionEvent):
            result = await self._ingest(
                [IngestItem(document=event.document, access=event.access)],
                context,
                prepare_vectors,
                ledger,
            )
            row = result["results"][0]
            outcome = {"STORED": "APPLIED"}.get(row["outcome"], row["outcome"])
            return out | {
                "outcome": outcome,
                "reason": row["reason"],
                "event_seq": row["event_seq"],
                "knowledge_seq": result["knowledge_seq"],
                "access": row["access"],
                "readiness": result["readiness"],
            }
        if not hasattr(state, "commit_event"):
            return out | await self._apply_in_two_steps(event, context)
        if isinstance(event, RevokeEvent):
            done = await self._commit(tenant, revoke=event.document_id, ledger=ledger)
            applied = done["event_seq"] is not None
            detail = {"event_seq": done["event_seq"], "access": None}
        else:
            done = await self._commit(
                tenant,
                target=event.document_id,
                access={} if event.access is None else event.access.entries(),
                keep_access=False,
                ledger=ledger,
            )
            applied = done["access"]["changed"] > 0
            detail = {"event_seq": None, "access": done["access"]}
        replayed = done.get("ledger") == "REPLAYED"
        at = _now()
        return out | {
            "outcome": "APPLIED" if applied and not replayed else "DUPLICATE",
            "reason": "event_id_replayed"
            if replayed
            else (None if applied else "state_already_as_requested"),
            "knowledge_seq": await asyncio.to_thread(state.snapshot, tenant),
            **detail,
            "readiness": self._mask_readiness(at),
        }

    @staticmethod
    def _mask_readiness(at):
        return {
            "durable_commit": _readiness("READY", "state_store_transaction_committed", at),
            "embedding_ready": _readiness("NOT_APPLICABLE", "no_new_text"),
            # Revocations and ACL are masks read from the authoritative state by every query
            # and checked again at the release fence.
            "index_visible": _readiness("READY", "query_time_mask_and_release_fence", at),
        }

    def _ledger(self, context, event):
        """The event-id ledger key and payload digest (RVI-07A), or None: an event without an
        id keeps the meaning it had (no ledger), as does a store without `commit_event`. The
        digest covers the tenant, the issuer and every field of the event but its id."""
        if event.event_id is None or not hasattr(self.storage.state, "commit_event"):
            return None
        issuer = context.issuer or (
            context.principal.principal_id if context.principal else "default"
        )
        body = {
            "tenant": context.tenant,
            "issuer": issuer,
            "event": event.model_dump(mode="json", exclude={"event_id"}),
        }
        digest = _sha(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return {"issuer": issuer, "event_id": event.event_id, "digest": digest, "kind": event.type}

    async def _commit(self, tenant, **kwargs):
        try:
            return await asyncio.to_thread(self.storage.state.commit_event, tenant, **kwargs)
        except ContractError as exc:
            if exc.code == "event_id_conflict":
                raise OperationError("apply_event", "INVALID_INPUT", "event_id_conflict") from None
            raise OperationError(
                "apply_event", "BACKEND_ERROR", type(exc).__name__, retryable=True
            ) from None
        except ValueError as exc:
            raise OperationError("apply_event", "INVALID_INPUT", str(exc)) from None
        except _backend_errors() as exc:  # the transaction rolled back; the type only
            raise OperationError(
                "apply_event", "BACKEND_ERROR", type(exc).__name__, retryable=True
            ) from None

    async def _apply_in_two_steps(self, event, context):
        """Revocation and ACL events on a store without `commit_event` (as before RVI-07A)."""
        tenant, state = context.tenant, self.storage.state
        if isinstance(event, RevokeEvent):
            before = await asyncio.to_thread(state.snapshot, tenant)
            await asyncio.to_thread(state.revoke, tenant, event.document_id)
            events = await asyncio.to_thread(state.events, tenant, before)
            mine = [
                e["seq"]
                for e in events
                if e["kind"] == "revoke" and e["document_id"] == event.document_id
            ]
            applied = bool(mine)
            detail = {"event_seq": mine[0] if mine else None, "access": None}
        else:
            access = await self._replace_access(
                tenant, event.document_id, event.access, "apply_event"
            )
            applied = access["changed"] > 0
            detail = {"event_seq": None, "access": access}
        return {
            "outcome": "APPLIED" if applied else "DUPLICATE",
            "reason": None if applied else "state_already_as_requested",
            "knowledge_seq": await asyncio.to_thread(state.snapshot, tenant),
            **detail,
            "readiness": self._mask_readiness(_now()),
        }

    # --- answers -------------------------------------------------------------------------------
    async def answer(self, request, context: RequestContext):
        """`UniversalEngine.answer` for this request, once. The envelope's `result` is the engine's
        result, unchanged."""
        self._check("answer")
        context = self._context("answer", context)
        try:
            question = (
                request
                if isinstance(request, TemporalQuestion)
                else TemporalQuestion.model_validate(request)
            )
        except ValidationError as exc:
            code = "FORBIDDEN_INPUT" if _is_forbidden(exc) else "INVALID_INPUT"
            raise OperationError("answer", code, _validation_code(exc)) from None
        request_id = context.request_id or uuid.uuid4().hex
        call = context.call or self.new_call()
        engine = UniversalEngine(
            retriever=TemporalRetriever(
                self.storage,
                self.embedding,
                allowed_ids=context.selected_document_ids,
                max_documents=self.max_documents,
                principal=context.principal,
            ),
            decision=self.decision,
            writer=self.writer,
            extractor=self.extractor,
            **({"handoff_packet": True} if self.handoff_packet else {}),
            **({"decision_cache": self.decision_cache} if self.decision_cache is not None else {}),
        )
        try:
            result = await engine.answer(context.tenant, question, call)
        except ConfigurationError as exc:
            raise OperationError("answer", "NOT_READY", type(exc).__name__) from exc
        except Exception as exc:
            raise OperationError("answer", "INTERNAL_ERROR", type(exc).__name__) from exc
        self._record(request_id, context, result)
        return {
            "api_version": API_VERSION,
            "request_id": request_id,
            "result": result,
            "usage": {
                "http_attempts": call.http_attempts,
                "decision_calls": call.decision_calls,
                "embedding_calls": call.embedding_calls,
                "generation_calls": call.generation_calls,
                "schema_repairs": call.schema_repairs,
                "reported_cost_usd": call.reported_cost_usd,
                "unknown_cost_calls": call.unknown_cost_calls,
            },
        }

    def _record(self, request_id, context, result):
        released = result.get("status") in ("FULL", "PARTIAL")
        # The units the released sentences cite, in order (0.1.0, parent PUB-2): not the writer's
        # evidence window (`result.evidence_ids`), which also holds units nobody cited.
        cited = dict.fromkeys(
            unit
            for sentence in result.get("sentences") or ()
            for unit in sentence.get("evidence_ids") or ()
        )
        self._lineage[request_id] = {
            "tenant": context.tenant,
            "principal_id": context.principal.principal_id if context.principal else None,
            "status": result.get("status"),
            "evidence_ids": tuple(cited) if released else (),
            "knowledge_snapshot": result.get("snapshot"),
            "release": result.get("release"),
            "config_hash": result.get("config_hash"),
            "storage": (result.get("storage") or {}).get("fingerprint"),
            "recorded_at": _now(),
        }
        self._lineage.move_to_end(request_id)
        while len(self._lineage) > self._lineage_capacity:
            self._lineage.popitem(last=False)

    # --- observation ---------------------------------------------------------------------------
    async def export_artifact_lineage(self, reference, context: RequestContext):
        """The released evidence of an answer this session gave, as source revision, unit, span
        and hashes, with the vector input it was retrieved by — under the permissions and
        revocations in force now. A reference of another tenant or principal is unknown."""
        self._check("export_artifact_lineage")
        context = self._context("export_artifact_lineage", context)
        request_id = reference.get("request_id") if isinstance(reference, Mapping) else reference
        record = self._lineage.get(request_id) if isinstance(request_id, str) else None
        principal = context.principal.principal_id if context.principal else None
        if (
            record is None
            or record["tenant"] != context.tenant
            or record["principal_id"] != principal
        ):
            raise OperationError(
                "export_artifact_lineage", "INVALID_INPUT", "lineage_reference_unknown"
            )
        return await asyncio.to_thread(self._lineage_of, request_id, record, context)

    def _lineage_of(self, request_id, record, context):
        state, tenant = self.storage.state, context.tenant
        hidden = state.hidden(tenant, context.principal) | state.revoked(tenant)
        units = [(_UNIT_ID.match(u), u) for u in record["evidence_ids"]]
        missing, artifacts = set(), []
        if any(m is None for m, _ in units):
            missing.add("UNPARSED_EVIDENCE_ID")
        units = [(m, u) for m, u in units if m is not None]
        withheld = [u for m, u in units if m["document"] in hidden]
        if withheld:
            missing.add("ACCESS_RESTRICTED_OR_REVOKED")
        visible = [(m, u) for m, u in units if m["document"] not in hidden]
        keys = {(m["document"], int(m["revision"])) for m, _ in visible}
        bodies = state.bodies(tenant, keys) if keys else {}
        space = self.embedding.describe().fingerprint if self.embedding is not None else None
        for m, unit_ref in visible:
            document = bodies.get((m["document"], int(m["revision"])))
            span = (
                next((s for s in document.units if s.id == m["unit"]), None) if document else None
            )
            if span is None:
                missing.add("SOURCE_UNIT_NOT_FOUND")
                continue
            source_sha = _sha(document.text)
            artifacts.append(
                {
                    "kind": "evidence_unit",
                    "ref": unit_ref,
                    "document_id": document.id,
                    "revision": document.revision,
                    "unit_id": span.id,
                    "source_sha256": source_sha,
                    "span": {"start": span.start, "end": span.end},
                    "unit_sha256": _sha(document.text[span.start : span.end]),
                    "derived_from": None,
                }
            )
            vector = self._vector_of(tenant, document, span.id, space, source_sha, unit_ref)
            if vector is None:
                missing.add("VECTOR_INPUT_UNKNOWN")
            else:
                artifacts.append(vector)
        return {
            "api_version": API_VERSION,
            "contract_version": OBSERVATION_VERSION,
            "request_id": request_id,
            "scope": "authorized_request_evidence",
            "completeness": "PARTIAL" if missing else "COMPLETE",
            "status": record["status"],
            "knowledge_snapshot": record["knowledge_snapshot"],
            "release": record["release"],
            "config_hash": record["config_hash"],
            "storage_fingerprint": record["storage"],
            "recorded_at": record["recorded_at"],
            "artifacts": artifacts,
            "missing_reasons": sorted(missing),
            "notes": [] if record["evidence_ids"] else ["no_released_evidence"],
        }

    def _vector_of(self, tenant, document, unit_id, space, source_sha, unit_ref):
        """The embedding input (a whole revision or its K8 segment) that holds the unit, and
        whether a vector of it is stored in this session's embedding space."""
        if space is None:
            return None
        try:
            segments = segment_revision(document)
        except ProviderError:
            return None
        from .storage.identity import embedding_input

        for segment in segments:
            if unit_id in segment.unit_ids:
                text = (
                    embedding_input(document)
                    if len(segments) == 1
                    else segment.embedding_input(document)
                )
                digest = input_hash(text)
                present = digest in set(self.storage.blobs.has_vectors(tenant, space, {digest}))
                return {
                    "kind": "vector",
                    "ref": f"{document.key}#{segment.segment_id}",
                    "document_id": document.id,
                    "revision": document.revision,
                    "unit_id": None,
                    "source_sha256": source_sha,
                    "span": {"start": segment.start, "end": segment.end},
                    "input_hash": digest,
                    "space": space,
                    "state": "PRESENT" if present else "ABSENT",
                    "derived_from": [unit_ref],
                }
        return None

    async def health(self, level="liveness", context: RequestContext | None = None):
        """Liveness: session, package resources and configuration (no I/O beyond the package).
        Readiness (asked explicitly, needs a tenant): also a bounded read of the state store and,
        for a vector service, a read-only count and the projection frontier."""
        if level not in ("liveness", "readiness"):
            raise OperationError("health", "INVALID_INPUT", "unknown_health_level")
        checks = self._liveness()
        if level == "readiness":
            if context is None:
                raise OperationError("health", "INVALID_INPUT", "readiness_needs_a_tenant")
            context = self._context("health", context)
            if not self.closed:
                checks += await self._readiness(context.tenant)
        statuses = [c["status"] for c in checks]
        status = (
            "NOT_READY"
            if "NOT_READY" in statuses
            else "DEGRADED"
            if "DEGRADED" in statuses
            else "OK"
        )
        return {
            "api_version": API_VERSION,
            "contract_version": OBSERVATION_VERSION,
            "level": level,
            "status": status,
            "measured_at": _now(),
            "checks": checks,
            "unverified": [c["component"] for c in checks if c["status"] == "UNKNOWN"],
            "provider_calls": 0,  # by construction; a runner verifies it with its own counters
            "state_mutated": False,
        }

    def _liveness(self):
        prompts = verify_all()
        checks = [
            {
                "component": "session",
                "status": "NOT_READY" if self.closed else "OK",
                "code": "session_closed" if self.closed else "open",
                "detail": None,
            },
            {
                "component": "policy_resource",
                "status": "OK",
                "code": "policy_sha256",
                "detail": universal_hash(),
            },
            {
                "component": "prompt_resources",
                "status": "OK" if prompts["ok"] else "NOT_READY",
                "code": "prompt_manifest_verified"
                if prompts["ok"]
                else prompts["failures"][0]["code"],
                "detail": prompts.get("manifest_sha256"),
            },
            {
                "component": "providers",
                "status": "OK",
                "code": "configured",
                "detail": ",".join(
                    f"{role}={_provider(p, role)['kind']}"
                    for role, p in (
                        ("decision", self.decision),
                        ("writer", self.writer),
                        ("extractor", self.extractor),
                        ("embedding", self.embedding),
                    )
                ),
            },
            {
                "component": "provider_reachability",
                "status": "UNKNOWN",
                "code": "not_checked_no_inference",
                "detail": None,
            },
        ]
        return checks

    async def _readiness(self, tenant):
        async def bounded(component, fn, ok_code):
            try:
                value = await asyncio.wait_for(
                    asyncio.to_thread(fn), timeout=READINESS_TIMEOUT_SECONDS
                )
                return {
                    "component": component,
                    "status": "OK",
                    "code": ok_code,
                    "detail": None,
                }, value
            except TimeoutError:
                code = "read_timeout"
            except _backend_errors() as exc:  # the type only: messages can hold a DSN
                code = "read_failed:" + type(exc).__name__
            return {
                "component": component,
                "status": "NOT_READY",
                "code": code,
                "detail": None,
            }, None

        state = self.storage.state
        check, _ = await bounded("state_store", lambda: state.generations(tenant), "read_ok")
        checks = [check]
        index = self.storage.index
        backend = getattr(index, "backend", None)
        if backend is None:
            checks.append(
                {
                    "component": "vector_index",
                    "status": "OK",
                    "code": "in_process_exact_index",
                    "detail": "matrices are built from committed vectors at query time; not warmed",
                }
            )
            return checks
        if self.embedding is None:
            return checks
        space = self.embedding.describe().fingerprint

        def count():
            try:
                return backend.count(tenant, space)
            except _backend_errors() as exc:
                # No table or collection for this space yet: nothing was projected, which is not
                # a failure of the service (queries rank unprojected vectors through the overlay).
                absent = type(exc).__name__ == "UndefinedTable" or (
                    isinstance(exc, ProviderError) and exc.status == 404
                )
                if absent:
                    return None
                raise

        check, counted = await bounded("vector_index", count, "count_ok")
        if check["status"] == "OK" and counted is None:
            check |= {"status": "DEGRADED", "code": "space_not_projected_yet"}
        checks.append(check)
        check, value = await bounded(
            "projection",
            lambda: (
                state.projection_frontier(tenant, space, backend.name)["vector_seq"],
                self.storage.blobs.vector_seq(tenant, space),
            ),
            "frontier_read",
        )
        if value is not None:
            frontier, latest = value
            if frontier < latest:
                # Queries still rank unprojected vectors exactly through the overlay.
                check |= {"status": "DEGRADED", "code": "projection_frontier_behind"}
            check["detail"] = f"frontier={frontier};vector_seq={latest}"
        checks.append(check)
        return checks

    # --- description ---------------------------------------------------------------------------
    def describe_runtime(self):
        """Identifiers of the package, policy, prompts, schemas, deployment and providers; no
        paths of data, no credentials, no connection strings."""
        schemas_sha, schemas = _schemas_sha256()
        policy = universal()
        try:
            dist = md.distribution("rvg-runtime")
            direct = dist.read_text("direct_url.json")
            editable = (
                bool(json.loads(direct).get("dir_info", {}).get("editable")) if direct else False
            )
        except md.PackageNotFoundError:
            editable = None
        described = self.storage.describe()
        return {
            "api_version": API_VERSION,
            "system_id": "rvg-runtime",
            "distribution": {"name": "rvg-runtime", "version": __version__},
            "policy": {
                "id": policy["version"],
                "sha256": universal_hash(),
                "semantic_sha256": _sha(
                    json.dumps(policy, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                ),
            },
            "prompt_manifest_sha256": manifest_sha256(),
            "schema_manifest_sha256": schemas_sha,
            "schemas": schemas,
            "backend": {
                "state_store": described["state"],
                "vector_index": described["index"],
                "precision": described["precision"],
                "storage_fingerprint": described["fingerprint"],
                "storage_config_hash": described["config_hash"],
            },
            "providers": {
                "decision": _provider(self.decision, "decision"),
                "writer": _provider(self.writer, "writer"),
                "extractor": _provider(self.extractor, "extractor"),
                "embedding": _provider(self.embedding, "embedding")
                | (
                    {"space": self.embedding.describe().fingerprint}
                    if self.embedding is not None
                    else {}
                ),
                "transport": self._transport_origin,
            },
            "settings": {
                "mode": self.settings.deepjev_mode if self.settings else None,
                "decision_provider": self.settings.jev_provider if self.settings else None,
            },
            "install": {
                "module_origin": str(Path(__file__).resolve().parent),
                "editable": editable,
                "source_commit": None,  # linked by the release manifest, never embedded
                "dirty": None,
            },
        }

    def capabilities(self):
        """The build's capability table with this session's configuration. Availability now is
        what `health` reports; nothing here opens a connection."""
        from .api import capabilities

        table = capabilities()
        table["session"] = {
            "state_store": self.storage.describe()["state"],
            "vector_index": self.storage.describe()["index"],
            "embedding_configured": self.embedding is not None,
            "handoff_packet": self.handoff_packet,
            "decision_cache": self.decision_cache.describe() if self.decision_cache else None,
            "acl_replace_available": hasattr(self.storage.state, "replace_access"),
            "providers": {
                role: _provider(p, role)["kind"]
                for role, p in (
                    ("decision", self.decision),
                    ("writer", self.writer),
                    ("extractor", self.extractor),
                    ("embedding", self.embedding),
                )
            },
            "availability": "UNKNOWN",  # see health()
        }
        return table


def open_session(storage, settings: Settings | None = None, **options) -> RuntimeSession:
    """`RuntimeSession.open`: a session to use as `async with open_session(...) as session:`."""
    return RuntimeSession.open(storage, settings, **options)


__all__ = [
    "API_VERSION",
    "OBSERVATION_VERSION",
    "AccessEvent",
    "DocumentAccess",
    "IngestItem",
    "OperationError",
    "RequestContext",
    "RevisionEvent",
    "RevokeEvent",
    "RuntimeSession",
    "open_session",
]
