"""Write `rvg_public_api_map.json`: the public API of this build mapped onto the neutral
operations of the KB-FF handoff contract (`kbff-rvg-handoff/1.0`, format `api_mapping`).

    python scripts/make_api_map.py --source-commit <40 hex> --junit <pytest junit xml> \
        --out <handoff>/rvg_public_api_map.json [--schemas <kbff contracts v1.0 dir>]

It runs after the release tests, outside the source tree (the map names the source commit, so it
is never committed into that commit). Only public paths (`rvg.*` names, `RuntimeSession`
methods and the `rvg.testing.fake_openrouter` test double) appear; semantics are those of
docs/api/PUBLIC_API.md."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

HANDOFF = "kbff-rvg-handoff/1.0"
CONTRACT_TEST_FILES = (
    "tests/test_session_api.py",
    "tests/test_session_backends.py",
    "tests/test_access_replace.py",
    "tests/test_health_audit.py",
    "tests/test_prompt_resources.py",
    "tests/test_source_equivalence.py",
    "tests/test_settings_compat.py",
    "tests/packaging/test_public_consumer.py",
    "tests/packaging/test_public_api.py",
    "tests/packaging/test_import.py",
    "tests/packaging/test_isolation.py",
    "tests/packaging/test_check_dist.py",
)
SESSION = "rvg:RuntimeSession"


def op(status, path, call, input_, output, errors, notes, evidence=()):
    return {
        "status": status,
        "public_path": path,
        "call": call,
        "input": input_,
        "output": output,
        "errors": errors,
        "notes": notes,
        "evidence": list(evidence),
    }


OP_ERRORS = (
    "raises rvg.OperationError(operation, code, native_code); code in UNSUPPORTED_CAPABILITY, "
    "INVALID_INPUT, FORBIDDEN_INPUT, NOT_READY, BACKEND_ERROR, PROVIDER_ERROR, TIMEOUT, "
    "INTERNAL_ERROR; OperationError.to_dict() follows rvg/resources/schemas/api/"
    "operation_error.schema.json"
)

OPERATIONS = {
    "capabilities": op(
        "MAPPED",
        "rvg:capabilities",
        "sync",
        "none (build table); RuntimeSession.capabilities() adds the session's configuration",
        "dict: per component status, evidence, implemented, evidence_level, scope, default, "
        "experimental; session_operations/events/access/observability sections",
        "none; opens no connection",
        "Availability now is health(), never capabilities(). availability is not claimed "
        "(session.availability = UNKNOWN).",
        ["tests/packaging/test_public_api.py", "tests/test_session_api.py"],
    ),
    "describe_runtime": op(
        "MAPPED",
        SESSION + ".describe_runtime",
        "sync",
        "session",
        "dict per rvg/resources/schemas/api/runtime_description.schema.json: distribution, "
        "policy id/sha256/semantic_sha256, prompt_manifest_sha256, schema_manifest_sha256, "
        "backend (state_store, vector_index, precision, fingerprints), providers (kind, model_id, "
        "network), install (module_origin, editable; source_commit and dirty are null by design)",
        "none",
        "No credential, DSN or data path. The source commit is linked by the release manifest, "
        "never embedded in the wheel.",
        ["tests/test_session_api.py::test_description_and_capabilities_carry_no_secret"],
    ),
    "ingest": op(
        "MAPPED",
        SESSION + ".ingest",
        "async",
        "documents: list of rvg.TemporalDocument | dict | rvg.IngestItem(document, access); "
        "context: rvg.RequestContext(tenant, principal=None, call=None); "
        "keyword prepare_vectors=True",
        "dict per ingest_result.schema.json: results[] (document_id, revision, outcome "
        "STORED|DUPLICATE|REJECTED|ERROR, reason, event_seq, access), knowledge_seq, vectors, "
        "readiness{durable_commit, embedding_ready, index_visible}",
        "per-document REJECTED/ERROR rows (nothing changed for REJECTED); context errors raise "
        "rvg.OperationError",
        "Each revision is its own transaction. IngestItem.access (DocumentAccess or None) is "
        "applied first, atomically, to the document id (all revisions); an immutability "
        "violation is refused before the ACL is touched. neutral outcome STORED = native STORED.",
        ["tests/test_session_api.py", "tests/test_session_backends.py"],
    ),
    "apply_event": op(
        "MAPPED",
        SESSION + ".apply_event",
        "async",
        "event: rvg.RevisionEvent | rvg.RevokeEvent | rvg.AccessEvent or dict with type "
        "revision|revoke|access (+ event_id); context: rvg.RequestContext",
        "dict per event_result.schema.json: outcome APPLIED|DUPLICATE|REJECTED|ERROR, reason, "
        "event_seq, knowledge_seq, access, readiness",
        OP_ERRORS,
        "The caller's trusted context authorises the event (the session object is the "
        "capability); a document's text is never an event. Duplicates are judged by content "
        "(no event-id ledger).",
        ["tests/test_session_api.py", "tests/test_access_replace.py"],
    ),
    "answer": op(
        "MAPPED",
        SESSION + ".answer",
        "async",
        "request: rvg.TemporalQuestion or dict {text, as_of, known_at, locale}; extra fields are "
        "refused (FORBIDDEN_INPUT); context: rvg.RequestContext(tenant, principal: "
        "rvg.TrustedPrincipal | None, request_id, selected_document_ids, call)",
        "dict per answer_envelope.schema.json: {api_version, request_id, result, usage}; result "
        "is UniversalEngine.answer's dict unchanged (status FULL|PARTIAL|ABSTAIN|"
        "UNRESOLVED_SCOPE|ERROR, text, sentences, short_answer, evidence_ids, missing, reason, "
        "error, release, snapshot, config_hash, storage, audit, trace, ...); evidence_ids is the "
        "writer's evidence window, the cited units are sentences[].evidence_ids; a result that "
        "is not released carries no draft value (short_answer null, no claims, 0.1.0)",
        "a failure inside the engine is result.status ERROR (never ABSTAIN); input and "
        "configuration errors raise rvg.OperationError",
        "Calls UniversalEngine.answer once; no retrieval, rewriting, verification or status "
        "change is added. Release mode is the product default `current`.",
        ["tests/test_session_api.py::test_the_session_answer_is_the_engine_answer_unchanged"],
    ),
    "health": op(
        "MAPPED",
        SESSION + ".health",
        "async",
        "level: 'liveness' | 'readiness'; readiness needs context (tenant)",
        "dict per health.schema.json: status OK|DEGRADED|NOT_READY, measured_at, checks[] "
        "(component, status OK|DEGRADED|NOT_READY|UNSUPPORTED|UNKNOWN, code, detail), "
        "unverified[], provider_calls 0, state_mutated false",
        OP_ERRORS,
        "liveness: session, package resources, provider configuration (no DB, network or model). "
        "readiness: bounded (5 s) read-only reads of the open state store and, for a vector "
        "service, a count and the projection frontier; never migrates, warms or calls a model. "
        "provider_calls/state_mutated are design statements: measure them.",
        ["tests/test_health_audit.py", "tests/test_session_backends.py"],
    ),
    "export_artifact_lineage": op(
        "MAPPED",
        SESSION + ".export_artifact_lineage",
        "async",
        "reference: request_id (str) or {request_id}; context: rvg.RequestContext (same tenant "
        "and principal as the answer)",
        "dict per lineage.schema.json: scope authorized_request_evidence, completeness "
        "COMPLETE|PARTIAL, artifacts[] (evidence_unit: document_id, revision, unit_id, "
        "source_sha256, span, unit_sha256; vector: input_hash, space, state PRESENT|ABSENT, "
        "derived_from), missing_reasons, release, knowledge_snapshot",
        "an unknown reference, another tenant or another principal: rvg.OperationError "
        "(INVALID_INPUT, lineage_reference_unknown) — the same error for all three",
        "Records the units the released sentences cite (sentences[].evidence_ids, not the "
        "writer's window; 0.1.0) of answers given by this session (memory, not persisted). "
        "Current ACL and revocations are applied at export: a withheld source appears only as "
        "missing_reasons ACCESS_RESTRICTED_OR_REVOKED. Hashes identify bytes, not support.",
        ["tests/test_session_api.py"],
    ),
    "close": op(
        "MAPPED",
        SESSION + ".aclose",
        "async",
        "session",
        "None",
        "none; idempotent; later operations raise OperationError(NOT_READY, session_closed)",
        "Closes only what the session opened (a storage opened from a configuration, an HTTP "
        "client it created). `async with rvg.open_session(...)` closes on exit.",
        ["tests/test_session_api.py"],
    ),
    "fixed_draft_verify": op(
        "UNSUPPORTED",
        None,
        None,
        "not available",
        "not available",
        "no public operation; capabilities()['session_operations']['fixed_draft_verification']"
        " is NOT_IMPLEMENTED",
        "No public verifier-only operation in this build (CR-001: B2 RVG cell BLOCKED_API). "
        "Internal controller functions are not a public path.",
    ),
}


def fm(neutral, native, same, notes):
    return {"neutral": neutral, "native": native, "same_semantics": same, "notes": notes}


FIELD_MAPPINGS = [
    fm(
        "document.tenant_id",
        "RequestContext.tenant",
        True,
        "tenant is given by the context of the call",
    ),
    fm("document.document_id", "TemporalDocument.id", True, "pattern ^[\\w-]+$, max 160"),
    fm(
        "document.revision",
        "TemporalDocument.revision",
        True,
        "a higher revision of the same id replaces the lower from its effective start; expiry "
        "of the successor does not revive the predecessor (R-REV-1, R-REV-4)",
    ),
    fm(
        "document.family_id",
        "TemporalDocument.family",
        True,
        "same identifier; RVG additionally applies its adopted implicit-revision prior (U10) to "
        "overlapping later documents of one family (a system choice, not a benchmark rule)",
    ),
    fm("document.title", "TemporalDocument.title", True, "max 500"),
    fm(
        "document.text",
        "TemporalDocument.text",
        True,
        "max 400000; long revisions are segmented for retrieval (K8), never cut",
    ),
    fm("document.units[].unit_id", "TemporalDocument.units[].id", True, "citation unit"),
    fm(
        "document.units[].start/end",
        "TemporalDocument.units[].start/end",
        True,
        "[start, end) character offsets",
    ),
    fm("document.published_at", "TemporalDocument.published_at", True, "timezone-aware"),
    fm(
        "document.received_at",
        "TemporalDocument.observed_at",
        True,
        "revisions with observed_at > known_at are unknown to the request; observed_at < "
        "published_at is rejected (observation_before_publication, R-INPUT)",
    ),
    fm("document.valid_from", "TemporalDocument.effective_from", True, "null = from published_at"),
    fm(
        "document.valid_until",
        "TemporalDocument.effective_until",
        True,
        "exclusive: in force while as_of < effective_until",
    ),
    fm("document.authority", "TemporalDocument.authority", True, "primary | secondary"),
    fm(
        "document.supersedes[]",
        "TemporalDocument.supersedes[] (document_id, unit_ids)",
        True,
        "empty unit_ids = the whole document",
    ),
    fm(
        "document.acl",
        "IngestItem.access / AccessEvent.access: rvg.DocumentAccess(allow, deny) or None",
        True,
        "per document id, latest value (R-ACL-3); allow None = no allow restriction; deny wins; "
        "names match TrustedPrincipal.principal_id or a group; allow = [] is UNSUPPORTED",
    ),
    fm(
        "document.provenance",
        "TemporalDocument.source",
        False,
        "one free string (max 1000) that the writer sees with the evidence; map source_uri if "
        "present, else channel; synthetic and license have no native field",
    ),
    fm(
        "document.locale",
        None,
        None,
        "documents carry no locale; the answer language is the question's locale",
    ),
    fm(
        "document.source_sha256",
        None,
        None,
        "not an input; RVG computes sha256(text) itself (lineage source_sha256)",
    ),
    fm(
        "runtime_query.request_id",
        "RequestContext.request_id",
        True,
        "pattern ^[A-Za-z0-9_-]{1,160}$; generated when absent",
    ),
    fm("runtime_query.tenant_id", "RequestContext.tenant", True, ""),
    fm("runtime_query.question", "TemporalQuestion.text", True, "max 2000"),
    fm(
        "runtime_query.locale",
        "TemporalQuestion.locale",
        True,
        "answer language only; never selects a policy",
    ),
    fm("runtime_query.as_of", "TemporalQuestion.as_of", True, "application time of the facts"),
    fm(
        "runtime_query.knowledge_cutoff",
        "TemporalQuestion.known_at",
        True,
        "record range: observed_at <= known_at",
    ),
    fm(
        "runtime_query.release_contract=RELEASE_CURRENT",
        "UniversalEngine default release mode 'current'",
        True,
        "if the tenant's knowledge snapshot moves while the answer is made, nothing is released "
        "(ABSTAIN source_changed, no retry); revocation and ACL are re-checked at the release "
        "fence",
    ),
    fm("runtime_query.release_contract=SNAPSHOT", None, None, "UNSUPPORTED by the public API"),
    fm(
        "runtime_query.session_context.selected_document_ids",
        "RequestContext.selected_document_ids",
        False,
        "RVG narrows retrieval candidates to these ids (allowed_ids); it does not use them as the "
        "question's scope context (R-SCOPE)",
    ),
    fm("runtime_query.session_context.turns", None, None, "UNSUPPORTED"),
    fm("runtime_query.answer_fields", None, None, "UNSUPPORTED"),
    fm(
        "runtime_query.deadline_ms",
        "RequestContext.call = RuntimeSession.new_call(deadline_seconds)",
        True,
        "a deadline for the whole request; exceeding it ends the request as result.status ERROR "
        "(deadline_exceeded)",
    ),
    fm("trusted_context.tenant_id", "RequestContext.tenant", True, ""),
    fm(
        "trusted_context.principal_id",
        "TrustedPrincipal.principal_id",
        True,
        "object from the trusted context only",
    ),
    fm("trusted_context.groups", "TrustedPrincipal.groups", True, ""),
    fm(
        "trusted_context.auth_source=OPERATOR_PRIVILEGED",
        None,
        None,
        "no privileged operation or audit export",
    ),
    fm("response.request_id", "envelope.request_id", True, ""),
    fm("response.status", "result.status", True, "same five names"),
    fm("response.native_status", "result.status", True, ""),
    fm("response.reason", "result.reason", True, "also result.abstain_reason when a stage failed"),
    fm(
        "response.text",
        "result.text",
        True,
        "verified sentences joined; '' for ABSTAIN, UNRESOLVED_SCOPE, ERROR",
    ),
    fm(
        "response.short_answer",
        "result.short_answer when result.status in (FULL, PARTIAL), else null",
        True,
        "status is the release contract. result.short_answer is the writer draft's value of the "
        "kept stage for every status (see known_gaps SHORT_ANSWER_*): published for FULL and "
        "PARTIAL, null for the others (kept in native_output)",
    ),
    fm(
        "response.sentences[].text / citations",
        "result.sentences[i].text / result.sentences[i].evidence_ids",
        True,
        "citation raw = 'document@revision#unit'; split reversibly at '@' and '#'",
    ),
    fm(
        "response.citations",
        "union of result.sentences[i].evidence_ids in order",
        True,
        "result.evidence_ids is NOT a citation list: it is every unit given to the writer at the "
        "kept stage (native_output only)",
    ),
    fm(
        "response.answer_kind",
        "result.answer_kind (set only by the N4 conflict-rewrite arm, off by default)",
        None,
        "absent -> 'unknown' (not inferred)",
    ),
    fm(
        "response.missing_items",
        "result.missing",
        True,
        "unanswered requested items reported by the controller",
    ),
    fm("response.claims", None, None, "no structured claims are shown to users; null"),
    fm(
        "response.release_record",
        "result.release {mode, knowledge_snapshot, revocation_seq, acl_epoch}",
        True,
        "released_at: null (the release instant is not recorded); present only for FULL/PARTIAL",
    ),
    fm(
        "response.error",
        "result.error",
        True,
        "{type, code, http_status, detail} for a provider/store/budget failure, or {stage, "
        "codes} for a failed stage; code mapping to the neutral enum is the adapter's",
    ),
    fm(
        "response.usage",
        "envelope.usage {http_attempts, decision_calls, embedding_calls, generation_calls, "
        "schema_repairs, reported_cost_usd, unknown_cost_calls}",
        None,
        "counts of this request; reported_cost_usd is the providers' report (unknown_cost_calls "
        "counts calls without one)",
    ),
]

STATUS_MAPPING = [
    {"native": s, "neutral": s} for s in ("FULL", "PARTIAL", "ABSTAIN", "UNRESOLVED_SCOPE", "ERROR")
]

EVENT_MAPPING = [
    {
        "neutral_type": "UPSERT_DOCUMENT",
        "status": "MAPPED",
        "native": SESSION + ".apply_event({'type': 'revision', 'event_id', 'document', 'access'})",
        "notes": "document.acl -> access (DocumentAccess | None), applied first; outcome APPLIED "
        "(new revision) | DUPLICATE (same bytes) | REJECTED (e.g. changed bytes of an existing "
        "revision)",
    },
    {
        "neutral_type": "REVOKE_DOCUMENT",
        "status": "MAPPED",
        "native": SESSION + ".apply_event({'type': 'revoke', 'event_id', 'document_id'})",
        "notes": "every revision of the document; DUPLICATE when already revoked",
    },
    {
        "neutral_type": "ACL_SET",
        "status": "PARTIAL",
        "native": SESSION + ".apply_event({'type': 'access', 'event_id', 'document_id', 'access'})",
        "notes": "whole-document ACL replaced in one transaction (SQLite, PostgreSQL); same ACL -> "
        "DUPLICATE; allow = [] (nobody) is UNSUPPORTED_CAPABILITY (acl_empty_allow_list)",
    },
]

KNOWN_GAPS = [
    (
        "FIXED_DRAFT_VERIFY_UNSUPPORTED",
        "No public verifier-only operation (CR-001).",
        "B2 RVG cell BLOCKED_API",
    ),
    (
        "EVENT_ID_IDEMPOTENCY",
        "No event-id ledger: a repeated event is DUPLICATE by content; a stale redelivery after a later change of the same object is applied again.",
        "operational replays must keep source order",
    ),
    (
        "EMPTY_ALLOW_LIST",
        "An ACL whose allow list is empty (nobody) cannot be represented and is refused.",
        "such cases UNSUPPORTED",
    ),
    (
        "R_ACL_2_HIDDEN_SUPERSEDER",
        "When a document that supersedes (supersedes[]) units of another is hidden from the principal or revoked, retrieval removes it before the validity gate, so the superseded units can be released. Same-id revisions share one ACL and are hidden together.",
        "behaviour preserved in this release; scored by the benchmark rule",
    ),
    (
        "SHORT_ANSWER_PARTIAL_UNFILTERED",
        "For PARTIAL, result.short_answer is the draft's value, not reduced to the verified sentences.",
        "an unverified value can be shown with a PARTIAL answer",
    ),
    (
        "SHORT_ANSWER_IN_UNRELEASED_RESULTS",
        "For ABSTAIN/UNRESOLVED_SCOPE/ERROR the engine dict can still hold the draft short_answer (and the CLI prints the dict); the release contract is the status.",
        "native_output may contain an unreleased draft value",
    ),
    (
        "RELEASE_MODES",
        "Only the product default release mode `current` is public; snapshot modes are evaluation arms, not public API.",
        "SNAPSHOT queries UNSUPPORTED",
    ),
    (
        "SESSION_CONTEXT",
        "selected_document_ids narrow retrieval only (not scope context); turns and answer_fields are not accepted.",
        "CONTEXT cells ADAPTED/UNSUPPORTED",
    ),
    (
        "LINEAGE_SCOPE",
        "Lineage covers released citations of answers made by the same session (memory, capacity 1024); no persistence, no operator audit export, no derived graph/summary artifacts.",
        "lineage of other sessions UNKNOWN",
    ),
    (
        "TIMESTAMPS",
        "The runtime records no commit, index-visible or release instants; readiness observed_at is the session's check time; release_record.released_at is null.",
        "runners measure times",
    ),
    (
        "ACL_THEN_REVISION",
        "An ingest item with an ACL applies the ACL and the revision in two transactions (ACL first): the new revision is never visible without its ACL; older revisions see the new ACL slightly earlier.",
        "none for restricted-first cases",
    ),
    (
        "OFFLINE_PROVIDERS",
        "Offline runs use the deterministic test double rvg.testing.fake_openrouter through the real provider, transport and parser code; its answers carry no model quality.",
        "mock/replay results are not model results",
    ),
    (
        "LIVE_MODEL_NOT_RETESTED",
        "No paid model call was made after the extraction; TESTED_LIVE entries refer to research runs on the same source bytes before it.",
        "NOT_TESTED_LIVE for this build",
    ),
    (
        "PROVIDER_REACHABILITY",
        "health never calls a provider; provider reachability is UNKNOWN in every report.",
        "none",
    ),
]


def contract_tests(junit):
    tree = ET.parse(junit)
    rows = []
    for case in tree.iter("testcase"):
        path = case.get("file") or case.get("classname", "").replace(".", "/") + ".py"
        path = path if path.startswith("tests/") else "tests/" + path.split("tests/")[-1]
        if path not in CONTRACT_TEST_FILES:
            continue
        if case.find("failure") is not None or case.find("error") is not None:
            result, notes = "FAIL", None
        elif case.find("skipped") is not None:
            result, notes = "SKIP", case.find("skipped").get("message")
        else:
            result, notes = "PASS", None
        rows.append(
            {"name": f"{path}::{case.get('name')}", "path": path, "result": result, "notes": notes}
        )
    return rows


def build(source_commit, version, junit):
    return {
        "handoff_contract": HANDOFF,
        "system": {
            "distribution": "rvg-runtime",
            "version": version,
            "import_name": "rvg",
            "source_commit": source_commit,
            "api_version": "rvg-runtime-api/0.1",
        },
        "construction": {
            "public_path": "rvg:open_session",
            "call": "async_context_manager",
            "notes": "rvg.open_session(storage, settings=None, *, root=None, environ=None, "
            "transport=None, providers=None, max_documents=24, handoff_packet=False, "
            "decision_cache=None) builds a session synchronously; use "
            "`async with`. storage: an open rvg.Storage/rvg.TemporalStore (caller-owned) or a "
            "storage config (path/dict/rvg.StorageConfig; DSNs by environment-variable name, "
            "values via `environ`), opened and owned by the session. Providers: `providers` dict "
            "(decision, writer, extractor, embedding) or built from rvg.Settings by the product "
            "factories. OFFLINE (no network, no paid call): providers = rvg.testing."
            "fake_openrouter.providers(rvg.JsonTransport(httpx.AsyncClient(transport=httpx."
            "MockTransport(rvg.testing.fake_openrouter.FakeOpenRouter().handler)), timeout=5, "
            "max_attempts=3)) — the real provider/transport/parser code with a deterministic "
            "stand-in (example: examples/api/public_consumer.py). Request budget: "
            "RequestContext.call or session.new_call(deadline_seconds). Opt-in tools (0.1.0 "
            "public names): rvg.CacheMode/rvg.DecisionCache (decision_cache), "
            "rvg.write_external_set/rvg.export_external_set (external float16 vector sets), "
            "rvg.build_candidate_codes (TurboQuant candidate-stage codes).",
        },
        "trusted_principal": {
            "public_path": "rvg:TrustedPrincipal",
            "call": "sync",
            "notes": "rvg.TrustedPrincipal(principal_id, groups=()) built from trusted_context "
            "only and passed as rvg.RequestContext(principal=...); a string principal is "
            "FORBIDDEN_INPUT and the question DTO refuses extra fields.",
        },
        "operations": OPERATIONS,
        "field_mappings": FIELD_MAPPINGS,
        "status_mapping": STATUS_MAPPING,
        "event_mapping": EVENT_MAPPING,
        "known_gaps": [{"id": i, "description": d, "impact": m} for i, d, m in KNOWN_GAPS],
        "contract_tests": contract_tests(junit),
    }


def validate(body, schemas):
    import jsonschema
    from referencing import Registry, Resource

    resources = []
    for path in Path(schemas).glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))
    registry = Registry().with_resources(resources)
    schema = registry.contents("urn:kbff:v1.0:api_mapping")
    jsonschema.Draft202012Validator(schema, registry=registry).validate(body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-commit", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--junit", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--schemas", help="KB-FF contract schema directory (v1.0) to validate against")
    args = p.parse_args()
    body = build(args.source_commit, args.version, args.junit)
    if args.schemas:
        validate(body, args.schemas)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = [t["name"] for t in body["contract_tests"] if t["result"] == "FAIL"]
    print(json.dumps({"tests": len(body["contract_tests"]), "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
