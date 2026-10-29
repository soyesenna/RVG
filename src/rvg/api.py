"""The public surface of RVG: what an application, a research runner or a benchmark submission
adapter imports. Every name is an existing runtime object re-exported, a read-only view of the
policy and of what this build actually supports, or the runtime session (`open_session`) that
assembles those objects; nothing here re-implements an answer.

Composition is explicit: a caller builds `Settings(...)` from its own configuration (arguments
and the environment; never a `.env` file), builds the providers from it (`decision_provider`,
`generation_provider`, `embedding_provider`, or its own objects with the same methods), opens a
deployment (`open_storage` or `TemporalStore`), and hands them to `UniversalEngine` — or lets
`open_session` do exactly that. A missing provider is a configuration error; nothing falls back to
another provider or to a network API. A principal comes from the caller's trusted auth context
(`TemporalRetriever(principal=...)`, `RequestContext(principal=...)`), never from the request
text."""

from __future__ import annotations

import copy
import importlib.util

from . import __version__
from .config import Settings, load_settings
from .errors import BudgetExceeded, ConfigurationError, ContractError, ProviderError
from .factory import decision_provider, embedding_provider, generation_provider
from .prompts import PromptResourceError
from .runtime import CallContext
from .session import (
    API_VERSION,
    OBSERVATION_VERSION,
    AccessEvent,
    DocumentAccess,
    IngestItem,
    OperationError,
    RequestContext,
    RevisionEvent,
    RevokeEvent,
    RuntimeSession,
    open_session,
)
from .storage import Storage, StorageConfig, load_config, open_storage
from .storage.external import export_external_set, write_external_set
from .storage.quantized import build_candidate_codes
from .temporal.access import TrustedPrincipal
from .temporal.models import AnswerDraft, TemporalDocument, TemporalQuestion
from .temporal.store import TemporalRetriever, TemporalStore
from .transport import JsonTransport
from .universal import universal, universal_hash
from .universal.cache import CacheMode, DecisionCache
from .universal.engine import UniversalEngine

__all__ = [
    "API_VERSION",
    "OBSERVATION_VERSION",
    "AccessEvent",
    "AnswerDraft",
    "BudgetExceeded",
    "CacheMode",
    "CallContext",
    "ConfigurationError",
    "ContractError",
    "DecisionCache",
    "DocumentAccess",
    "IngestItem",
    "JsonTransport",
    "OperationError",
    "PromptResourceError",
    "ProviderError",
    "RequestContext",
    "RevisionEvent",
    "RevokeEvent",
    "RuntimeSession",
    "Settings",
    "Storage",
    "StorageConfig",
    "TemporalDocument",
    "TemporalQuestion",
    "TemporalRetriever",
    "TemporalStore",
    "TrustedPrincipal",
    "UniversalEngine",
    "build_candidate_codes",
    "capabilities",
    "decision_provider",
    "embedding_provider",
    "export_external_set",
    "generation_provider",
    "load_config",
    "load_settings",
    "open_session",
    "open_storage",
    "policy",
    "policy_hash",
    "write_external_set",
]

STATUSES = ("FULL", "PARTIAL", "ABSTAIN", "UNRESOLVED_SCOPE", "ERROR")


def policy() -> dict:
    """A copy of the one adopted policy (`rvg/resources/policies/universal.json`)."""
    return copy.deepcopy(universal())


def policy_hash() -> str:
    """sha256 of the policy file's bytes; stamped on every answer as `config_hash`."""
    return universal_hash()


# The strongest evidence behind a status word (docs/capabilities.md). A status is about the code
# and its recorded tests; whether a component is reachable now is `RuntimeSession.health`.
EVIDENCE = {
    "ADOPTED": "TESTED_OFFLINE",
    "ADOPTED_OPT_IN": "TESTED_OFFLINE",
    "IMPLEMENTED": "TESTED_OFFLINE",
    "TESTED_OFFLINE": "TESTED_OFFLINE",
    "TESTED_REAL_BACKEND": "TESTED_REAL_BACKEND",
    "TESTED_LIVE": "TESTED_LIVE",
    "EXPERIMENTAL": "TESTED_OFFLINE",
    "EXPERIMENTAL_ARM": "TESTED_OFFLINE",
    "NOT_IMPLEMENTED": "NOT_TESTED",
}


def capabilities() -> dict:
    """What this build contains and how far each part has been verified. A status names the
    strongest evidence recorded for that component and `evidence` says where it comes from
    (docs/capabilities.md); a component that is not implemented says so instead of being
    served by another one. Each entry also says, separately, whether the code exists
    (`implemented`), the evidence level and its `scope`, whether it is the product default and
    whether it is experimental. This is a static description of the build: it opens no
    connection and says nothing about availability now (that is `RuntimeSession.health`).
    Every entry has a row in docs/capabilities.md with the same status (a packaging test compares
    the two tables); `policy_defaults` reads its values from the policy file itself."""

    def cap(status, evidence, *, default=False, scope=None):
        return {
            "status": status,
            "evidence": evidence,
            "implemented": status != "NOT_IMPLEMENTED",
            "evidence_level": EVIDENCE[status],
            "scope": scope or evidence,
            "default": default or status == "ADOPTED",
            "experimental": status in ("EXPERIMENTAL", "EXPERIMENTAL_ARM"),
        }

    before = "research repository before extraction, same source bytes"
    live = before + "; no live model call after extraction"
    suite = "RVG test suite of this build"
    adopted = universal()

    def setting(path, evidence):
        """An adopted policy key: its status and its value as the policy file holds it."""
        value = adopted
        for part in path.split("."):
            value = value[part]
        return cap("ADOPTED", evidence) | {"value": copy.deepcopy(value)}

    return {
        "package": {"distribution": "rvg-runtime", "version": __version__},
        "api_version": API_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "policy": {"sha256": universal_hash(), "version": universal()["version"]},
        "statuses": list(STATUSES),
        "release_modes": {
            "current": cap(
                "IMPLEMENTED",
                "default before 0.1.0.dev27; an arm with release.mode = current (the rollback)",
            ),
            "as_of_snapshot": cap("EXPERIMENTAL_ARM", "evaluation runners only"),
            "latest_at_release": cap("EXPERIMENTAL_ARM", "evaluation runners only; N3L-2 REJECT"),
            "scoped_dependency": cap(
                "ADOPTED",
                "default of every engine since 0.1.0.dev27 (U24): the policy's release block, "
                "vectors scoped, vector moves after the check, one slack retry; ACL moves after "
                "the check since 0.1.0.dev42 (U32)",
            ),
        },
        "answer_path": cap(
            "ADOPTED",
            "UniversalEngine.answer: retrieval, validity, scope, writing, control and the cascade; "
            "research repository N0-N6, N0 tape replay 120/120 after extraction",
        ),
        "policy_defaults": {
            "release.mode": setting("release.mode", "since 0.1.0.dev27 (U24)"),
            "release.retry": setting("release.retry", "since 0.1.0.dev27 (U24)"),
            "release.retry_rule": setting("release.retry_rule", "since 0.1.0.dev27 (RVI-03 F3)"),
            "release.revalidate": setting(
                "release.revalidate", "since 0.1.0.dev27 (RVI-03 F3; true only in an arm)"
            ),
            "release.vectors": setting("release.vectors", "since 0.1.0.dev27 (RVI-03 F2)"),
            "release.vector_moves": setting(
                "release.vector_moves", "since 0.1.0.dev27 (FIX-V9-CHURN)"
            ),
            "release.acl_moves": setting(
                "release.acl_moves",
                "since 0.1.0.dev42 (U32); needs Release.hidden from the state store, held "
                "otherwise",
            ),
            "cascade.evidence_window": setting(
                "cascade.evidence_window", "since 0.1.0.dev17 (N6-2)"
            ),
            "cascade.consistency": setting("cascade.consistency", "since 0.1.0.dev18 (U22)"),
            "cascade.scope_top1": setting(
                "cascade.scope_top1", "since 0.1.0.dev42 (U32); first adopted in U30"
            ),
            "profile.*.answered": setting(
                "profile.*.answered", "0.1.0.dev20 reverted the 0.6 of 0.1.0.dev19 (U20 §8)"
            ),
            "models.generation_provider": setting(
                "models.generation_provider", "since 0.1.0.dev16 (U21)"
            ),
        },
        "experiment_arms": {
            "unknown_revision": cap("EXPERIMENTAL_ARM", "N2 keep_both/tentative; REJECT"),
            "conflict_partial_rewrite": cap("EXPERIMENTAL_ARM", "N4 conflict_partial; REJECT"),
            "writer_schema": cap("EXPERIMENTAL_ARM", "N5 W1/W2; REJECT"),
            "writer_contract_form_v1": cap("EXPERIMENTAL_ARM", "N6-2; dropped in development"),
            "scope_top1_entity_tie_v1": cap("EXPERIMENTAL_ARM", "U30 S2; below its rule"),
        },
        "release_checks": {
            "authorize": cap(
                "ADOPTED", "revocation and permission read again at the release instant, every mode"
            ),
            "acl_epoch": cap(
                "TESTED_REAL_BACKEND",
                "current and as_of_snapshot hold when the tenant ACL epoch moved, "
                "latest_at_release retries (0.1.0.dev11); scoped_dependency uses "
                "release.acl_moves",
                scope=suite,
            ),
        },
        "catalogs": {
            "resident": cap("ADOPTED", "default: parsed revisions kept in memory, reused"),
            "columnar": cap(
                "TESTED_OFFLINE",
                "opt-in catalog = columnar (N1F-6): fixed-width columns, documents read from the "
                "state store when a query needs them; rebuilt per document generation",
            ),
        },
        "state_stores": {
            "sqlite": cap("TESTED_REAL_BACKEND", "RVG test suite"),
            "postgres": cap("TESTED_REAL_BACKEND", "RVG test suite with a PostgreSQL server"),
        },
        "vector_indexes": {
            "local_exact": cap("ADOPTED", "default; RVG test suite, N0 replay"),
            "local_exact_base_delta": cap(
                "ADOPTED",
                "default layout with a matrix directory since 0.1.0.dev41 (FIX-MATRIX-GEN)",
            ),
            "local_exact_incremental": cap(
                "EXPERIMENTAL", "opt-in layouts segments, paged (N1F-2)"
            ),
            "local_exact_candidate_stage": cap(
                "ADOPTED_OPT_IN",
                "TurboQuant 4-bit candidate stage in front of local exact (N1F-4; "
                "vectors.candidate_stage = turboquant_mse4, codes built explicitly with "
                "rvg.build_candidate_codes); its search "
                "precision is APPROXIMATE",
            ),
            "external_float16": cap(
                "TESTED_OFFLINE",
                "opt-in read-only float16 vector set ranked in place (N1F-5; vectors.source = "
                "external_float16, vectors.external); precision EXACT_NATIVE, or APPROXIMATE "
                "with the candidate stage; optional query projection truncate_l2 (E9); sets "
                "written by rvg.write_external_set or rvg.export_external_set "
                "(docs/external-vectors.md)",
            ),
            "pgvector": cap("TESTED_REAL_BACKEND", "RVG test suite with pgvector"),
            "qdrant": cap("TESTED_REAL_BACKEND", "RVG test suite with Qdrant (REST)"),
            "approximate": cap("NOT_IMPLEMENTED", "no ANN index"),
        },
        "decision_providers": {
            "openrouter_jev": cap("TESTED_LIVE", before, scope=live),
            "vercel_gateway_jev": cap("TESTED_OFFLINE", "wire-format tests"),
            "typesafe_direct_jev": cap("TESTED_OFFLINE", "wire-format tests"),
            "mock": cap("TESTED_OFFLINE", "RVG test suite"),
            "local_judge": cap("NOT_IMPLEMENTED", "no local semantic judge"),
        },
        "generation_providers": {
            "openrouter": cap("TESTED_LIVE", before, scope=live),
            "claude_code": cap(
                "ADOPTED",
                "default writer and extractor since 0.1.0.dev16 (U21): the subscription Claude "
                "Code CLI with isolated flags; results returned once printed and recurring calls "
                "given a pre-started process since 0.1.0.dev44 (PERF-1, "
                "JsonTransport(claude_code_prewarm=...), 0 = off)",
                scope="research repository live runs with this package (U21, PERF-1)",
            ),
            "mock": cap("TESTED_OFFLINE", "RVG test suite"),
        },
        "embedding_providers": {
            "openrouter": cap("TESTED_LIVE", before, scope=live),
            "declared_dimensions": cap(
                "NOT_IMPLEMENTED",
                "the policy's models.embedding_dimensions (3,072) is a declaration no code reads; "
                "the space is the configured provider's, 1,536 dimensions in every measurement "
                "(D-DIM-DECL), and Settings.embedding_dimensions defaults to 1,536 since 0.1.0; "
                "since 0.1.0.dev56 RuntimeSession.open and the CLI log a note when the two differ, "
                "nothing else changes",
            ),
            "mock": cap("TESTED_OFFLINE", "RVG test suite"),
        },
        "record_replay": cap("TESTED_OFFLINE", "RVG test suite; N0 tape replay 120/120"),
        "migration": cap(
            "TESTED_REAL_BACKEND",
            "rvg.storage.migration: online move and rollback between deployments (PORT-3)",
        ),
        "testing": {
            "records": cap("TESTED_OFFLINE", "rvg.testing.records: request recording harness"),
            "fake_openrouter": cap("TESTED_OFFLINE", "rvg.testing.fake_openrouter: test double"),
        },
        "session_operations": {
            "capabilities": cap("IMPLEMENTED", "static table; no connection", scope=suite),
            "describe_runtime": cap("IMPLEMENTED", "RuntimeSession.describe_runtime", scope=suite),
            "ingest": cap("TESTED_REAL_BACKEND", "RuntimeSession.ingest", scope=suite),
            "apply_event": cap("TESTED_REAL_BACKEND", "RuntimeSession.apply_event", scope=suite),
            "answer": cap(
                "IMPLEMENTED",
                "RuntimeSession.answer delegates to UniversalEngine.answer",
                scope=suite + " (fake provider); engine evidence as above",
            ),
            "health": cap("TESTED_REAL_BACKEND", "RuntimeSession.health", scope=suite),
            "export_artifact_lineage": cap(
                "IMPLEMENTED",
                "released evidence recorded by the session, current permissions applied",
                scope=suite,
            ),
            "close": cap("IMPLEMENTED", "RuntimeSession.aclose", scope=suite),
            "fixed_draft_verification": cap("NOT_IMPLEMENTED", "no public verifier-only operation"),
        },
        "cache": {
            "decision": cap(
                "ADOPTED_OPT_IN",
                "opt-in decision_cache=CacheMode(...) (UniversalEngine, RuntimeSession; code "
                "0.1.0.dev57; records and looks up by default since 0.1.0): a decision already "
                "made is reused for a question whose evidence "
                "window shares the top document or overlaps and that the judge (Jev) calls the "
                "same question; no writer or extractor call, a released draft judged again for "
                "the new question on the current window, a hold kept on the same window; a cited "
                "unit outside the current window is never reused; a release from the cache says "
                "so in its release record (source.cache). Adopted opt-in by the user's decision "
                "(2026-10-27) although the parent contracts PROD-2 and PROD-3 ended REJECT "
                "(rule 4' (b) techqa, rule 5 by 0.04 points): hits keep the original decision "
                "(99.5%), no other question's answer released (0/13,000, 0/6,500 at theta 0.95)",
                scope=suite + "; parent PROD-2/PROD-3 live-judge runs over 16,294 paraphrases",
            ),
        },
        "handoff": {
            "packet": cap(
                "ADOPTED_OPT_IN",
                "opt-in handoff_packet=True (UniversalEngine, RuntimeSession, CLI "
                "--handoff-packet; 0.1.0.dev56, parent PROD-1 ADOPT): result.handoff (handoff-v1) "
                "on a result not released, never on a released one nor in a release record; unit "
                "text only when the requester may see the unit now; replay of 3,080 recorded "
                "requests byte-identical with it on, 1,489 packets without a leak; flags marked "
                "low confidence (read precision 14.5%)",
                scope=suite + "; parent PROD-1 replays and readings",
            ),
        },
        "events": {
            "revision": cap("IMPLEMENTED", "a new immutable revision", scope=suite),
            "revoke": cap("IMPLEMENTED", "every revision of a document", scope=suite),
            "access_replace": cap(
                "TESTED_REAL_BACKEND",
                "whole-document ACL in one transaction (SQLite, PostgreSQL)",
                scope=suite,
            ),
            "event_id_idempotency": cap(
                "TESTED_REAL_BACKEND",
                "(tenant, issuer, event_id) ledger with a payload digest: the same key and "
                "payload replays the recorded result, another payload is event_id_conflict "
                "(RVI-07A; SQLite, PostgreSQL)",
                scope=suite,
            ),
            "atomic_revision_and_access": cap(
                "TESTED_REAL_BACKEND",
                "an item's ACL, its revision or revocation, the outbox and the ledger row in one "
                "transaction (RVI-07A; SQLite, PostgreSQL; fault-injection tested)",
                scope=suite,
            ),
        },
        "access": {
            "allow_deny_groups": cap("ADOPTED", "N3 ACL; deny wins; groups via TrustedPrincipal"),
            "empty_allow_list": cap(
                "TESTED_REAL_BACKEND",
                "allow=() is an explicit deny-all, stored as one allow entry for a reserved name "
                "no principal may hold (RVI-07A)",
                scope=suite,
            ),
            "selected_document_ids": cap(
                "ADOPTED", "a user's document selection narrows retrieval (allowed_ids)"
            ),
            "excluded_document_ids": cap(
                "TESTED_OFFLINE",
                "a deny list removes ids from retrieval and from the release check's retrievable "
                "set (TemporalRetriever(excluded_ids=...), N1F-6); on an unchanging knowledge "
                "base the same decisions as the allow list of every other id",
            ),
            "conversation_turns": cap("NOT_IMPLEMENTED", "no multi-turn context input"),
            "answer_fields": cap("NOT_IMPLEMENTED", "no structured output-field request"),
        },
        "validity": {
            "supersession_plane": cap(
                "TESTED_REAL_BACKEND",
                "RVI-02: the validity gate applies higher revisions and supersedes declarations "
                "of every revision in the tenant's state to the retrieved documents — hidden, "
                "revoked, unselected or unretrieved ones included — by code rules on metadata; "
                "their text is never shown and public outputs do not name them",
                scope=suite,
            ),
            "text_only_supersession_of_hidden_documents": cap(
                "NOT_IMPLEMENTED",
                "a replacement stated only in a hidden document's text (no supersedes field) is "
                "not detected: that would mean showing hidden text to a model",
            ),
        },
        "observability": {
            "health_liveness": cap("IMPLEMENTED", "local state and package resources", scope=suite),
            "health_readiness": cap(
                "TESTED_REAL_BACKEND", "bounded read-only backend reads", scope=suite
            ),
            "lineage_request": cap(
                "IMPLEMENTED",
                "session memory of released evidence; not persisted across sessions",
                scope=suite,
            ),
            "lineage_operator_audit": cap("NOT_IMPLEMENTED", "no privileged audit export"),
            "derived_artifact_lineage": cap(
                "NOT_IMPLEMENTED", "the runtime builds no graph or summary artifacts"
            ),
        },
        "optional_dependencies": {
            "postgres": importlib.util.find_spec("psycopg") is not None,
        },
    }
