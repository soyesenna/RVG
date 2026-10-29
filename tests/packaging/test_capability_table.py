"""FIX-CAPS (private research repository docs/rvi/FIX_CAPS_PROTOCOL.md): `rvg.capabilities()` and
docs/capabilities.md say the same thing, in the English text and in its Korean original
(docs/capabilities.ko.md). Every entry of the code table (anything with a status) answers to one
row of each document, named here; the status words of a row (outside parentheses) are exactly the
statuses of the entries that answer to it; every row with a status word is answered by some entry
(rows of things not in this build carry no status word: not in this build / 미반영, REJECT,
INCONCLUSIVE); an adopted policy key's value is the policy file's value and is written in its row."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import rvg

DOCS = Path(__file__).resolve().parents[2] / "docs"
WORDS = (
    "ADOPTED_OPT_IN",
    "ADOPTED",
    "IMPLEMENTED",
    "TESTED_OFFLINE",
    "TESTED_REAL_BACKEND",
    "TESTED_LIVE",
    "EXPERIMENTAL_ARM",
    "EXPERIMENTAL",
    "NOT_IMPLEMENTED",
)

# code entry -> the start of the first cell of its row in docs/capabilities.ko.md
ROWS_KO = {
    ("release_modes", "current"): "공개 모드 `current`",
    ("release_modes", "as_of_snapshot"): "공개 모드 `as_of_snapshot`",
    ("release_modes", "latest_at_release"): "공개 모드 `as_of_snapshot`",
    ("release_modes", "scoped_dependency"): "공개 모드 `scoped_dependency`",
    ("answer_path",): "`UniversalEngine.answer`",
    ("policy_defaults", "release.mode"): "공개 모드 `scoped_dependency`",
    ("policy_defaults", "release.retry"): "공개 모드 `scoped_dependency`",
    ("policy_defaults", "release.retry_rule"): "`release.retry_rule`",
    ("policy_defaults", "release.revalidate"): "`release.retry_rule`",
    ("policy_defaults", "release.vectors"): "`release.vectors`",
    ("policy_defaults", "release.vector_moves"): "`release.vector_moves`",
    ("policy_defaults", "release.acl_moves"): "`release.acl_moves`",
    ("policy_defaults", "cascade.evidence_window"): "작성 근거 창",
    ("policy_defaults", "cascade.consistency"): "둘째 단계 일치 규칙",
    ("policy_defaults", "cascade.scope_top1"): "U30 범위 결정 1위 제외 두 초안",
    ("policy_defaults", "profile.*.answered"): "통제기 프로필 `answered`",
    ("policy_defaults", "models.generation_provider"): "Claude Code CLI 작성기",
    ("experiment_arms", "unknown_revision"): "N2 `unknown_revision`",
    ("experiment_arms", "conflict_partial_rewrite"): "N4 `conflict_partial=rewrite`",
    ("experiment_arms", "writer_schema"): "N4 `conflict_partial=rewrite`",
    ("experiment_arms", "writer_contract_form_v1"): "답 형태 작성 계약",
    ("experiment_arms", "scope_top1_entity_tie_v1"): "U30 범위 결정 1위 제외 동률 규칙",
    ("release_checks", "authorize"): "공개 직전 회수·권한 확인",
    ("release_checks", "acl_epoch"): "공개 직전 tenant ACL epoch 확인",
    ("catalogs", "resident"): "상주 카탈로그",
    ("catalogs", "columnar"): "열 기반 카탈로그",
    ("access", "excluded_document_ids"): "거부 목록",
    ("state_stores", "sqlite"): "SQLite 상태 저장소",
    ("state_stores", "postgres"): "PostgreSQL 상태 저장소",
    ("vector_indexes", "local_exact"): "local exact 색인 layout `rebuild`",
    ("vector_indexes", "local_exact_base_delta"): "local exact 색인 layout `base_delta`",
    ("vector_indexes", "local_exact_incremental"): "local exact 증분 layout",
    ("vector_indexes", "local_exact_candidate_stage"): "local exact 앞 TurboQuant",
    ("vector_indexes", "external_float16"): "외부 float16 벡터 묶음",
    ("vector_indexes", "pgvector"): "pgvector 색인",
    ("vector_indexes", "qdrant"): "Qdrant 색인",
    ("vector_indexes", "approximate"): "근사 검색",
    ("migration",): "배포 간 온라인 이전",
    ("decision_providers", "openrouter_jev"): "OpenRouter Jev 판정",
    ("decision_providers", "vercel_gateway_jev"): "Vercel AI Gateway Jev",
    ("decision_providers", "typesafe_direct_jev"): "Vercel AI Gateway Jev",
    ("decision_providers", "mock"): "mock 판정",
    ("decision_providers", "local_judge"): "local semantic judge",
    ("generation_providers", "openrouter"): "OpenRouter 작성기·추출기",
    ("generation_providers", "claude_code"): "Claude Code CLI 작성기",
    ("generation_providers", "mock"): "mock 판정",
    ("embedding_providers", "openrouter"): "OpenRouter 작성기·추출기",
    ("embedding_providers", "mock"): "mock 판정",
    ("embedding_providers", "declared_dimensions"): "정책 선언 `models.embedding_dimensions`",
    ("handoff", "packet"): "넘김 패킷",
    ("cache", "decision"): "결정 캐시",
    ("record_replay",): "`RecordingTransport`",
    ("testing", "records"): "`rvg.testing.records`",
    ("testing", "fake_openrouter"): "`rvg.testing.fake_openrouter`",
    ("session_operations", "capabilities"): "`capabilities`·`describe_runtime`·`close`",
    ("session_operations", "describe_runtime"): "`capabilities`·`describe_runtime`·`close`",
    ("session_operations", "close"): "`capabilities`·`describe_runtime`·`close`",
    ("session_operations", "ingest"): "`ingest`·`apply_event`",
    ("session_operations", "apply_event"): "`ingest`·`apply_event`",
    ("session_operations", "answer"): "`answer`(엔진 결과",
    ("session_operations", "health"): "`health` liveness",
    ("session_operations", "export_artifact_lineage"): "`export_artifact_lineage`",
    ("session_operations", "fixed_draft_verification"): "고정 초안 verifier-only",
    ("events", "revision"): "사건 `revision`",
    ("events", "revoke"): "사건 `revision`",
    ("events", "access_replace"): "원자적 ACL 교체",
    ("events", "event_id_idempotency"): "사건 ID 멱등성",
    ("events", "atomic_revision_and_access"): "항목 단위 원자성",
    ("access", "allow_deny_groups"): "ACL allow/deny",
    ("access", "empty_allow_list"): "아무도 허용하지 않는 ACL",
    ("access", "selected_document_ids"): "사용자 문서 선택",
    ("access", "conversation_turns"): "대화 turn",
    ("access", "answer_fields"): "대화 turn",
    ("validity", "supersession_plane"): "유효성 평면",
    ("validity", "text_only_supersession_of_hidden_documents"): "숨은 문서 본문에만",
    ("observability", "health_liveness"): "`health` liveness",
    ("observability", "health_readiness"): "`health` liveness",
    ("observability", "lineage_request"): "`export_artifact_lineage`",
    ("observability", "lineage_operator_audit"): "운영자 감사 lineage",
    ("observability", "derived_artifact_lineage"): "운영자 감사 lineage",
}


# code entry -> the start of the first cell of its row in docs/capabilities.md
ROWS_EN = {
    ("release_modes", "current"): "Release mode `current`",
    ("release_modes", "as_of_snapshot"): "Release modes `as_of_snapshot`",
    ("release_modes", "latest_at_release"): "Release modes `as_of_snapshot`",
    ("release_modes", "scoped_dependency"): "Release mode `scoped_dependency`",
    ("answer_path",): "`UniversalEngine.answer`",
    ("policy_defaults", "release.mode"): "Release mode `scoped_dependency`",
    ("policy_defaults", "release.retry"): "Release mode `scoped_dependency`",
    ("policy_defaults", "release.retry_rule"): "`release.retry_rule`",
    ("policy_defaults", "release.revalidate"): "`release.retry_rule`",
    ("policy_defaults", "release.vectors"): "`release.vectors`",
    ("policy_defaults", "release.vector_moves"): "`release.vector_moves`",
    ("policy_defaults", "release.acl_moves"): "`release.acl_moves`",
    ("policy_defaults", "cascade.evidence_window"): "Writing evidence window",
    ("policy_defaults", "cascade.consistency"): "Second-stage consistency rule",
    ("policy_defaults", "cascade.scope_top1"): "U30 two drafts excluding",
    ("policy_defaults", "profile.*.answered"): "Controller profile `answered`",
    ("policy_defaults", "models.generation_provider"): "Claude Code CLI writer",
    ("experiment_arms", "unknown_revision"): "N2 `unknown_revision`",
    ("experiment_arms", "conflict_partial_rewrite"): "N4 `conflict_partial=rewrite`",
    ("experiment_arms", "writer_schema"): "N4 `conflict_partial=rewrite`",
    ("experiment_arms", "writer_contract_form_v1"): "Answer-form writing contract",
    ("experiment_arms", "scope_top1_entity_tie_v1"): "U30 tie rule excluding",
    ("release_checks", "authorize"): "Revocation and permission check right before release",
    ("release_checks", "acl_epoch"): "Tenant ACL epoch check right before release",
    ("catalogs", "resident"): "Resident catalog",
    ("catalogs", "columnar"): "Columnar catalog",
    ("access", "excluded_document_ids"): "Exclusion list",
    ("state_stores", "sqlite"): "SQLite state store",
    ("state_stores", "postgres"): "PostgreSQL state store",
    ("vector_indexes", "local_exact"): "local exact index layout `rebuild`",
    ("vector_indexes", "local_exact_base_delta"): "local exact index layout `base_delta`",
    ("vector_indexes", "local_exact_incremental"): "local exact incremental layout",
    ("vector_indexes", "local_exact_candidate_stage"): "TurboQuant 4-bit candidate stage",
    ("vector_indexes", "external_float16"): "External float16 vector set",
    ("vector_indexes", "pgvector"): "pgvector index",
    ("vector_indexes", "qdrant"): "Qdrant index",
    ("vector_indexes", "approximate"): "Approximate search",
    ("migration",): "Online move and rollback between deployments",
    ("decision_providers", "openrouter_jev"): "OpenRouter Jev judgement",
    ("decision_providers", "vercel_gateway_jev"): "Vercel AI Gateway Jev",
    ("decision_providers", "typesafe_direct_jev"): "Vercel AI Gateway Jev",
    ("decision_providers", "mock"): "mock judgement",
    ("decision_providers", "local_judge"): "local semantic judge",
    ("generation_providers", "openrouter"): "OpenRouter writer and extractor",
    ("generation_providers", "claude_code"): "Claude Code CLI writer",
    ("generation_providers", "mock"): "mock judgement",
    ("embedding_providers", "openrouter"): "OpenRouter writer and extractor",
    ("embedding_providers", "mock"): "mock judgement",
    (
        "embedding_providers",
        "declared_dimensions",
    ): "Policy declaration `models.embedding_dimensions`",
    ("handoff", "packet"): "Handoff packet",
    ("cache", "decision"): "Decision cache",
    ("record_replay",): "`RecordingTransport`",
    ("testing", "records"): "`rvg.testing.records`",
    ("testing", "fake_openrouter"): "`rvg.testing.fake_openrouter`",
    ("session_operations", "capabilities"): "`capabilities`, `describe_runtime`, `close`",
    ("session_operations", "describe_runtime"): "`capabilities`, `describe_runtime`, `close`",
    ("session_operations", "close"): "`capabilities`, `describe_runtime`, `close`",
    ("session_operations", "ingest"): "`ingest`, `apply_event`",
    ("session_operations", "apply_event"): "`ingest`, `apply_event`",
    ("session_operations", "answer"): "`answer` (the envelope",
    ("session_operations", "health"): "`health` liveness",
    ("session_operations", "export_artifact_lineage"): "`export_artifact_lineage`",
    ("session_operations", "fixed_draft_verification"): "Fixed-draft verifier-only",
    ("events", "revision"): "Events `revision`",
    ("events", "revoke"): "Events `revision`",
    ("events", "access_replace"): "Atomic ACL replacement",
    ("events", "event_id_idempotency"): "Event ID idempotency",
    ("events", "atomic_revision_and_access"): "Per-item atomicity",
    ("access", "allow_deny_groups"): "ACL allow/deny",
    ("access", "empty_allow_list"): "ACL that allows nobody",
    ("access", "selected_document_ids"): "User document selection",
    ("access", "conversation_turns"): "Conversation turns",
    ("access", "answer_fields"): "Conversation turns",
    ("validity", "supersession_plane"): "Validity plane",
    (
        "validity",
        "text_only_supersession_of_hidden_documents",
    ): "Replacement written only in the text",
    ("observability", "health_liveness"): "`health` liveness",
    ("observability", "health_readiness"): "`health` liveness",
    ("observability", "lineage_request"): "`export_artifact_lineage`",
    ("observability", "lineage_operator_audit"): "Operator audit lineage",
    ("observability", "derived_artifact_lineage"): "Operator audit lineage",
}

# document, row names, the header of its first column, the start of the policy hash row
LANGUAGES = {
    "en": ("capabilities.md", ROWS_EN, "Component", "Policy `universal-v1`"),
    "ko": ("capabilities.ko.md", ROWS_KO, "구성 요소", "정책 `universal-v1`"),
}


def doc_rows(name, header):
    """(first cell, status cell, whole row) of every table row of the document."""
    rows = []
    for line in (DOCS / name).read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == header:
            continue
        rows.append((cells[0], cells[1], line))
    return rows


def words(status_cell):
    """The status words of a cell outside its parentheses."""
    bare = re.sub(r"\([^)]*\)", "", status_cell)
    return {w for w in re.findall(r"[A-Z_]+", bare) if w in WORDS}


def entries(table, path=()):
    """Every entry of the code table that carries a status, by its path."""
    for key, value in table.items():
        if isinstance(value, dict) and "status" in value:
            yield (*path, key), value
        elif isinstance(value, dict):
            yield from entries(value, (*path, key))


def scalars(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from scalars(v)
    else:
        yield str(value).lower()


@pytest.mark.parametrize("language", sorted(LANGUAGES))
def test_the_code_table_and_the_document_say_the_same(language):
    name, names, header, policy_row = LANGUAGES[language]
    caps = rvg.capabilities()
    rows = doc_rows(name, header)
    code = dict(entries(caps))
    assert set(code) == set(names), "every code entry is named here and nothing else"

    answered = {}
    for path, piece in names.items():
        found = [r for r in rows if r[0].startswith(piece)]
        assert len(found) == 1, (path, piece, len(found))
        answered.setdefault(found[0], []).append(path)

    for (first, status, line), paths in answered.items():
        assert words(status) == {code[p]["status"] for p in paths}, (first, status, paths)
        for path in paths:
            if path[0] == "policy_defaults":
                for value in scalars(code[path]["value"]):
                    assert value in line.lower(), (path, value)

    for first, status, line in rows:
        if words(status) and (first, status, line) not in answered:
            assert policy_row in first, f"row without a code entry: {first}"
            assert caps["policy"]["sha256"][:8] in line


def test_an_adopted_policy_key_reads_the_policy_file():
    caps = rvg.capabilities()
    policy = rvg.policy()
    assert caps["policy_defaults"]["release.acl_moves"]["value"] == policy["release"]["acl_moves"]
    assert (
        caps["policy_defaults"]["profile.*.answered"]["value"] == policy["profile"]["*"]["answered"]
    )
    assert all(e["default"] for e in caps["policy_defaults"].values())
    opt_in = caps["vector_indexes"]["local_exact_candidate_stage"]
    assert opt_in["status"] == "ADOPTED_OPT_IN" and not opt_in["default"]
    assert all(e["experimental"] for e in caps["experiment_arms"].values())
