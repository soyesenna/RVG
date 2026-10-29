# RVG 공개 runtime API (`rvg-runtime-api/0.1`)

외부 애플리케이션·연구 실행기·benchmark 제출 adapter가 쓰는 공개 경로다. 모든 이름은 `import rvg`의 공개 이름(`rvg.__all__`)이다. 내부 모듈(`rvg.control`, `rvg.temporal.gate`, `rvg.storage.*` 등)을 직접 부르지 않는다. 기계가독 대응표는 릴리스마다 인계 디렉터리의 `rvg_public_api_map.json`(형식: KB-FF `kbff-rvg-handoff/1.0`의 `api_mapping`)으로 나간다. 그 파일은 `scripts/make_api_map.py`가 만든다.

이 파일은 한국어 원본이다. 정본은 영어판 [PUBLIC_API.md](PUBLIC_API.md)이고, 두 판의 수치는 같다(`scripts/check_doc_numbers.py`).

**이 API는 기존 객체를 조립·위임할 뿐이다.** `answer`는 `UniversalEngine.answer`를 한 번 부르고 결과를 바꾸지 않는다. 검색·재작성·재검증·상태 상향·다른 provider fallback은 없다. 출력 JSON schema는 `rvg/resources/schemas/api/*.schema.json`이다.

## 수명과 구성

```python
import rvg

async with rvg.open_session(storage, settings, root=..., environ=..., transport=..., providers=...) as session:
    ...
```

- `rvg.open_session(storage, settings=None, *, root=None, environ=None, transport=None, providers=None, max_documents=24)`: **sync**로 session을 만든다. 사용은 **async context manager**로 한다.
  - `storage`: 열린 `rvg.Storage`/`rvg.TemporalStore`(호출자 소유, session이 닫지 않음), 또는 저장소 구성(`storage.json` 경로·dict·`rvg.StorageConfig`)이다. 구성을 주면 session이 `rvg.open_storage`로 열고 닫는다.
  - `settings`: `rvg.Settings`. 호출자가 자기 설정에서 `rvg.Settings(...)`(인자·환경변수만 읽고 `.env` 파일은 읽지 않음)를 만들어 넘기는 것이 기본 사용법이다. 인자는 필드 이름이나 별칭(`rvg_mode`)만 받고, 대소문자는 가리지 않는다. 그 밖의 이름은 `ConfigurationError(CONFIG_UNKNOWN_SETTING)`이다(0.1.0; 전에는 조용히 무시해 `Settings(rvg_mode="live")`가 mock으로 남았다). 환경변수·`.env`의 무관한 이름은 무시한다. 빼면 `rvg.load_settings(root 또는 CWD)` — 그 위치의 `.env`를 읽는 단독 실행용 편의 동작이다. provider는 `providers`(호출자가 만든 객체 dict: `decision`, `writer`, `extractor`, `embedding`)가 있으면 그대로 쓴다. 없으면 제품 factory(`decision_provider` 등)로 `transport` 위에 만든다. `transport`도 없으면 session이 `httpx.AsyncClient(trust_env=False)`를 만들어 소유한다.
  - 저장소 구성은 DSN을 **환경변수 이름**으로만 가리킨다(`dsn_env`, `url_env`). `environ`으로 값을 넘길 수 있다.
- `handoff_packet=False`(dev56, opt-in, `open_session`·`RuntimeSession`의 키워드): 켜면 이 session이 공개하지 않은 답마다 `result.handoff`를 붙인다(아래 **넘김 패킷**). 기본은 꺼짐이고 설정·정책 해시에 들어가지 않는다.
- `await session.aclose()`: session이 연 저장소와 HTTP client만 닫는다. 여러 번 불러도 되고, 이후 동작은 `OperationError(NOT_READY, session_closed)`다.
- 요청 문맥: `rvg.RequestContext(tenant, principal=None, request_id=None, selected_document_ids=None, call=None)`.
  - `principal`은 인증 계층이 만든 `rvg.TrustedPrincipal(principal_id, groups=())` **객체**만 받는다. 문자열은 `FORBIDDEN_INPUT`이다. `None`이면 ACL 항목이 없는 문서만 보인다.
  - `selected_document_ids`는 실제 사용자가 고른 문서로 **검색 후보만 줄인다**(기존 `allowed_ids`). 질문 범위 맥락으로 해석하지 않는다.
  - `call`은 요청 예산·마감(`rvg.CallContext`)이고, 기본은 `session.new_call(deadline_seconds=None)`(= `Settings.context()`)이다.

## 동작

| 중립 동작 | 공개 경로 | 방식 | 입력 | 출력(schema) | 오류 |
|---|---|---|---|---|---|
| capabilities | `rvg.capabilities()` / `RuntimeSession.capabilities()` | sync | 없음 / session | 정적 표(+`session` 구성). 항목마다 `status`·`evidence`·`implemented`·`evidence_level`·`scope`·`default`·`experimental` | 없음(연결을 열지 않음) |
| describe_runtime | `RuntimeSession.describe_runtime()` | sync | session | `runtime_description` | 없음 |
| ingest | `RuntimeSession.ingest(documents, context, *, prepare_vectors=True)` | async | `TemporalDocument`·dict·`IngestItem(document, access)` 목록 | `ingest_result` | 문서별 `REJECTED`/`ERROR` 행, 문맥 오류는 `OperationError` |
| apply_event | `RuntimeSession.apply_event(event, context, *, prepare_vectors=True)` | async | `RevisionEvent`·`RevokeEvent`·`AccessEvent` 또는 dict(`type`: revision/revoke/access) | `event_result` | `OperationError`(`INVALID_INPUT`, `FORBIDDEN_INPUT`, `UNSUPPORTED_CAPABILITY`) |
| answer | `RuntimeSession.answer(request, context)` | async | `TemporalQuestion` 또는 dict(`text`, `as_of`, `known_at`, `locale`) | `answer_envelope`(`result`는 엔진 dict 그대로) | 엔진 안 실패는 `result.status = ERROR`. 입력·구성 오류는 `OperationError` |
| health | `RuntimeSession.health(level="liveness", context=None)` | async | `liveness` 또는 `readiness`(tenant 필요) | `health` | `OperationError(INVALID_INPUT)` |
| export_artifact_lineage | `RuntimeSession.export_artifact_lineage(reference, context)` | async | request_id(또는 `{"request_id": ...}`) | `lineage` | 모르는 참조·다른 tenant·다른 principal은 모두 `lineage_reference_unknown` |
| close | `RuntimeSession.aclose()` | async | session | 없음 | 없음 |
| fixed_draft_verify | 없음 | — | — | — | **UNSUPPORTED**: 공개 검증기 단독 동작 없음 |

`OperationError`는 `code`(`UNSUPPORTED_CAPABILITY`·`INVALID_INPUT`·`FORBIDDEN_INPUT`·`NOT_READY`·`BACKEND_ERROR`·`PROVIDER_ERROR`·`TIMEOUT`·`INTERNAL_ERROR`)와 `native_code`(규칙·실패 타입 이름)를 갖고 `to_dict()`로 직렬화한다(`operation_error` schema). 메시지에 요청·문서 본문·비밀·DSN을 넣지 않는다.

## 의미

**ingest.** 항목마다(그 ACL과 revision을 함께) **한 트랜잭션**으로 저장한다(RVI-07A, dev3부터: 중간 실패는 ACL도 바꾸지 않는다). 결과는 `STORED`(새 사건), `DUPLICATE`(같은 bytes가 이미 있음), `REJECTED`(규칙이 거부했고 아무것도 바뀌지 않음; 예 `immutable_temporal_revision_changed`, `…:observation_before_publication`), `ERROR`(backend 실패, 타입 이름만)다. `IngestItem.access`가 주어지면 같은 트랜잭션에서 그 문서의 ACL을 바꾼다(`None`: 비움, `DocumentAccess(allow, deny)`: 교체, `DocumentAccess(allow=())`: 명시적 deny-all, 기본 `"unchanged"`: 손대지 않음). 불변 위반은 무엇이든 쓰기 전에 거부한다. backend 실패는 그 항목의 `ERROR`이며 트랜잭션 전체가 되돌려진다. `prepare_vectors=True`면 tenant에서 벡터가 없는 revision을 임베딩한다(기존 `TemporalRetriever.index`).

준비 상태는 셋으로 따로 낸다. 각각 `state`(READY/NOT_READY/UNKNOWN/NOT_APPLICABLE), `observed_at`(session이 확인한 시각), `reason`을 갖는다.

- `durable_commit`: 상태 저장소 트랜잭션이 끝났다.
- `embedding_ready`: 저장한 revision의 임베딩 입력이 모두 벡터로 저장됐다. 아니면 `vectors_pending:N`.
- `index_visible`: 다음 질의의 색인 view가 커밋된 벡터·세대로 만들어진다(`query_time_view:<index>`). 벡터가 없는 revision은 BM25 pending 경로로 검색된다(`lexical_fallback`).

runtime은 원문 commit 시각을 따로 기록하지 않는다. 사건 시각은 호출하는 쪽이 잰다.

**apply_event.** `revision`은 ingest와 같다. `revoke`는 그 문서의 모든 revision을 회수한다(이미 회수면 `DUPLICATE`). `access`는 문서의 ACL 전체를 한 트랜잭션에서 바꾼다(`SQLiteState/PostgresState.replace_access`). 바뀐 항목마다 ACL epoch가 1씩 오르고, `set_access`와 같은 outbox 항목이 남아 배포 이전 catch-up이 그대로 재생한다. 이미 같은 ACL이면 아무것도 바꾸지 않는다(`DUPLICATE`). `event_id`가 있는 사건은 **사건 ID 원장**을 지난다(RVI-07A, dev3부터): key는 `(tenant, issuer, event_id)`이고 `issuer`는 신뢰된 `RequestContext.issuer`(없으면 principal ID, 그것도 없으면 `default`)다. 같은 key·같은 내용(tenant·issuer·사건의 모든 필드의 digest)이면 처음 결과를 돌려주는 `DUPLICATE`(`event_id_replayed`), 같은 key·다른 내용이면 `OperationError(INVALID_INPUT, event_id_conflict)`이고 상태는 바뀌지 않는다. 원장 확인·ACL·revision/회수·outbox·원장 행은 한 트랜잭션이다. `event_id`가 없는 사건은 예전처럼 **내용 기준** 중복만 본다. 회수·ACL 사건의 backend 실패는 `OperationError(BACKEND_ERROR)`다. ACL은 document_id 단위(모든 revision 공통)이고 deny가 allow보다 우선한다. `allow=()`는 명시적 deny-all이다: 어떤 principal도 가질 수 없는 예약 이름(`rvg-reserved:` 접두어) 하나만 허용하는 allow 항목으로 저장하므로 저장 schema가 같고 이전 판도 그 문서를 모두에게 숨긴다. 예약 접두어를 가진 principal·그룹·ACL 이름은 거부한다. 원장은 PORT-3 tenant 이전(export/import)에 아직 포함되지 않는다.

**answer.** 봉투 `{api_version, request_id, result, usage}`를 돌려준다. `result`는 `UniversalEngine.answer`의 결과 그대로다(`status` FULL/PARTIAL/ABSTAIN/UNRESOLVED_SCOPE/ERROR, `short_answer`, `sentences`, `evidence_ids`, `release`, `snapshot`, `config_hash`, `storage`, `audit`, `trace` 등).
- **공개하지 않은 결과에는 초안 값이 없다(0.1.0).** `ABSTAIN`·`UNRESOLVED_SCOPE`·`ERROR`면 `short_answer`는 `null`, `text`는 빈 문자열, `sentences`·`assessment.claims`·`assessment_before_recovery.claims`는 빈 목록, `trace.consistency.partner_short_answer`는 `null`이다. 보류한 초안은 opt-in 넘김 패킷(`result.handoff`)에만 있다.
- **`evidence_ids`는 작성기 근거 창이다.** 작성기에게 준 단위 전체이고, 인용하지 않은 단위도 들어 있다. 공개 문장이 인용한 단위는 `sentences[].evidence_ids`다.
- **`trace`는 감사용이다.** 단계별 상태·사유·단위 id, 캐시 후보와 판정 확률, 범위 결정 기록을 담는다. 공개하지 않은 결과의 trace에는 초안의 본문·주장·짧은 답이 없다. 다만 둘째 단계가 `gap`(정책 기본 아님)이면 초안의 빠진 부분으로 만든 검색어가 `trace.queries`와 `audit[].query`에 남는다. 형식은 판마다 바뀔 수 있어 안정성 약속 밖이다. `usage`는 그 요청의 호출 수·보고 비용이다. 질문 DTO는 추가 필드를 거부한다(`principal_id`, `expected_answer`, gold 표지 → `FORBIDDEN_INPUT`). 공개 모드는 정책의 기본(dev27부터 `scoped_dependency`, 그 전 `current`)뿐이고 요청으로 고를 수 없다(snapshot 모드는 공개 API에 없다). 보장 범위와 측정 수치는 [보장과 교환](../GUARANTEES.ko.md)에 있다.

**opt-in 도구의 공개 이름(0.1.0).** 문서가 쓰라고 안내하는 도구는 모두 `rvg` 최상위 이름이다. 내부 모듈 경로도 별칭으로 계속 동작하지만 안정성 약속은 공개 이름에만 있다.
- `rvg.CacheMode`, `rvg.DecisionCache`: 결정 캐시(아래).
- `rvg.write_external_set(out_dir, rows, space, *, shard_rows=65536)`, `rvg.export_external_set(state, tenant, space, out_dir)`: `vectors.source: external_float16`에 쓸 외부 float16 벡터 묶음을 만든다(형식 `docs/external-vectors.md`, 예제 `examples/storage/external_float16.py`).
- `await rvg.build_candidate_codes(storage, tenant, space, dims=None)`: 후보 단계(`vectors.candidate_stage: turboquant_mse4`)의 코드를 만든다. `space`는 `EmbeddingSpace`(차원을 거기서 읽음) 또는 공간 지문(그때는 `dims`가 필요)이다. 후보 단계가 꺼진 저장소면 `ContractError(candidate_stage_off)`다.

**명령행(0.1.0).** `rvg ingest FILE.jsonl --storage storage.json [--tenant T]`는 한 줄에 JSON 객체 하나를 읽는다. 각 줄은 문서(`TemporalDocument` 필드), `{"document": …, "access": {"allow": […], "deny": […]}}`, 또는 사건(`{"type": "revision" | "revoke" | "access", …}`, `apply_event`와 같은 형식)이다. 줄 순서대로 적용한 뒤 벡터가 없는 revision을 임베딩한다(`--no-vectors`면 하지 않음). 결과는 항목별 개수와 거부된 줄(줄 번호와 사유만, 내용은 내지 않음)이다. `rvg universal-query … --principal ID --group G`(여러 번)는 그 principal로 묻는다. 명령행을 부르는 쪽(운영자)이 principal을 정할 수 있다고 믿는다는 뜻이다. 도움말은 영어다.

**넘김 패킷(dev56, opt-in, 부모 PROD-1).** `handoff_packet=True`(session, `UniversalEngine`, CLI `--handoff-packet`)면 공개하지 않은 결과(`ABSTAIN`·`UNRESOLVED_SCOPE`·`ERROR`)에 `result.handoff`(schema `handoff_packet`, `handoff-v1`)가 붙는다. 사람이 넘겨받은 요청을 판단할 재료다.
- **공개가 아니다.** `kind: handoff_draft`, `released: false`다. 공개 결과(`FULL`·`PARTIAL`)에는 붙지 않고(봉투 schema가 막는다), 공개 기록(`release`)과 lineage에도 들어가지 않는다. 옵션을 켜도 공개 결과와 공개 기록은 바이트 단위로 같고, 모델·저장소 호출도 같다(패킷을 만들 때 읽기 전용 공개 확인 한 번만 더한다).
- **담는 것.** 단계(`stage1`·`stage1_top1`·`stage2`·`partner_draft`·`conflict_rewrite`)마다 작성기 근거 창, 초안 본문과 인용, 작성기 거절 조건(`not_answerable`·`no_sentences`·`empty_short_answer`·`citation_outside_evidence`)과 원 출력, 통제기의 주장→문장 대응·라벨·확률·신호 대 문턱·상태·사유를 담는다. 그 밖에 사유 코드 목록(`reasons`), 넘기는 초안(`handed_draft`: 둘째 단계 → 첫 단계 → 1위 초안 순으로 초안이 있는 것), 둘째 단계와 짝 초안 비교(`comparison`), 근거 단위 본문과 효력 상태(`evidence`), 자동 표시(`flags`: `claim_not_supported`·`answered_below`, 문구 자동 표시·신뢰도 낮음(코드 문자열 `automatic flag, low confidence`) — 부모 PROD-1 (b)에서 판독한 표시 정밀도가 14.5%[10.3, 20.1]라 신뢰도가 낮다고 적는다)를 담는다. 전제 검사는 정책에 없어서 `premise: not_checked`다.
- **가림.** 패킷을 만드는 순간 요청자가 볼 수 있는 단위만 본문을 담는다: 문서가 공개 확인(회수 아님·ACL 허용)을 통과하고, 거부 목록에 없고, 선택이 있으면 그 안에 있고, 유효성 게이트가 그 단위를 ELIGIBLE로 본 경우다. 아니면 `{id, gate, withheld}`만 담는다. 한 단계의 작성기가 본 단위가 하나라도 가려지면 그 단계가 쓴 글(초안·주장·원 출력·짧은 답) 전부를 가린다. 공개 확인을 읽지 못하면 모든 단위를 가린다.

**결정 캐시(코드 dev57, 사용자 결정으로 opt-in 채택 dev58, 부모 PROD-2·PROD-3은 REJECT — 근거·운영점은 `docs/GUARANTEES.md` §6).** `decision_cache=rvg.CacheMode(rvg.DecisionCache(path), …)`(session, `UniversalEngine`)로 켠다(0.1.0부터 공개 이름; 내부 경로 `rvg.universal.cache`도 동작). 끝난 결정(공개·보류)을 로컬 저장소에 적재하고(`record`), 근거 창이 상위 1 문서를 공유하거나 겹치고(`match`, τ 기본 0.5) Jev가 같은 질문이라 본 새 질문에 다시 쓴다(`lookup`, θ 기본 0.9). 기본은 둘 다 켜짐이다(`record=True`·`lookup=True`, 0.1.0에서 `record` 기본을 True로 바꿨다). 조회만 하려면 `record=False`, 채우기만 하려면 `lookup=False`를 명시한다.
- **작성기를 부르지 않는다.** 공개 항목은 새 질문으로 지금 근거 창 위에서 통제기가 다시 판정한 뒤 공개 확인을 거친다(창이 다르면 주장도 다시 판정). 실패하면 평소 경로다. 보류 항목은 같은 창일 때만 같은 보류를 돌려준다.
- **동치 판정.** Jev boolean 한 문항(prompt `universal.cache.EQUIVALENCE`, 부모 EVID-8 Jev 동치 판정 변형 A와 같은 문구)의 확률이다. 정규화 뒤 같은 문장인 후보가 있으면 판정 없이 그것을 쓴다(p = 1).
- **재사용하지 않는 것.** 인용 단위가 지금 근거 창(현재 효력·권한·거부 목록·선택을 지난 검색 결과) 밖에 있는 항목(예: scope_top1 상위 1 초안으로 공개된 항목), 정책 해시·arm·tenant가 다른 항목이다.
- **짧은 답.** 캐시 공개의 짧은 답은 원 공개의 짧은 답이다(회복으로 문장을 고쳐 쓴 공개 포함).
- **기록.** 캐시에서 나간 공개는 `release.source.cache = {entry, p, theta, match}`를 남기고 lineage에도 남는다. trace `cache`에 후보와 판정 확률이 남는다. 동치 판정·재검사 호출은 tape에 들어가 재생된다.

**health.** `liveness`는 session 상태, 정책 리소스 hash, prompt manifest 검증, provider 구성만 본다. DB·네트워크·모델 호출은 없다. `readiness`는 이미 열린 상태 저장소의 세대 읽기를 5초 제한으로 한다. vector service는 읽기 전용 개수와 projection frontier를 더 본다(아직 projection이 없으면 `DEGRADED space_not_projected_yet`, frontier가 뒤처지면 `DEGRADED projection_frontier_behind`, 질의는 exact overlay로 계속된다). schema 생성·migration·행렬 warmup·provider 호출은 하지 않는다. 오류는 예외 **타입 이름**만 낸다. `provider_calls: 0`·`state_mutated: false`는 설계상의 자기보고이며, 호출자가 따로 계측해야 한다. `unverified`는 확인하지 않은 구성 요소(예: provider 도달성)를 적는다.

**유효성과 권한(RVI-02·02b, dev4·dev5부터).** 검색은 principal이 볼 수 있고 회수되지 않았으며 사용자가 고른 문서만 돌려준다(내용 평면). 유효성 게이트는 그 문서들을 대상으로 하는 **높은 revision(R-REV-1)과 `supersedes` 선언(R-REV-2)을 tenant의 권위 저장소 전체에서** 찾는다 — 보이지 않거나 회수됐거나 선택되지 않았거나 검색되지 않은 revision도 포함한다(유효성 평면). 그 revision의 metadata(대상·unit·family·게시·관측·효력 시각·권위)만 코드 규칙에 쓰고, 본문은 작성기·판정기·사용자에게 가지 않으며, 게이트 사유도 그 문서를 이름 짓지 않는다(`explicit_replacement:withheld`). 효력 시작 전이거나 `known_at` 뒤에 관측된 선언은 아직 적용하지 않고, 선언한 문서가 회수·대체돼도 옛 값을 되살리지 않는다. 명시 대체는 두 문서의 family가 달라도 적용한다(RVI-02b, KB-FF R-REV-2). 순서(선언 문서가 나중에 게시)·권위(primary는 primary만 대체)가 맞지 않는 선언은 추측하지 않고 미해결(보류)이다. 그 결과 숨은 개정이 있는 질문은 옛 값 대신 보류가 되며, 상태·사유·tenant 지식 번호의 차이는 남는 존재 신호다. 본문으로만 적힌(구조화 `supersedes` 없는) 숨은 개정은 감지하지 않는다.

**export_artifact_lineage.** 이 session이 낸 답의 **공개된 인용 unit**만 기록한다. 공개 문장의 `sentences[].evidence_ids`이고, 작성기 근거 창(`result.evidence_ids`)이 아니다(0.1.0에서 고침: 전에는 인용하지 않은 창 단위까지 기록했다). 메모리에 기본 1,024개를 두고 session이 끝나면 사라진다. export 때 **지금의** ACL·회수를 다시 적용한다. 각 unit은 `evidence_unit`(문서·revision·unit·원문 sha256·구간·unit sha256)과 `vector`(그 unit이 든 임베딩 입력 — 전체 revision 또는 K8 조각 — 의 input hash, 공간, `PRESENT/ABSENT`, `derived_from`)로 나간다. 숨겨지거나 회수된 source는 ID·제목·hash 없이 빠지고 `completeness=PARTIAL`, `missing_reasons=["ACCESS_RESTRICTED_OR_REVOKED"]`만 남는다. 공개하지 않은 답(ABSTAIN/ERROR)은 인용이 없다(`notes: ["no_released_evidence"]`). hash는 bytes 식별이지 문장 지지의 증명이 아니다. 파생 graph·요약은 이 runtime에 없다. 운영자 감사 export는 없다.

## 버전

라이브러리 버전(`rvg.__version__`), API 계약(`rvg.API_VERSION` = `rvg-runtime-api/0.1`), 관측 계약(`rvg.OBSERVATION_VERSION` = `rvg-observation/0.1`), 정책 hash(`rvg.policy_hash()`), prompt manifest hash(`describe_runtime()["prompt_manifest_sha256"]`)는 서로 다른 값이다. 추가 필드는 minor, 의미 변경은 새 계약 버전으로 한다.

## 예제

`examples/api/public_consumer.py`는 공개 이름과 시험 대역(`rvg.testing.fake_openrouter`)만으로 모든 동작을 한 번씩 수행한다(네트워크·모델 없음).
