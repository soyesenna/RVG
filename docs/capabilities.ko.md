# 지원 범위와 검증 수준

`rvg.capabilities()`(또는 `rvg capabilities`)가 같은 내용을 코드 안의 정적 표로 돌려준다. 상태 단어가 있는 이 문서의 행마다 그 표에 같은 상태의 항목이 있고, 시험(`tests/packaging/test_capability_table.py`)이 두 표를 비교한다. 정책 키 행의 값은 그 표가 정책 파일에서 읽어 함께 돌려준다(`policy_defaults`). 무엇을 보장하고 무엇을 교환하는지(측정 수치와 근거)는 [보장과 교환](GUARANTEES.ko.md)에 있다. 상태는 기록된 **가장 강한 근거**이고, 근거가 어디서 나왔는지 함께 적는다. 미구현 구성 요소는 다른 구성 요소로 대신하지 않고 `NOT_IMPLEMENTED`로 남긴다.

이 파일은 한국어 원본이다. 정본은 영어판 [capabilities.md](capabilities.md)이고, 시험은 두 판을 모두 읽으며, 두 판의 수치는 같다(`scripts/check_doc_numbers.py`).

상태 어휘: IMPLEMENTED(코드 있음) · TESTED_OFFLINE(결정적·모의·재생 시험) · TESTED_REAL_BACKEND(실제 DB 서비스에서 적합성 시험) · TESTED_LIVE(실제 모델 호출) · ADOPTED(제품 기본) · ADOPTED_OPT_IN(채택했으나 기본은 꺼짐, arm으로 켬) · EXPERIMENTAL / EXPERIMENTAL_ARM(평가 실행기에서만 켬) · NOT_IMPLEMENTED.

## 답변 경로

| 구성 요소 | 상태 | 근거 |
|---|---|---|
| `UniversalEngine.answer`(검색→유효성→범위→작성→통제→cascade 1단계) | ADOPTED | 연구 저장소 N0~N6 실험, 이관 뒤 N0 테이프 재생 120/120 |
| 정책 `universal-v1`(sha256 `24777fc8…`, dev42부터(U32 `release.acl_moves`·`cascade.scope_top1`); dev27~dev41 `b85003cd…`(U24 `release` 블록), dev18과 dev20~dev26 `5b7d666b…`, dev19 `0a4b7f06…`(U23 answered 0.6, dev20에서 되돌림), dev17 `49b9367f…`, dev16 `caef8ed5…`, dev11~dev15 `4681b686…`, dev5까지 `d8c74b40…`) | ADOPTED | 패키지 리소스, 시험이 hash를 고정 |
| 공개 모드 `current` | IMPLEMENTED | dev26까지의 엔진 기본값. dev27부터는 arm `release.mode = current`로 켬(실험의 되돌리기). 제품의 되돌리기는 dev26 wheel |
| 공개 모드 `as_of_snapshot`, `latest_at_release` | EXPERIMENTAL_ARM | 평가 실행기만 `Arm`으로 켬. latest 전환은 N3L-2에서 REJECT |
| 공개 모드 `scoped_dependency`, `release.retry`(0/1, `current`·`scoped_dependency`) | ADOPTED(dev27, 정책 `release` 블록: `scoped_dependency`·`retry: 1`) | U24(부모 `docs/UNIVERSAL_U24_REPORT.md`): 회귀 풀 tape 1,152요청 답 수준 동일, live 변경 시나리오 안전 위반 0·churn 무관 공개 96.7%·LA2 10/10. arm이 공개 키를 하나라도 두면 그 arm의 블록 전체를 쓰고(빠진 키는 dev26까지의 값), 없으면 정책 블록. 보장 범위는 `rvg.universal.scoped` docstring |
| `release.vectors`(`ignore`/`scoped`, `scoped_dependency`만) | ADOPTED(dev27 정책 값 `scoped`) | RVI-03 F2: 벡터 쓰기(outbox `vector` 행: 대기 행의 벡터 도착·벡터 교체)를 관련성으로 가름. `ignore`는 arm으로만(이전 scoped와 같음) |
| `release.retry_rule`(`any`/`slack`/`cited`)·`release.revalidate`(`scoped_dependency` + `retry: 1`만) | ADOPTED(dev27 정책 값 `slack`·`false`) | RVI-03 F3: 재생성 규칙(여유 조건, 인용 문서 한정)과 재검증 우선(첫 초안의 입력을 다시 읽어 바이트 단위로 같으면 재생성 없이 다시 공개 확인). `any`·`cited`·`revalidate: true`는 arm으로만 |
| `release.vector_moves`(`hold`/`after_check`, `vectors: scoped`만) | ADOPTED(dev27 정책 값 `after_check`) | FIX-V9-CHURN: 공개 확인 중 도착한 벡터 행을, 판정이 그 입력을 읽지 않았으면 공개 뒤 사건으로 넘김(`after_check`). `hold`(F2 V9)는 arm으로만 |
| `release.acl_moves`(`hold`/`after_check`, `scoped_dependency`만) | ADOPTED(정책 기본값 `after_check`, dev42부터, 부모 U32 ADOPT; dev40~dev41은 ADOPTED_OPT_IN) | 공개 확인이 ACL을 fence 때의 가시성(공개 영수증 `Release.hidden`)으로만 읽고, 확인 중 도착한 ACL 행은 답이 읽은 문서·family에 닿을 때만 보류(`acl_moved_during_check`), 그 밖은 공개 뒤 사건(`after_check`). **저장소 요구:** 상태 저장소가 `Release.hidden`을 줘야 한다(SQLite·PostgreSQL은 준다). 주지 않으면 지금처럼 보류한다. `hold`는 arm으로만(arm이 공개 키를 두면 블록 전체를 쓰므로 dev41 정책 블록을 통째로 둔다). 부모 `docs/UNIVERSAL_U32_REPORT.md`: ACL churn live 472요청에서 정책 기본값 D 대 되돌리기 R, 안전 위반 0, 관련 변경 첫 시도 공개 0/11, 무관 변경 아래 공개 72/72(R 42/74), `acl_moved` 보류 0(R 32), p95 ×1.06·구독 호출 ×1.01, churn 없는 재생 1,152건 동일. 처음 채택은 부모 `docs/rvi/FIX_ACL_WINDOW_REPORT.md` |
| N2 `unknown_revision` keep_both/tentative | EXPERIMENTAL_ARM(REJECT) | 기본 `prior` 유지 |
| N4 `conflict_partial=rewrite`, N5 `writer_schema=W1/W2` | EXPERIMENTAL_ARM(REJECT) | 기본 `off`/`W0` 유지 |
| 작성 근거 창 `cascade.evidence_window=neighbors-v1`(작성 단위마다 같은 판에서 게이트를 통과한 앞·뒤 단위) | ADOPTED(dev17) | 부모 N6-2 확인 분할에서 ADOPT. `none`은 arm으로만 |
| 답 형태 작성 계약 `cascade.writer_contract=form-v1` | EXPERIMENTAL_ARM(개발에서 탈락) | 기본 `W0` 유지(N6-2) |
| 둘째 단계 일치 규칙 `cascade.consistency`: 짝 초안을 둘째 단계 근거 역순으로(`widened`), 같은-답 판정(`same_text`, θ 0.7) | ADOPTED(dev18) | 부모 U22에서 ADOPT. U8 규칙(`first`·`identical_text`)·`normalized`·`off`는 arm으로만 |
| 통제기 프로필 `answered` 문턱 0.84 | ADOPTED(dev20에서 되돌림) | dev19의 0.6(부모 U23 ADOPT, ③ 확인 조건부)은 부모 U20 §8의 ③ 확인에서 미달이었다. 전제를 비튼 답 없는 질문에서 해로운 공개가 41 대 32/500이었고, 문턱이 전제 검사 역할을 했다. 다른 문턱·끔(0)·첫 단계 `write_top`은 arm(`cascade.answered_threshold`·`cascade.write_top`)으로만 |
| U27-a RVI-06 상류 상충 탐지 arm 이식 `cascade.upstream_conflict` | 미반영(생성기 전용) | dev30은 RVI-06 L4(dev14, REJECT)의 arm을 dev27 위로 옮겼다. 부모 U27-a(`docs/UNIVERSAL_U27A_REPORT.md`, 채점기 v3.1 ADOPT)가 시험용 보고 모드 출력을 만드는 데만 썼다. 코드는 브랜치 `rvg/u27-scorer-techqa`(c85da32, dev30)에만 있고, 이 판에는 없다 |
| U30 범위 결정 1위 제외 두 초안 `cascade.scope_top1: dual-draft-v1`(S3) | ADOPTED(정책 기본값, dev42부터, 부모 U32 ADOPT; dev38~dev41은 ADOPTED_OPT_IN) | **발동 조건:** 범위 결정이 RESOLVED인데 검색 1위 단위의 제목이 anchor가 아닐 때, 곧 범위 결정이 검색 1위 문서를 작성 풀에서 뺀 요청에서만 작동한다. 그때 1위 제목의 단위로 첫 초안을 하나 더 쓰고 판정한다. 그것만 공개되면 공개하고, 둘 다 공개되는데 짧은 답이 다르면 보류하며, 그 밖은 정책 경로다. 발동한 요청에서만 작성기 호출 1회와 그 판정이 늘어난다. `off`는 arm의 cascade 키 `scope_top1: off`로. 부모 `docs/UNIVERSAL_U32_REPORT.md`: 정책 모드 두 벌에서 정적 gold 정답 280→284(작성기 재표본 280), 오류율 +0.21%p, 해로운 공개 변화 0, 회귀 풀 1,152건 재생에서 발동 13건(U30 표적)만 달라짐. 처음 채택은 부모 `docs/UNIVERSAL_U30_REPORT.md`(발동 1,652요청 중 24, 1.5%) |
| U30 범위 결정 1위 제외 동률 규칙 `cascade.scope_top1: entity-tie-v1`(S2) | EXPERIMENTAL_ARM(부모 U30 ① 미달) | 1위 문서의 검색 단위 본문에 anchor 제목 전체가 있으면 1위 제목의 단위로 작성한다. 정답 +2·+3으로 채택 기준(+3 두 벌)에 못 미쳤다. arm으로만 |
| U26 `answered` 완화 구간 게이트(전제 검사·대상 일치) | INCONCLUSIVE(미반영) | 코드는 브랜치 `rvg/u26-premise-target`(22a038e, dev29)에만, 미반영. 정답 공개는 늘었으나 해로운 공개 +7~9/500으로 미달, 게이트 통과 92~98%. 부모 `docs/UNIVERSAL_U26_REPORT.md` |
| U25 작성기 지시 base-v2·근거 보기 lean-v1 | REJECT(미반영) | 코드는 브랜치 `rvg/u25-writer-refusal`(3978998, dev28)에만, 미반영. 부모 `docs/UNIVERSAL_U25_REPORT.md` |
| U20 범위 anchor `cascade.scope_anchor` | REJECT(미반영) | 코드는 브랜치 `rvg/u20-entity-anchor`(1977977)에만, 미반영. 부모 `docs/UNIVERSAL_U20_REPORT.md` |
| 공개 직전 회수·권한 확인(`release_authorize`) | ADOPTED | 모든 모드 |
| 공개 직전 tenant ACL epoch 확인(시작 epoch과 다르면 current·as_of 보류, latest 재시도)과 latest의 대체 선언 확인 | TESTED_REAL_BACKEND(PostgreSQL) | dev11(부모 RVI ACL 결함 수정), dev15부터 main. `current`·`as_of_snapshot`·`latest_at_release`에만 적용하고, 기본 `scoped_dependency`는 자기 ACL 규칙(`release.acl_moves`)을 쓴다 |

## 저장소·색인

| 구성 요소 | 상태 | 근거 |
|---|---|---|
| 상주 카탈로그(기본, `catalog: resident`) | ADOPTED | 파싱한 판을 메모리에 두고 세대 사이에 재사용한다. 100k에서 행당 약 5.6 KB(부모 `docs/N1F5_REPORT.md` §4) |
| 열 기반 카탈로그(`catalog: columnar`) | TESTED_OFFLINE(dev55, 부모 N1F-6 준비) | 고정폭 열(정렬 이름표 코드·revision·observed·inserted·32바이트 입력 hash)만 두고, 문서는 질의가 쓰는 행만 상태 저장소에서 읽는다(256건씩, LRU 4,096 판). 상주 카탈로그와 행·키·hash·문서·필터가 같다(시험, RVG 전체 시험 열 기반 강제 통과, N0 재생). 세대마다 다시 만든다(파싱 재사용 없음). SQLite는 `iter_bodies`로 흘려 읽어 만든다. 근거: `tests/test_columnar_catalog.py`, 부모 `docs/N1F6_REPORT.md` |
| SQLite 상태 저장소 | TESTED_REAL_BACKEND | RVG 시험 |
| PostgreSQL 상태 저장소(`postgres` extra) | TESTED_REAL_BACKEND | RVG 시험(`DEEPJEV_TEST_POSTGRES_DSN`), 연구 저장소 PORT-1 |
| local exact 색인 layout `base_delta`(matrix 디렉터리가 있을 때의 기본, `auto`) | ADOPTED(dev41, FIX-MATRIX-GEN) | 벡터 쓰기는 작은 delta에 더하고, delta가 θ(0.10)·base를 넘을 때만 base를 다시 쓴다. 파일은 `v3/base_delta/`. 부모 `docs/rvi/FIX_MATRIX_GEN_REPORT.md`: 1건/초 × 읽기 4에서 matrix 디렉터리 최대 점유가 전체 재작성의 3.0배에서 1.0배대로, 검색 결과·결정은 비트 동일(N0 재생 엄격 120/120, 결정적 재생). `open_storage` 설정의 해시가 파일 저장소에서 바뀐다(`layout: rebuild`를 적으면 예전 해시). 결과의 저장소 설명은 그대로 |
| local exact 색인 layout `rebuild`(행렬 스냅샷·memmap, `v2/`) | ADOPTED | 상주 색인(matrix 디렉터리 없음)과 TurboQuant 후보 단계의 기본. 파일 저장소에서는 `vectors.layout: rebuild`로 켬(되돌리기). RVG 시험, N0 재생 |
| local exact 증분 layout `segments`/`paged` | EXPERIMENTAL | N1F-2 arm, 기본 아님. 파일은 `v3/<layout>/` |
| local exact 앞 TurboQuant 4비트 후보 단계(`vectors.candidate_stage: turboquant_mse4`) | ADOPTED_OPT_IN(dev15, 부모 N1F-4 ADOPT) | 근사 후보 + 정확 재정렬. 부모 `docs/N1F4_REPORT.md`(2026-10-18): 상위 20 비트 동일 17,114/17,114, N0 재생 네 벌 엄격 120/120, 메모리 1/16, 12k행에서는 정확 경로보다 느림(p50 15.5 ms 대 1.05 ms, 이득은 fullwiki 규모). local exact에서만 켜지고 `auto` layout은 `rebuild`로 풀린다. 코드는 대량 색인 뒤 `rvg.build_candidate_codes(storage, tenant, space)`(0.1.0 공개 이름; `EmbeddingSpace` 또는 지문과 `dims`)로 만든다. 옛 내부 경로 `rvg.storage.quantized.build_codes`도 동작한다(내부 모듈). 정밀도를 `APPROXIMATE`로 선언하므로 기본 공개 확인(scoped)과 함께 쓰면 답이 읽지 않은 새 revision도 UNKNOWN(보류)이 된다 — 코드상 동작이며 측정 없음 |
| 외부 float16 벡터 묶음(`vectors.source: external_float16`, `vectors.external`) | TESTED_OFFLINE(dev54, 부모 N1F-5 준비) | 기존 float16 shard를 복사하지 않고 읽기 전용으로 순위를 매긴다(fullwiki 523만 문단용). 카탈로그 행은 RVG 입력 hash로만 외부 행에 연결되고, 입력이 다른 행(빈 제목·잘린 본문)은 벡터 없음이다. 점수는 shard 경계 고정 블록 분할에서 float16을 정확히 올린 행·float64 정규화 질의의 내적을 행 float64 노름으로 나눈 값(`score_dtype` float64 또는 float32)이고, 정밀도는 `EXACT_NATIVE`다(후보 단계를 켜면 `APPROXIMATE`). 묶음에 없는 입력의 상태 저장소 벡터는 같은 순위에 들어간다. 선택 속성 `query_projection: truncate_l2`(E9)는 같은 모델·같은 공간에서 차원만 큰 질의를 앞 d차원으로 잘라 정규화한다. 이때 blob 벡터를 공개 확인에 주지 않아 사건이 있으면 UNKNOWN(보류)이다. 기본값과 기본 설정 해시는 그대로다. 묶음은 `rvg.write_external_set`(`(title, text, vector)` 또는 `(input_hash, vector)` 행)이나 `rvg.export_external_set`(상태 저장소의 한 공간)으로 만든다(0.1.0 공개 이름, 내부 경로 `rvg.storage.external`도 동작). 형식은 `docs/external-vectors.md`(`rvg-external-vectors/1`), 예제는 `examples/storage/external_float16.py`다. 근거: RVG 시험 `tests/test_external_vectors.py`·`tests/test_external_writer.py`·`tests/packaging/test_external_example.py`, 부모 `docs/N1F5_REPORT.md` |
| pgvector 색인(`postgres` extra) | TESTED_REAL_BACKEND | RVG 시험(`DEEPJEV_TEST_PGVECTOR_DSN`), 연구 저장소 PORT-2 |
| Qdrant 색인(REST, 추가 의존 없음) | TESTED_REAL_BACKEND | RVG 시험(`DEEPJEV_TEST_QDRANT_URL`), 연구 저장소 PORT-2 |
| 배포 간 온라인 이전·rollback(`rvg.storage.migration`) | TESTED_REAL_BACKEND | RVG 시험, 연구 저장소 PORT-3 |
| 근사 검색(ANN) | NOT_IMPLEMENTED | — |

DSN을 주지 않으면 PostgreSQL·pgvector·Qdrant 시험은 건너뛴다(skip). 건너뛴 실행을 "지원 확인"으로 읽지 않는다. CI에는 실제 서비스(PostgreSQL 16 + pgvector, Qdrant)로 이 시험을 돌리는 `backends` 작업이 있다.

## provider

| 구성 요소 | 상태 | 근거 |
|---|---|---|
| OpenRouter Jev 판정 | TESTED_LIVE | 이관 전 연구 저장소 live 실행(같은 source bytes). 이관 뒤 live 실행 없음 |
| Vercel AI Gateway Jev, TypeSafe 직접 Jev | TESTED_OFFLINE | wire 형식 시험 |
| OpenRouter 작성기·추출기(Qwen), 임베딩(Gemini) | TESTED_LIVE | 이관 전 연구 저장소 live 실행 |
| Claude Code CLI 작성기·추출기(구독 Haiku, `claude_code` 공급자, `claude -p`) | ADOPTED(dev16, 부모 U21) | 격리 플래그(도구·설정 원천·MCP·세션 파일 없음, 호출마다 빈 작업 폴더). **PERF-1(dev44, 부모 `docs/rvi/PERF_1_REPORT.md` ADOPT, 2026-10-21):** 같은 150요청 live(정적 gold 100 + 실제 개정 fresh 50, 요청 동시 16·CLI 슬롯 8)에서 요청 지연 p50 17.06→11.75초(−31.1%), p95 40.74→26.94초(−33.9%), dev43 live tape 재생 150/150 바이트 동일, Jev 호출·429 증가 없음. 결과 JSON이 나오면 바로 반환하고 프로세스 종료(약 1초)는 배경에서 거둔다(시간 초과·취소 의미와 고아·좀비 없음은 그대로). 같은 모델·지시문·스키마가 다시 오는 호출은 미리 띄운 프로세스(같은 플래그, stdin stream-json으로 입력 하나만 받고 끝남, 세션 공유 없음)에 준다. **운영 요구:** `JsonTransport(claude_code_prewarm=2)`가 기본이며 유휴 프로세스는 이벤트 루프마다 최대 2개, 각 평균 약 270 MB(p95 353 MB, 부모 PERF-1 A′, Claude Code 2.1.306). 300초보다 오래된 것은 다음 호출 때 거두고, 호출이 없으면 루프·프로그램이 끝날 때까지 남는다. `claude_code_prewarm=0`이면 dev43까지처럼 호출마다 프로세스를 띄운다. 작성기는 스키마에 요청마다 다른 근거 id가 들어가 미리 띄우지 않는다(추출기처럼 반복되는 호출만). 모델·지시문·입력·tape 키는 바뀌지 않는다 |
| mock 판정·임베딩·생성 | TESTED_OFFLINE | mock 생성은 고정 응답이 없으면 범용 경로의 구조화 출력(작성기 초안·슬롯·재작성·검색어)을 가짜 공급자(`rvg.testing.fake_openrouter`)와 같은 규칙으로 만든다(작성기는 숫자가 있는 첫 근거 단위를 인용). 그래서 mock 모드도 네트워크 없이 끝까지 답한다(README 예제: 권한자 FULL "500", 비권한자 ABSTAIN). 판정 품질의 근거가 아니다. 시험 `tests/test_mock_mode.py`. 엔진 smoke는 `rvg.testing.fake_openrouter` |
| local semantic judge | NOT_IMPLEMENTED | 누락 시 외부 API로 대체하지 않는다 |
| 정책 선언 `models.embedding_dimensions`(3,072) | NOT_IMPLEMENTED | 코드가 읽지 않는 선언 필드다(D-DIM-DECL, 부모 `docs/N1F6_REPORT.md`). 임베딩 공간은 설정한 공급자의 것이고, 모든 측정은 1,536차원이다. dev56부터 `RuntimeSession.open`·CLI가 둘이 다르면 경고 로그만 낸다(0.1.0부터 프로세스당 한 번 INFO, 설정 기본 차원은 1,536). 결과·설정 해시·정책 해시·호출은 그대로다. 선언을 실제 동작에 묶는 수정은 정책 해시가 바뀌므로 별도 계약 후보다 |

## 녹화/재생·도구

| 구성 요소 | 상태 | 근거 |
|---|---|---|
| `RecordingTransport`/`ReplayTransport`/`Tape`(HTTP 시도 단위) | TESTED_OFFLINE | RVG 시험, N0 120요청 테이프 재생 |
| `rvg.testing.records`(요청 녹화 harness, 결정 서명 비교) | TESTED_OFFLINE | RVG 시험 |
| `rvg.testing.fake_openrouter` | TESTED_OFFLINE | 시험 대역 |

## 공개 session API(SEP-04, `rvg-runtime-api/0.1`)

`rvg.open_session`이 기존 객체를 조립하고 중립 동작을 연다. 자세한 의미는 [공개 API](api/PUBLIC_API.ko.md)에 있다. `capabilities()`의 `session_operations`·`events`·`access`·`observability` 항목이 같은 내용을 보고한다.

| 구성 요소 | 상태 | 근거 |
|---|---|---|
| `ingest`·`apply_event` | TESTED_REAL_BACKEND | RVG 시험(SQLite, PostgreSQL 상태) |
| 사건 `revision`(새 불변 revision)·`revoke`(문서의 모든 revision 회수) | IMPLEMENTED | RVG 시험 |
| 항목 단위 원자성(항목의 ACL, revision 또는 회수, outbox, 사건 원장 행을 한 트랜잭션) | TESTED_REAL_BACKEND | RVI-07A(dev3), 결함 주입 시험(SQLite, PostgreSQL) |
| 원자적 ACL 교체(`replace_access`, 사건 `access`) | TESTED_REAL_BACKEND | RVG 시험(SQLite, PostgreSQL): 전부 또는 전무, 같은 ACL은 무변경, outbox는 `set_access` 형식 |
| `capabilities`·`describe_runtime`·`close` | IMPLEMENTED | 정적 표·런타임 설명·session 닫기, 연결을 열지 않는다 |
| `answer`(엔진 결과 그대로의 봉투) | IMPLEMENTED | 엔진 직접 호출과 결과 dict 동일(가짜 provider). 공개하지 않은 결과에는 초안 값이 없다(0.1.0) |
| `health` liveness / readiness | IMPLEMENTED / TESTED_REAL_BACKEND | 무변경·비밀 없음 시험(PostgreSQL·pgvector·Qdrant) |
| `export_artifact_lineage`(session 기록, 현재 권한 적용) | IMPLEMENTED | 공개 문장이 인용한 단위만 기록한다(0.1.0). 시험: 회수·권한 변경 뒤 숨김, 다른 principal 참조 거부 |
| 사건 ID 멱등성 | TESTED_REAL_BACKEND | RVI-07A(dev3): `(tenant, issuer, event_id)` 원장과 내용 digest. 같은 key·같은 내용이면 처음 결과, 다른 내용이면 `event_id_conflict`. RVG 시험(SQLite, PostgreSQL). `event_id`가 없는 사건은 내용 기준 중복만 본다 |
| 아무도 허용하지 않는 ACL(빈 allow 목록) | TESTED_REAL_BACKEND | RVI-07A(dev3): `allow=()`는 명시적 deny-all이다. 어떤 principal도 가질 수 없는 예약 이름 하나만 허용하는 항목으로 저장한다. RVG 시험 |
| ACL allow/deny·그룹(`TrustedPrincipal`) | ADOPTED | N3 ACL, deny가 allow보다 우선 |
| 사용자 문서 선택(`selected_document_ids`) | ADOPTED | 검색 후보만 줄인다(`allowed_ids`) |
| 거부 목록(`TemporalRetriever(excluded_ids=...)`) | TESTED_OFFLINE(dev55, 부모 N1F-6) | 그 id의 모든 판을 검색과 공개 확인의 검색 가능 집합에서 뺀다. 질문 중 KB가 바뀌지 않으면 "나머지 전부" 허용 목록과 같은 결정이다(새 문서는 허용 목록이 막고 거부 목록은 들인다). 평가의 gold 제거용. 근거: `tests/test_columnar_catalog.py`, 부모 `docs/N1F6_REPORT.md` |
| 유효성 평면(높은 revision과 `supersedes` 선언을 tenant의 권위 저장소 전체에서 적용) | TESTED_REAL_BACKEND | RVI-02·02b(dev4·dev5): 숨김·회수·미선택·미검색 문서도 포함하고, 그 본문은 보이지 않는다 |
| 숨은 문서 본문에만 적힌 대체(`supersedes` 필드 없음) | NOT_IMPLEMENTED | 찾으려면 숨은 본문을 모델에 보여야 하므로 하지 않는다 |
| 대화 turn·구조화 출력 칸·snapshot 공개 모드 요청 | NOT_IMPLEMENTED | 공개 API가 받지 않는다 |
| 운영자 감사 lineage, 파생 artifact lineage | NOT_IMPLEMENTED | 감사 경로 없음. graph·요약을 만들지 않는다 |
| 고정 초안 verifier-only 인터페이스 | NOT_IMPLEMENTED | 공개 검증기 단독 동작 없음 |
| 결정 캐시(`decision_cache=CacheMode(...)`, `rvg.universal.cache`) | ADOPTED_OPT_IN(코드 dev57·문서 dev58, 사용자 결정 2026-10-27; 부모 계약 PROD-2·PROD-3은 REJECT) | opt-in, 기본은 꺼짐. **채택 근거와 계약 미달**: 부모 PROD-2·PROD-3의 사전 규칙은 미달이다(④ techqa·fullwiki·tydi-te, ④′(b) techqa 78.71%<80.19%, ⑤ 캐시 J1 7.47%>7.43%). 그래도 사용자가 opt-in 채택을 결정했다. 확인된 성질: 적중 결정의 원 결정 일치 99.5%, 다른 질문에 공개 0(θ 0.95, 0/13,000·0/6,500), 적중 57%(θ 0.95)·72%(θ 0.9), 작성기 호출 적중당 2.3회 절약, 적중 경로 약 1초. 보장·운영점은 `GUARANTEES.md` §6. 공개·보류로 끝난 결정을 항목으로 적재하고(`record`, 기본 True — 0.1.0), 첫 단계 근거 창이 상위 1 문서를 공유하거나 겹치고(자카드 ≥ τ, 기본 0.5; `exact`·`top1`·`overlap` 선택) Jev가 같은 질문이라 본(p ≥ θ, 기본 0.9) 새 질문에 그 결정을 다시 쓴다(`lookup`). 작성기·추출기를 부르지 않는다. 공개 항목은 새 질문으로 지금 창 위에서 통제기가 다시 판정하고(같은 창이면 주장 판정 재사용, 다르면 재판정, 회복 없음) 공개 확인을 거친다. 보류 항목은 같은 창일 때만 같은 보류를 돌려준다. 지금 창 밖의 단위를 인용한 항목은 재사용하지 않는다(철회·권한·거부 목록·효력). 캐시에서 나간 공개는 공개 기록에 `source.cache`(항목·p·θ·match)를 남긴다. 동치 판정·재검사 호출은 tape에 들어가 재생된다. 근거: `tests/test_decision_cache.py` |
| 넘김 패킷(`handoff_packet=True`, `result.handoff`, `handoff-v1`) | ADOPTED_OPT_IN(dev56, 부모 PROD-1 ADOPT) | opt-in, 기본은 꺼짐. `UniversalEngine(handoff_packet=True)`·`open_session(…, handoff_packet=True)`·CLI `--handoff-packet`. 공개하지 않은 결과(`ABSTAIN`·`UNRESOLVED_SCOPE`·`ERROR`)에만 붙는다. 공개 결과와 공개 기록·lineage에는 들어가지 않는다(`kind: handoff_draft`, `released: false`, schema `handoff_packet`). 보류 초안과 인용, 사유 코드, 주장→문장 대응·라벨·확률, 신호 대 문턱, 근거 단위 본문과 효력 상태, 짝 초안 비교, 작성기 거절 조건과 원 출력을 담는다. 단위 본문은 요청자가 지금 볼 수 있을 때만 담는다(공개 확인·거부 목록·선택·gate ELIGIBLE). 단계가 본 단위 하나라도 가려지면 그 단계가 쓴 글 전부를 가린다. 부모 PROD-1(`docs/PROD_1_REPORT.md`): 기록 3,080요청을 켜고 재생해 공개 결과·공개 기록이 바이트 단위로 같았고, 패킷 1,489개의 누출이 0이었다. 채움률은 항목 1~6 모두 100%였다. 자동 표시는 판독 정밀도가 14.5%라 자동 표시·신뢰도 낮음(코드 문자열 `automatic flag, low confidence`)으로 적는다. 근거: `tests/test_handoff_packet.py` |

없는 기능은 성공을 흉내 내지 않는다. 지금 연결이 되는지는 capabilities가 아니라 `health`가 말한다.

## 알려진 한계(삭제하지 않음)

- N1: 상주 색인·후보 단계·`layout: rebuild`에서는 스탬프(문서·회수 변경)마다 행렬 전체 재생성(100k JSON 벡터 약 53초), 첫 질의 cold 비용, 여러 프로세스가 같은 저장소를 쓸 때의 생성 경쟁은 lease·fencing으로 막지만 비용은 남는다.
- N3: `current`(dev26까지 기본)는 답을 만드는 동안 지식 번호가 바뀌면 공개하지 않는다. 동시 쓰기가 계속되는 부하에서는 공개가 거의 0이 된다(연구 저장소 N3L 발견). dev27 기본(scoped)은 답이 읽은 것과 무관한 변경이면 공개한다.
- U24(dev27): 기본 공개 확인은 상태 저장소의 outbox·ACL·회수·문서 읽기를 쓴다(SQLite·PostgreSQL 상태 저장소가 갖춤). 이를 갖추지 않은 사용자 정의 저장소에서는 해당 변경이 UNKNOWN이 되어 보류한다(안전 쪽, 가용성 손실). 평가 실행기에서는 arm `release.mode = current`로 비교할 수 있고, 제품의 되돌리기는 dev26 wheel이다. 관련 변경 뒤 재생성으로 요청당 생성 호출이 한 번 늘 수 있다(U24 live 관련 변경 100요청당 Haiku +97). churn 아래 `acl_moved_during_check` 보류는 dev42(부모 U32)부터 답이 읽은 문서·family에 닿는 ACL 행일 때만 남는다(무관 변경 보류 32 → 0). 상태 저장소가 `Release.hidden`을 주지 않으면 예전처럼 보류한다.
- N2: 개정 관계가 불확실할 때 선험(prior)으로 옛 판을 대체한다 — 합성 상충 사례에서 단정 1건이 남는다(기각된 대안보다 실제 개정 정답률이 높아 유지).
- 모든 provider 결과는 해당 모델 조합의 결과다. local judge 등으로 바꾸면 다시 평가해야 한다.
- 이관은 동작 보존 복사다. 정확성·보안·로컬 실행을 새로 인증한 것이 아니다.
