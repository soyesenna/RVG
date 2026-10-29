# 구조

이 파일은 한국어 원본이다. 정본은 영어판 [architecture.md](architecture.md)이고, 두 판의 수치는 같다(`scripts/check_doc_numbers.py`).

## 모듈

연구 저장소의 `deepjev` 배치를 그대로 옮겼다(이름·경계 재설계 없음).

| 모듈 | 역할 |
|---|---|
| `rvg.api` / `rvg` | 공개 이름(기존 객체 re-export), `policy()`, `policy_hash()`, `capabilities()`, session API. `import rvg`는 아무 모듈도 싣지 않고 이름을 처음 쓸 때 `rvg.api`를 싣는다 |
| `rvg.session` | 공개 runtime session(`open_session`, `RuntimeSession`, `RequestContext`, 사건·ACL DTO, `OperationError`): 기존 저장소·provider·검색기·엔진을 조립·위임해 ingest·사건·답·health·lineage를 연다([공개 API](api/PUBLIC_API.ko.md)) |
| `rvg.prompts` | prompt·판정 문항 리소스 loader(manifest의 크기·sha256 확인, strict UTF-8, fallback 없음) |
| `rvg.cli` / `python -m rvg` | `rvg ingest`, `rvg universal-query`, `rvg policy`, `rvg capabilities` |
| `rvg.universal` | 정책 리소스(`config`), `UniversalEngine`(`engine`), 요청 trace(`trace`), 녹화/재생 transport(`replay`), 실험 arm 주입 포트(`experiment`, 평가 실행기 전용·비안정) |
| `rvg.control` | cascade, 통제기, 검증(`Asker`), 범위, 신호, 작성기 계약, 추출식 복구, 문자열 정규화(`grade.flat`) |
| `rvg.temporal` | 문서·질문·초안 모델, 유효성 게이트, 시점 검색기, N1 행렬 스냅샷, N3 principal·ACL |
| `rvg.storage` | 포트(`ports`), 배포 주입(`factory.open_storage`), adapter(SQLite·PostgreSQL·pgvector·Qdrant·local exact·증분 layout), catalog·조각·projection·공개 fence, 배포 간 이전(`migration`) |
| `rvg.providers` | Jev 판정(OpenRouter·Vercel·TypeSafe), 생성(OpenRouter, 구독 Claude Code CLI `claude_code`), 임베딩(OpenRouter), mock, 요청·응답 검증 |
| `rvg.config`, `rvg.factory`, `rvg.transport`, `rvg.gateway`, `rvg.contracts`, `rvg.runtime`, `rvg.errors` | provider 설정·조립, HTTP 재시도·예산, 판정 캐시, 계약, 요청 예산, 공개 오류 |
| `rvg.retrieval`, `rvg.grounded`, `rvg.approval` | 제품 경로가 쓰는 조각만: BM25·tokens·canonical hash, PassageStore·quote 정렬, 3-way 판정 문항 |
| `rvg.testing` | 개발·평가용: mock provider, 녹화/재생, `records`(요청 녹화 harness·결정 서명), `fake_openrouter`. 제품 모듈은 import하지 않는다(시험으로 강제) |
| `rvg/resources` | `policies/universal.json`, `prompts/`(prompt·판정 문항 107개 + `manifest.json`), `schemas/{runtime_request,evidence_unit,trace_node}.schema.json`, `schemas/api/*.schema.json`(공개 API 출력) |

## 의존 방향

```text
연구 실행기·benchmark adapter ──→ rvg 공개 API
rvg 제품 모듈 ──→ rvg 포트·계약
rvg adapter ──→ 포트 + 선택 driver(psycopg는 postgres extra, Qdrant는 httpx REST)
금지: rvg → 연구 저장소(deepjev)·scripts·benchmark·gold·데이터 파일
```

`tests/packaging/test_isolation.py`가 AST import와 실제 답변 중 열린 파일(감사 hook)로 확인한다. `tests/test_storage_ports.py`는 엔진·검색기·저장소 공통 모듈이 DB driver를 직접 import하지 않음을 확인한다.

## 정책·prompt·버전

- 정책은 패키지 리소스 하나(`importlib.resources`)이며 CWD나 checkout 위치로 찾지 않는다. `config_hash`로 모든 답에 찍힌다. 정책 bytes는 시험이 고정한다.
- prompt·판정 문항은 패키지 리소스 `rvg/resources/prompts/`(한 문자열 = 한 파일, UTF-8 bytes 그대로, 끝 개행 없음)이며 `manifest.json`이 크기·sha256을 고정한다(SEP-04). 조립(연결·`format`·`replace`·f-string 보간)은 코드에 있다. 이관 전 소스와의 동등성은 `tests/test_source_equivalence.py`(docstring 제외 AST 동등)가 모든 분기에 대해 확인한다. prompt를 바꾸면 manifest hash가 바뀌고, 녹화 테이프도 새로 필요하다.
- 버전은 서로 다르다: 라이브러리(`rvg.__version__`), 공개 API `rvg-runtime-api/0.1`, 관측 `rvg-observation/0.1`, prompt manifest(hash), 정책 `universal-v1`(hash), PostgreSQL 저장 schema `1`(`rvg_meta`), 저장소 설정 `schema_version 1`, 조각 규칙 `units-greedy/1`·`whole-revision/1`, 구조화 출력 parser `structured-json-strict-1`, 임베딩 공간 `version 2`(설정). 빌드 백엔드는 고정돼 있어(`hatchling==1.33.0`) wheel·sdist가 바이트까지 같게 다시 만들어진다(0.1.0).
- 설정 필드·환경변수 이름(`DEEPJEV_MODE`, `DEEPJEV_*_DSN` 등)과 trace ContextVar 이름(`deepjev_request_id` 등)은 기존 배포·녹화와의 호환을 위해 그대로다. 새 이름은 화이트리스트 alias `RVG_MODE`(= `DEEPJEV_MODE`) 하나뿐이다(아래). 이동 모듈 docstring의 옛 경로 표기는 SEP-04에서 `rvg.*`·`rvg/resources/policies/universal.json`으로 고쳤다(docstring만, 코드는 그대로).

## 설정

provider는 `Settings`(pydantic-settings)가 환경변수와 `--root`의 `.env`에서 읽는다. 우선순위는 명시 인자 > 환경변수 > `.env` > 기본값이다. 생성자 인자는 필드 이름과 별칭만 받고, 그 밖의 이름은 `ConfigurationError(CONFIG_UNKNOWN_SETTING)`이다(0.1.0). 환경변수·`.env`에서는 `RVG_MODE`를 `DEEPJEV_MODE`와 같은 키로 읽는다. 한 source 안에 둘 다 있고 값이 다르면 `CONFIG_ALIAS_CONFLICT`(값은 드러내지 않음)이고, 같으면 하나로 본다. 기본 모드는 `mock`(외부 호출 0). `live`에서 키가 없으면 `ConfigurationError`이며 다른 provider로 넘어가지 않는다. 배포 저장소는 `storage.json`(`examples/storage/`)으로 고르고, DSN은 환경변수 이름으로만 가리킨다.
