# 0001 — 연구 저장소에서 runtime 초기 이관

이 파일은 한국어 원본이다. 정본은 영어판 [0001-initial-extraction.md](0001-initial-extraction.md)이고, 두 판의 수치는 같다(`scripts/check_doc_numbers.py`).

- 방식: 선택 복사. 이 저장소의 이력은 새로 시작했다. 원본 저장소의 커밋·보고서·원문은 가져오지 않았다(전체 이력 보존 아님).
- 원본: 연구 저장소 commit `24d8625d1e137fa5b9712fcfa080e5f41096037a`(`src/deepjev/`, `configs/`, `tests/`).
- 파일별 원본 blob·대상 blob 대응은 연구 저장소 `docs/RVG_MIGRATION_PROVENANCE.md`, 분류 근거는 `docs/RVG_EXTRACTION_INVENTORY.md`에 있다.

## 바뀐 것

| 구분 | 내용 |
|---|---|
| bytes 그대로 | 제품 모듈 54개(경로만 바뀐 `testing/records.py` 포함), 정책 JSON, runtime schema 3개 |
| 내용 변경 | `universal/config.py`: 정책을 checkout 기준 경로 대신 패키지 리소스로 읽음. 정책 bytes·hash 불변 |
| 정의 단위 분할 | `control/grade.py`(`flat`), `grounded/engine.py`(`align_quote`), `grounded/store.py`(`PassageStore`), `temporal/engine.py`(주장 추출·판정 문항), `approval/core.py`(판정 문항), `retrieval/{data,search,store}.py`(canonical hash, BM25·tokens, split_spans): 이동한 정의의 source bytes 동일, 나머지(채점기·레거시 엔진·SciFact 도구·ML gate)는 연구 저장소 |
| import 줄만 변경 | 시험 28개, `testing/fake_openrouter.py` |
| 새로 작성 | `api.py`, `cli.py`, `__main__.py`, `__init__.py`(지연 export), `testing/__init__.py`, 패키지 표지 docstring, 패키징·격리 시험, `scripts/check_dist.py`, 문서 |

## 동등성 관문

- 이관 전 기준선 캡슐(정책·source·prompt 상수·smoke 요청·schema·저장소 schema·설정)과 이관 뒤 `rvg`가 같아야 한다.
- N0 120요청 테이프 재생이 이관 전 코드의 재생과 결정 서명·trace·float·release 기록까지 같아야 한다.
- 결과 수치는 연구 저장소 `docs/RVG_LIBRARY_EXTRACTION_REPORT.md`에 있다.
