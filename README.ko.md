# RVG — Retrieval-Verified Generation(한국어)

영어 README는 [README.md](README.md)다. Universal RVG runtime 라이브러리. import 이름 `rvg`, 배포 이름 `rvg-runtime`(PyPI).

허가되고 해당 시점에 유효한 기관 KB 문서만을 근거로 답을 만들고, 고정된 통제기가 근거로 확인한 부분만 `FULL` / `PARTIAL`로 공개하며 나머지는 `ABSTAIN`(또는 `UNRESOLVED_SCOPE`·`ERROR`)으로 돌려준다. 모든 KB·언어에 **하나의 정책**(`rvg/resources/policies/universal.json`)을 쓴다. KB·언어별 profile, threshold, 검색 방식, 사람 보정은 없다.

이 저장소는 연구 저장소(Deep Jev)에서 **동작을 보존해 분리한** runtime이다. 연구·평가·자료·보고서는 연구 저장소에 남는다. 분리했다고 정확성·보안·로컬 실행이 새로 인증된 것은 아니다.

## 설치

```bash
pip install rvg-runtime                 # core: httpx, numpy, pydantic, pydantic-settings
pip install "rvg-runtime[postgres]"     # PostgreSQL 상태 저장소·pgvector 색인 adapter(psycopg, LGPL-3.0)
```

## 사용

```python
import httpx
import rvg

rvg.policy_hash()        # 모든 답의 config_hash
rvg.capabilities()       # 이 build의 구성 요소와 검증 수준

# 설정은 호출하는 쪽이 자기 설정에서 골라 넘긴다. Settings(...)는 인자·환경변수만 읽고 .env 파일은 읽지 않는다.
settings = rvg.Settings(deepjev_mode="live", openrouter_api_key=my_secrets["openrouter"])
storage = rvg.open_storage("storage.json")     # 배포 선택(SQLite/PostgreSQL, 색인)
async with httpx.AsyncClient(trust_env=False) as client:
    transport = rvg.JsonTransport(client, timeout=20, max_attempts=3)
    engine = rvg.UniversalEngine(
        retriever=rvg.TemporalRetriever(storage, rvg.embedding_provider(settings, transport)),
        decision=rvg.decision_provider(settings, transport),
        writer=rvg.generation_provider(settings, transport, role="surface"),
        extractor=rvg.generation_provider(settings, transport, role="semantic"),
    )
    question = rvg.TemporalQuestion(text="...", as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z")
    result = await engine.answer("tenant", question, settings.context())
```

같은 조립을 한 번에 여는 공개 session API도 있다([공개 API](docs/api/PUBLIC_API.md)). 엔진 결과는 바꾸지 않고, ingest·사건·health·lineage를 함께 연다.

```python
async with rvg.open_session("storage.json", settings) as session:
    ctx = rvg.RequestContext(tenant="tenant", principal=rvg.TrustedPrincipal("user-1"))
    await session.ingest([document], ctx)
    envelope = await session.answer({"text": "...", "as_of": "...", "known_at": "..."}, ctx)
    envelope["result"]            # UniversalEngine.answer의 결과 그대로
    await session.health("readiness", ctx)
    await session.export_artifact_lineage(envelope["request_id"], ctx)
```

권한 주체는 신뢰된 인증 문맥에서 `rvg.TemporalRetriever(..., principal=rvg.TrustedPrincipal(...))` 또는 `rvg.RequestContext(principal=...)`로만 넣는다. 명령행: `rvg ingest 문서.jsonl --storage storage.json`(문서·access·사건, 한 줄에 하나), `rvg universal-query "질문" --storage storage.json --principal alice --group staff`, `rvg policy`, `rvg capabilities`. 구성 예제는 [examples/storage](examples/storage). 이미 있는 벡터(float16)를 그 자리에서 쓰려면 `rvg.write_external_set`으로 외부 벡터 묶음을 쓰고 `vectors.source: external_float16`을 켭니다([docs/external-vectors.md](docs/external-vectors.md), 예제 [examples/storage/external_float16.py](examples/storage/external_float16.py)).

## 설정

- **라이브러리로 쓸 때**: 호출하는 쪽(연구 하네스, 서비스)이 `rvg.Settings(...)`를 만들어 `open_session(storage, settings)`·provider factory에 넘긴다. `Settings(...)`는 인자 > 환경변수 > 기본값만 쓰고 `.env` 파일은 읽지 않는다. 호출하는 쪽은 자기 설정(자격 증명·모드·예산)에서 필요한 값만 골라 인자로 넘기고, RVG 설정 이름을 자기 `.env`에 섞지 않는다.
- **단독 실행**(CLI `rvg`, 스크립트): `rvg.load_settings(root)`가 `root/.env`를 더 읽는다(우선순위 인자 > 환경변수 > `.env` > 기본값). CLI의 `--root` 기본은 현재 위치이며 설치된 패키지나 소스 위치의 `.env`는 읽지 않는다. 전체 목록과 기본값은 [.env.example](.env.example) — 그대로 복사하면 아무것도 바뀌지 않고, 모드(`RVG_MODE`)와 키만 채우면 된다.
- `open_session(storage)`처럼 `settings`를 빼면 `load_settings(root 또는 현재 위치)`를 부르는 **편의 동작**이다. 실행 위치의 `.env`가 결과를 바꿀 수 있으므로 라이브러리 호출자는 `settings`를 넘긴다.
- 모드 이름은 `RVG_MODE`(= 옛 이름 `DEEPJEV_MODE`). threshold·검색 구성은 설정이 아니라 정책 파일(`universal.json`) 하나다.

## 개발

```bash
uv sync --all-extras
uv run pytest -q          # DEEPJEV_TEST_POSTGRES_DSN, DEEPJEV_TEST_PGVECTOR_DSN, DEEPJEV_TEST_QDRANT_URL을 주면 실제 backend 시험 포함
uv build --out-dir dist && uv run python scripts/check_dist.py dist/*.whl dist/*.tar.gz
```

## 문서

문서의 정본은 영어판이다. 아래 한국어판은 원본(보안·릴리스는 요약)이며, 원본은 영어판과 같은 수치를 담는다(`scripts/check_doc_numbers.py`가 대조한다).

- [공개 runtime API](docs/api/PUBLIC_API.ko.md)(영어 정본 [PUBLIC_API.md](docs/api/PUBLIC_API.md))
- [보장과 교환(측정 수치·기본값·저장소 요구·되돌리기)](docs/GUARANTEES.ko.md)(영어 정본 [GUARANTEES.md](docs/GUARANTEES.md))
- [지원 범위·검증 수준·알려진 한계](docs/capabilities.ko.md)(영어 정본 [capabilities.md](docs/capabilities.md))
- [구조·의존 방향·버전](docs/architecture.ko.md)(영어 정본 [architecture.md](docs/architecture.md))
- 영어로만 있는 문서: [외부 벡터 묶음 형식](docs/external-vectors.md)
- [이관 기록](docs/migrations/0001-initial-extraction.ko.md)(영어 정본 [0001-initial-extraction.md](docs/migrations/0001-initial-extraction.md))
- [보안(한국어 요약)](SECURITY.ko.md)(영어 정본 [SECURITY.md](SECURITY.md)), [릴리스 절차(한국어 요약)](docs/RELEASING.ko.md)(영어 정본 [RELEASING.md](docs/RELEASING.md)), [변경 이력](CHANGELOG.md)

라이선스는 Apache-2.0이다([LICENSE](LICENSE), [NOTICE](NOTICE)). 측정 근거 보고서는 비공개 연구 저장소에 있고, 논문(TechRxiv)으로 공개할 예정이다.
