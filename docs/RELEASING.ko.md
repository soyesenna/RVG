# rvg-runtime 배포 — 한국어 요약

정본은 영어 [RELEASING.md](RELEASING.md)다. 이 문서는 그 요약이고, 둘이 다르면 정본을 따른다.

- PyPI 배포는 Trusted Publishing(OIDC)이다. 토큰을 두지 않는다. 첫 태그 전에 PyPI 계정 소유자가 `rvg-runtime`에 pending publisher(소유자 `soyesenna`, 저장소 `RVG`, workflow `release.yml`, environment `pypi`)를 등록하고, GitHub에 environment `pypi`를 만든다.
- environment 검토자는 공개 저장소나 유료 플랜에서만 둘 수 있다. 검토자가 없으면 태그 push가 곧 배포이니 먼저 리허설한다.
- **리허설.** Actions에서 Release를 손으로 실행한다(`gh workflow run release.yml -f publish=…`).
  - `none`(기본)은 드라이런이다. 버전 확인·시험·빌드·check_dist·twine까지 하고, 파일은 artifact `dist`로 올리며 sha256은 실행 요약에 적는다. 배포하지 않는다.
  - `testpypi`는 TestPyPI로 Trusted Publishing한다. test.pypi.org에 같은 값의 pending publisher(environment `testpypi`)가 있어야 한다. 버전마다 한 번만 올라간다.
  - `pypi`는 실배포다. `none`이 아니면 `expected_version`이 필요하고, pyproject 버전과 같아야 한다.
- 판마다 버전·lock·CHANGELOG를 맞추고 로컬 검사(동기화·시험·빌드·check_dist·새 환경 설치)와 리허설을 거친 뒤 `vX.Y.Z` 태그를 push한다. 빌드 백엔드는 정확한 판으로 고정해 둔다(wheel이 생성기 판을 기록한다). 올릴 때는 일부러 올리고 CHANGELOG에 적는다.
