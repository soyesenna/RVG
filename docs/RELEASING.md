# Releasing rvg-runtime

The public repository (https://github.com/soyesenna/RVG) holds release snapshots; development
happens in a private repository. Each public release is one commit made from a reviewed tree.
A Korean summary is in [RELEASING.ko.md](RELEASING.ko.md).

## One-time setup: PyPI Trusted Publishing

`.github/workflows/release.yml` publishes with PyPI Trusted Publishing (OpenID Connect); no API
token is stored anywhere. Before the first tag, the PyPI account owner registers a **pending
publisher** for the project `rvg-runtime` (PyPI → Account → Publishing → Add a pending publisher):

| Field | Value |
|---|---|
| PyPI project name | `rvg-runtime` |
| Owner | `soyesenna` |
| Repository name | `RVG` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Also create the GitHub environment `pypi` (Settings → Environments) and, if the plan allows it,
require a reviewer for it, so that a tag cannot publish without approval. (Required reviewers need
a public repository or a paid plan; without them, pushing a `v*` tag publishes, so rehearse
first.)

For rehearsals on TestPyPI, register the same pending publisher on https://test.pypi.org (a
separate account and site) with the environment `testpypi`:

| Field | Value |
|---|---|
| PyPI project name | `rvg-runtime` |
| Owner | `soyesenna` |
| Repository name | `RVG` |
| Workflow name | `release.yml` |
| Environment name | `testpypi` |

## Rehearsal: run the release workflow by hand

The release workflow also runs by hand (Actions → Release → Run workflow), on the branch or tag
chosen there, with two inputs:

| `publish` | What runs |
|---|---|
| `none` (default) | Dry run: version check, tests, build, `check_dist.py`, `twine check --strict`; the files are uploaded as the run's artifact `dist` and their sha256 are listed in the run summary. Nothing is published. |
| `testpypi` | The same, then Trusted Publishing to TestPyPI (environment `testpypi`). |
| `pypi` | The same, then Trusted Publishing to PyPI (environment `pypi`). |

`expected_version` must equal the version in `pyproject.toml`; it is required unless
`publish=none`. On a tag, the tag must also be `v<version>`.

```bash
gh workflow run release.yml -R soyesenna/RVG -f publish=none
gh workflow run release.yml -R soyesenna/RVG -f publish=testpypi -f expected_version=0.1.0
gh run download <run id> -R soyesenna/RVG -n dist      # the built files, to compare or install
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ rvg-runtime==0.1.0
```

TestPyPI accepts each version once: a second `testpypi` run of the same version fails at upload.
The dependencies come from PyPI (`--extra-index-url`), since TestPyPI does not mirror them.

## Each release

1. Set the version in `pyproject.toml`, `src/rvg/__init__.py` and `tests/packaging/test_import.py`;
   regenerate `uv.lock` (`uv lock`); add the release section to `CHANGELOG.md`. The build backend
   stays pinned (`build-system.requires = ["hatchling==X.Y.Z"]`): the wheel records its
   generator's version, so raise the pin on purpose and note it in the changelog.
2. Check locally: `uv sync --frozen --all-extras`, `uv run pytest -q`, `uv build`,
   `uv run python scripts/check_dist.py dist/*.whl dist/*.tar.gz`, then install the wheel into a
   fresh virtual environment and run `python -c "import rvg; print(rvg.__version__)"` and
   `rvg --help`.
3. Rehearse: run the workflow by hand with `publish=none` (and, once per version, `testpypi`),
   then compare the sha256 in the run summary with the local build.
4. Commit, tag `vX.Y.Z` (annotated) and push the commit and the tag. The tag runs the release
   workflow: it checks that the tag matches the version, tests, builds, checks the distributions
   and publishes them to PyPI. (Or run it by hand with `publish=pypi` and `expected_version`.)
