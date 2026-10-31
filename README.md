# RVG — Retrieval-Verified Generation

[![CI](https://github.com/soyesenna/RVG/actions/workflows/ci.yml/badge.svg)](https://github.com/soyesenna/RVG/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/soyesenna/RVG/blob/v0.1.1/LICENSE)

`rvg-runtime` (import name `rvg`) answers questions from an institution's own knowledge base under a
**release contract: attribute or hand off.** A writer drafts an answer from documents the asker may
see and that are in force at the question's time. A fixed controller then checks every sentence
against the cited evidence with a judge model. Only what the evidence supports is released
(`FULL` or `PARTIAL`); everything else is held (`ABSTAIN`) and handed to a person with the reason.
One policy file serves every knowledge base and language — no per-KB thresholds, profiles,
retrieval settings or human calibration.

Korean README: [README.ko.md](https://github.com/soyesenna/RVG/blob/v0.1.1/README.ko.md).

> **About this repository.** RVG was developed and measured in a private research repository
> (September to October 2026, development versions 0.1.0.dev0 to 0.1.0.dev58). This public
> repository holds release snapshots, so its history starts with the 0.1.0 release commit; the
> development steps are summarized in [CHANGELOG.md](https://github.com/soyesenna/RVG/blob/v0.1.1/CHANGELOG.md). Paper:
> [doi:10.5281/zenodo.23796302](https://doi.org/10.5281/zenodo.23796302).

## What it guarantees, and what it does not

Structural guarantees (enforced by code, kept by the test suite):

- **Only checked text is released.** No path releases a sentence the controller did not verify
  against cited, permitted, valid evidence. A result that is not released (`ABSTAIN`,
  `UNRESOLVED_SCOPE`, `ERROR`) carries no value of the held draft — no short answer, claim or
  sentence; the draft is only in the opt-in handoff packet.
- **One policy.** Every answer is stamped with the policy hash (`config_hash`).
- **Checked again right before release.** Revocation and the asker's permission are re-read just
  before an answer is released; a scoped fence holds the answer when a change it read could matter.
- **Replayable decisions.** Recording the HTTP exchanges ("tapes") and replaying them reproduces
  every decision, released answer and release record.
- **The same ranking on every platform.** Retrieval scores that decide a ranking are summed in a
  fixed order (no BLAS), so the same stored vectors and query rank the same, to the last bit, on
  any CPU or BLAS.

Measured properties (a fixed model combination, measured in a private research repository —
see [docs/GUARANTEES.md](https://github.com/soyesenna/RVG/blob/v0.1.1/docs/GUARANTEES.md) for every number, its sample and caveats):

- On real Wikipedia revisions, RVG released no superseded value as current (0 of 150 releases).
  The same model, given the revision dates and the validity rule in its prompt, also reached 0;
  RVG applies that metadata deterministically rather than by instruction.
- Compared at the **same coverage**, RVG's error rate is *not* distinguishable from a plain RAG
  answer filtered after the fact; the gain is the contract (attribution, validity, pre-release
  checks, replay), not higher accuracy.
- The price: RVG holds more answerable questions (72.5 vs 85.25 correct releases per 100 static
  questions) and is about 2× slower at the median.

What it does **not** guarantee: correctness of the model outputs themselves, behaviour with other
writer/judge/embedding models than the ones measured, group-membership changes during a release,
or writes that bypass the state store. See [docs/capabilities.md](https://github.com/soyesenna/RVG/blob/v0.1.1/docs/capabilities.md) for the
verification level of every component.

## Install

```bash
pip install rvg-runtime              # core: httpx, numpy, pydantic, pydantic-settings
pip install "rvg-runtime[postgres]"  # PostgreSQL state store and pgvector index (psycopg, LGPL-3.0)
```

Python 3.11 or newer.

## A 60-second example (no network, no keys)

[`examples/api/public_consumer.py`](https://github.com/soyesenna/RVG/blob/v0.1.1/examples/api/public_consumer.py) runs every public operation
against a fresh SQLite store with a deterministic, offline stand-in for the model provider:

```bash
python examples/api/public_consumer.py
```

```python
import httpx, rvg
from rvg.testing.fake_openrouter import FakeOpenRouter, providers  # offline test double

storage = {"schema_version": 1, "state": {"backend": "sqlite", "path": "state.sqlite3"}}
fake = FakeOpenRouter()
async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
    transport = rvg.JsonTransport(client, timeout=5, max_attempts=3)
    async with rvg.open_session(storage, rvg.Settings(), providers=providers(transport)) as session:
        ctx = rvg.RequestContext(tenant="example", principal=rvg.TrustedPrincipal("alice", groups=("staff",)))
        await session.ingest([...], ctx)               # documents, revisions, access rules
        envelope = await session.answer(
            {"text": "What is the budget of project A?",
             "as_of": "2026-09-20T00:00:00Z", "known_at": "2026-09-20T00:00:00Z"}, ctx)
        envelope["result"]["status"]                    # FULL / PARTIAL / ABSTAIN
```

It prints one JSON object. For the asker allowed to read the document, the answer is `FULL`
(`"500"`, with its evidence units). For a principal without access it is `ABSTAIN`. A revoked
document never reaches an answer. The stand-in shows that the operations connect, not how well a
model answers.

## Providers and secrets

RVG talks to three model roles. Secrets come from environment variables (or arguments you pass);
nothing is read from files unless you use the standalone loader `rvg.load_settings(root)`, which
also reads `root/.env`. See [.env.example](https://github.com/soyesenna/RVG/blob/v0.1.1/.env.example) for every setting.

| Role | Default | Setting |
|---|---|---|
| Judge (controller) | Jev `typesafe/jev-1.14` via OpenRouter Decisions | `JEV_PROVIDER=openrouter`, `OPENROUTER_API_KEY`; or `vercel_gateway` (`AI_GATEWAY_API_KEY`), `typesafe_direct` (`TYPESAFE_API_KEY`) |
| Embeddings | `google/gemini-embedding-2` via OpenRouter, 1,536 dimensions | `EMBEDDING_PROVIDER=openrouter`, `OPENROUTER_API_KEY`, `EMBEDDING_DIMENSIONS` (default 1536, the measured configuration) |
| Writer and extractor | Claude Code CLI, `claude-haiku-5-5` | `PROPOSER_PROVIDER=claude_code` (an installed, logged-in `claude` CLI; another binary with `rvg.JsonTransport(..., claude_code_binary=...)`), or `PROPOSER_PROVIDER=openrouter` with `SEMANTIC_PROPOSER_MODEL` / `SURFACE_PROPOSER_MODEL` through the API |

Set `RVG_MODE=live` to call providers. The default `mock` makes no external call. In mock mode,
`rvg ingest` and `rvg universal-query` run offline end to end with deterministic stand-ins; they
show the wiring, not answer quality. As a library,
build `rvg.Settings(...)` yourself and pass it in; `Settings(...)` reads arguments and environment
variables only:

```python
settings = rvg.Settings(deepjev_mode="live", openrouter_api_key=os.environ["OPENROUTER_API_KEY"])
async with rvg.open_session("storage.json", settings) as session:
    ...
```

Command line: `rvg ingest documents.jsonl --storage storage.json` (documents, access lists and
events, one JSON object per line, then embeddings), `rvg universal-query "question" --storage
storage.json --principal alice --group staff`, `rvg policy`, `rvg capabilities`; `rvg --help`.
Storage configurations: [examples/storage](https://github.com/soyesenna/RVG/tree/v0.1.1/examples/storage) (SQLite, PostgreSQL + pgvector;
Qdrant through its REST API). To rank vectors you already have in place (float16), write an
external vector set with `rvg.write_external_set` (or `rvg.export_external_set` from a state
store) and set `vectors.source: external_float16`: [docs/external-vectors.md](https://github.com/soyesenna/RVG/blob/v0.1.1/docs/external-vectors.md),
example [examples/storage/external_float16.py](https://github.com/soyesenna/RVG/blob/v0.1.1/examples/storage/external_float16.py).

## Opt-in features

All are off by default and do not change the policy or settings hashes.

- **Handoff packet** — `open_session(..., handoff_packet=True)` (or CLI `--handoff-packet`) attaches
  `result.handoff` to every answer that is not released: the held draft and its citations, reason
  codes, claim labels, the evidence text the asker may see, the paired draft and the writer's
  refusal. Never attached to a released answer.
- **Decision cache** —
  `open_session(..., decision_cache=rvg.CacheMode(rvg.DecisionCache("cache.sqlite3")))`. By
  default the cache both records every finished decision (released
  or held) and looks it up for later questions; pass `record=False` for a lookup-only cache (or
  `lookup=False` to fill one). A decision already made is reused for a question whose evidence
  window shares the top document or overlaps, and that the judge calls the same question
  (probability ≥ `theta`). No writer call; the draft is judged again for the new question on the
  current evidence, so revoked, hidden or no longer valid units are never reused. The release
  record says so (`source.cache`). Operating points (θ 0.95 / 0.9) and the measured limits are in
  [docs/GUARANTEES.md](https://github.com/soyesenna/RVG/blob/v0.1.1/docs/GUARANTEES.md) §6.
- **Candidate stage** — storage `"vectors": {"candidate_stage": "turboquant_mse4"}` (local exact):
  4-bit TurboQuant codes propose candidates, the exact vectors rank them. Build the codes after
  loading, with `await rvg.build_candidate_codes(storage, tenant, space)` (`space`: the
  embedding space, e.g. `embedding.describe()`); its precision is `APPROXIMATE`.

## Replay and audit

- Every answer carries a trace and, when released, a release record (knowledge snapshot,
  revocation sequence, ACL epoch, source). `session.export_artifact_lineage(request_id, ctx)`
  returns the evidence a request released, with current permissions applied.
- `rvg.testing.records` records requests with their HTTP exchanges. `rvg.universal.replay` replays
  them without a network. These modules are internal: they are outside the public API's stability
  promise.

## Repository layout

- `src/rvg` — the package. Public names are `rvg.__all__` ([docs/api/PUBLIC_API.md](https://github.com/soyesenna/RVG/blob/v0.1.1/docs/api/PUBLIC_API.md)).
  The single policy is `src/rvg/resources/policies/universal.json`; prompts and JSON schemas are
  package resources.
- `tests` — the test suite (no network; PostgreSQL, pgvector and Qdrant tests run when
  `DEEPJEV_TEST_POSTGRES_DSN`, `DEEPJEV_TEST_PGVECTOR_DSN`, `DEEPJEV_TEST_QDRANT_URL` are set).
- `examples` — the public-API consumer, the external vector set example and storage
  configurations.
- `scripts` — **maintainer tooling**, not part of the package: release checks
  (`check_dist.py`, `release_tools.py`), the public API map (`make_api_map.py`) and the number
  check between the English documents and their Korean originals (`check_doc_numbers.py`).
- `docs` — guarantees, capabilities, architecture, the public API, the external vector set format,
  releasing, and the extraction record.

The documents are in English. The Korean originals of the guarantees, capabilities, architecture,
public API and extraction-record documents are kept beside them as `*.ko.md`, with Korean
summaries of the security and release policies (`SECURITY.ko.md`, `docs/RELEASING.ko.md`); the
English text is canonical, and `scripts/check_doc_numbers.py` checks that each translation carries
the same numbers. The documents cite reports of the private research repository in
which RVG was developed and measured ("private research repository report `docs/...`", or an
experiment ID). Those reports are not public; the main measurements are reported in the paper
(see [Citation](#citation)).
RVG was developed by Jooyoung Kim (independent researcher).

## Development

```bash
uv sync --all-extras
uv run pytest -q
uv build && uv run python scripts/check_dist.py dist/*.whl dist/*.tar.gz
```

Contributing: [CONTRIBUTING.md](https://github.com/soyesenna/RVG/blob/v0.1.1/CONTRIBUTING.md). Security: [SECURITY.md](https://github.com/soyesenna/RVG/blob/v0.1.1/SECURITY.md).
Changes: [CHANGELOG.md](https://github.com/soyesenna/RVG/blob/v0.1.1/CHANGELOG.md). Releasing: [docs/RELEASING.md](https://github.com/soyesenna/RVG/blob/v0.1.1/docs/RELEASING.md).

## Citation

If you use RVG, please cite the paper (preprint on Zenodo,
[doi:10.5281/zenodo.23796302](https://doi.org/10.5281/zenodo.23796302)):

```bibtex
@misc{kim_2026_23796302,
  author    = {Kim, Jooyoung},
  title     = {Retrieval-Verified Generation: A Release Contract for Retrieval-Augmented Answering},
  month     = oct,
  year      = 2026,
  publisher = {Zenodo},
  version   = 1,
  doi       = {10.5281/zenodo.23796302},
  url       = {https://doi.org/10.5281/zenodo.23796302}
}
```

The released software `rvg-runtime` 0.1.0 is archived on Zenodo as
[doi:10.5281/zenodo.23796304](https://doi.org/10.5281/zenodo.23796304).

## License

Apache License 2.0 — see [LICENSE](https://github.com/soyesenna/RVG/blob/v0.1.1/LICENSE) and [NOTICE](https://github.com/soyesenna/RVG/blob/v0.1.1/NOTICE).
