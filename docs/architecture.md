# Structure

The Korean original is [architecture.ko.md](architecture.ko.md). This English text is canonical; both carry the same numbers (`scripts/check_doc_numbers.py`).

## Modules

The `deepjev` layout of the research repository was moved as it was (no redesign of names or boundaries).

| Module | Role |
|---|---|
| `rvg.api` / `rvg` | public names (re-exports of existing objects), `policy()`, `policy_hash()`, `capabilities()`, the session API. `import rvg` loads no module; it loads `rvg.api` when a name is first used |
| `rvg.session` | the public runtime session (`open_session`, `RuntimeSession`, `RequestContext`, event and ACL DTOs, `OperationError`): assembles the existing stores, providers, retriever and engine, delegates to them, and opens ingest, events, answers, health and lineage ([Public API](api/PUBLIC_API.md)) |
| `rvg.prompts` | loader of the prompt and judgement-question resources (checks the manifest's sizes and sha256, strict UTF-8, no fallback) |
| `rvg.cli` / `python -m rvg` | `rvg ingest`, `rvg universal-query`, `rvg policy`, `rvg capabilities` |
| `rvg.universal` | the policy resource (`config`), `UniversalEngine` (`engine`), the request trace (`trace`), the recording and replay transport (`replay`), the experiment arm injection port (`experiment`, evaluation runner only, unstable) |
| `rvg.control` | cascade, controller, verification (`Asker`), scope, signals, the writer contract, extractive recovery, string normalisation (`grade.flat`) |
| `rvg.temporal` | document, question and draft models, the validity gate, the point-in-time retriever, N1 matrix snapshots, N3 principals and ACLs |
| `rvg.storage` | ports (`ports`), deployment injection (`factory.open_storage`), adapters (SQLite, PostgreSQL, pgvector, Qdrant, local exact, incremental layouts), catalogs, segments, projection, the release fence, moves between deployments (`migration`) |
| `rvg.providers` | Jev judgement (OpenRouter, Vercel, TypeSafe), generation (OpenRouter, the subscription Claude Code CLI `claude_code`), embeddings (OpenRouter), mock, request and response validation |
| `rvg.config`, `rvg.factory`, `rvg.transport`, `rvg.gateway`, `rvg.contracts`, `rvg.runtime`, `rvg.errors` | provider settings and assembly, HTTP retries and budgets, the judgement cache, contracts, request budgets, public errors |
| `rvg.retrieval`, `rvg.grounded`, `rvg.approval` | only the pieces the product path uses: BM25, tokens, canonical hash, PassageStore, quote alignment, the 3-way judgement question |
| `rvg.testing` | for development and evaluation: mock providers, recording and replay, `records` (request recording harness, decision signatures), `fake_openrouter`. Product modules do not import it (enforced by a test) |
| `rvg/resources` | `policies/universal.json`, `prompts/` (107 prompts and judgement questions + `manifest.json`), `schemas/{runtime_request,evidence_unit,trace_node}.schema.json`, `schemas/api/*.schema.json` (public API output) |

## Dependency direction

```text
research runners, benchmark adapters ──→ rvg public API
rvg product modules ──→ rvg ports and contracts
rvg adapters ──→ ports + optional drivers (psycopg through the postgres extra, Qdrant over httpx REST)
forbidden: rvg → the research repository (deepjev), scripts, benchmark, gold, data files
```

`tests/packaging/test_isolation.py` checks this with AST imports and with the files actually opened during an answer (an audit hook). `tests/test_storage_ports.py` checks that the engine, the retriever and the common storage modules do not import DB drivers directly.

## Policy, prompts and versions

- The policy is one package resource (`importlib.resources`); it is never looked up from the CWD or the checkout location. It is stamped on every answer as `config_hash`. A test pins the policy bytes.
- Prompts and judgement questions are package resources in `rvg/resources/prompts/` (one string = one file, the UTF-8 bytes as they are, no trailing newline), and `manifest.json` pins their sizes and sha256 (SEP-04). Assembly (concatenation, `format`, `replace`, f-string interpolation) is in the code. Equivalence with the sources before the extraction is checked for every branch by `tests/test_source_equivalence.py` (AST equality without docstrings). Changing a prompt changes the manifest hash and needs newly recorded tapes.
- The versions are different from each other: the library (`rvg.__version__`), the public API `rvg-runtime-api/0.1`, observation `rvg-observation/0.1`, the prompt manifest (hash), the policy `universal-v1` (hash), the PostgreSQL storage schema `1` (`rvg_meta`), the storage configuration `schema_version 1`, the segmentation rules `units-greedy/1` and `whole-revision/1`, the structured output parser `structured-json-strict-1`, the embedding space `version 2` (settings). The build backend is pinned (`hatchling==1.33.0`), so wheels and sdists rebuild byte for byte (0.1.0).
- The setting field and environment variable names (`DEEPJEV_MODE`, `DEEPJEV_*_DSN` and others) and the trace ContextVar names (`deepjev_request_id` and others) stay as they were, for compatibility with existing deployments and recordings. The only new name is the whitelisted alias `RVG_MODE` (= `DEEPJEV_MODE`) (below). The old path notation in the docstrings of moved modules was changed to `rvg.*` and `rvg/resources/policies/universal.json` in SEP-04 (docstrings only; the code is unchanged).

## Settings

`Settings` (pydantic-settings) reads the provider settings from environment variables and from the `.env` of `--root`. The precedence is explicit arguments > environment variables > `.env` > defaults. Constructor arguments are only field names and aliases; any other name is `ConfigurationError(CONFIG_UNKNOWN_SETTING)` (0.1.0). In the environment and in `.env`, `RVG_MODE` is read as the same key as `DEEPJEV_MODE`. If both are in one source with different values, it is `CONFIG_ALIAS_CONFLICT` (the values are not shown); if they are equal, they count as one. The default mode is `mock` (0 external calls). In `live` a missing key is a `ConfigurationError`, with no fallback to another provider. The deployment's store is chosen by `storage.json` (`examples/storage/`), and DSNs are pointed to only by environment variable name.
