# Changelog

API, storage schema, policy, prompt and backend-capability changes are listed separately.

## 0.1.0 — first public release, 2026-10-29

The first public release of `rvg-runtime` (import `rvg`), the runtime of RVG — Retrieval-Verified
Generation. Apache-2.0, by Jooyoung Kim.

- **Release contract: attribute or hand off.** Answers are written from documents the asker may
  see and that are in force at the question's time. A fixed controller judges every sentence
  against its cited evidence; only verified text is released (`FULL`, `PARTIAL`), the rest is held
  (`ABSTAIN`) with its reason. A result that is not released carries no value of the held draft
  (no short answer, claim or sentence); the draft is only in the opt-in handoff packet.
- **One policy** for every knowledge base and language (`rvg/resources/policies/universal.json`,
  sha256 `24777fc8…`); every answer carries the policy hash.
- **Pre-release checks.** Revocation and the asker's permission are re-read just before release;
  the default scoped fence holds an answer when a change it read could matter.
- **Replay.** Recorded HTTP exchanges replay every decision, released answer and release record.
- **Cross-platform deterministic tie-break and ranking.** Dense search scans with BLAS, then
  rescores in a fixed summation order (numpy's pairwise sum, no BLAS) every row that can still
  reach the top-k under a proven error bound; the query is normalised in the same fixed order.
  The same stored vectors and query give the same ranking and the same score bits on every CPU
  and BLAS, with ties in catalog order (`(id, revision)` ascending).
- **Storage.** SQLite and PostgreSQL state stores; local exact vector index (with incremental
  layouts), pgvector and Qdrant adapters; optional float16 external vector sets and a columnar
  catalog for large corpora.
- **External vector set builder.** `rvg.write_external_set` writes a float16 vector set from
  `(title, text, vector)` or `(input_hash, vector)` rows, and `rvg.export_external_set` writes a
  state store's vectors; the format (`rvg-external-vectors/1`) is documented in
  `docs/external-vectors.md`, with an offline example in `examples/storage/external_float16.py`.
- **Providers.** Jev judge (OpenRouter Decisions by default; Vercel AI Gateway, TypeSafe direct),
  OpenRouter embeddings, and the Claude Code CLI or OpenRouter as writer and extractor.
- **Public API** `rvg-runtime-api/0.1` (`rvg.open_session`: ingest, events, answer, health,
  lineage — the units the released sentences cite) and the `rvg` command line (`rvg ingest` for
  documents, access lists and events from JSON Lines; `rvg universal-query --principal ID
  --group G`). The opt-in tools are public names too: `rvg.CacheMode`, `rvg.DecisionCache`,
  `rvg.build_candidate_codes`.
- **Opt-in features** (off by default): the handoff packet for held answers, and the decision cache
  that reuses a decision for a question the judge calls the same, re-checked on current evidence.
- **Documents**, in English with their Korean originals beside them (`*.ko.md`):
  `docs/GUARANTEES.md` (guarantees, measured properties and their limits),
  `docs/capabilities.md` (verification level of every component), `docs/api/PUBLIC_API.md`,
  `docs/architecture.md`; `docs/external-vectors.md` (the external vector set format).
  `scripts/check_doc_numbers.py` checks that each translation carries the same numbers.
- **Fixed before release, from a developer test of the public repository.**
  - The decision cache records by default (`CacheMode(DecisionCache(path))` alone recorded
    nothing, so nothing was reused); `record=False` gives a lookup-only cache.
  - Artifact lineage lists the cited units only, not the writer's whole evidence window.
  - `rvg.Settings(...)` accepts field names and the `rvg_mode` alias only; an unknown argument
    is a `ConfigurationError` (it was silently ignored).
  - The default embedding dimensions are 1,536, the configuration every measurement used.
  - The build backend is pinned (`hatchling==1.33.0`), so the wheel bytes do not depend on the
    day of the build.
  - CI runs the real-backend tests (PostgreSQL + pgvector, Qdrant) in service containers; two
    multi-process PostgreSQL tests no longer depend on the machine's speed.
  - `.env.example` is in English with the defaults of the code; `PROPOSER_PROVIDER=openrouter`
    with a non-OpenRouter model name fails when the session opens; the policy's
    embedding-dimensions note is logged once, at INFO.
  - Mock mode (the default) answers end to end with no network: the mock writer drafts by the
    rules of the HTTP-level fake, so the README example gives FULL "500" to a principal who may
    see the budget and ABSTAIN otherwise (it was `ERROR generation_schema_invalid`).
  - The workflows use actions that run on Node.js 24, and the release rehearsal (dry run,
    TestPyPI) is documented in `docs/RELEASING.md`.
  - Configuration errors and the handoff packet's flag note (`automatic flag, low confidence`)
    are in English; `docs/RELEASING.ko.md` is the Korean summary of `docs/RELEASING.md`.
  - Closing a deployment closes its local exact index: `Storage.close()` and
    `TemporalStore.close()` let go of every generation's pinned file and memory map at once,
    and a dense search that raises lets go of its generation (they stayed open until the
    garbage collector reached the store's reference cycle, and the suite ran out of
    descriptors under macOS's default `ulimit -n 256`); CI runs the tests within that limit.

## Pre-release development, summarized

`rvg-runtime` was developed and measured in a private research repository from September to
October 2026 (development versions 0.1.0.dev0 – 0.1.0.dev58, never published). The documents
of this repository cite that repository's reports ("private research repository report
`docs/...`", or an experiment ID); those reports are not public
and the measurements will be released with the paper. The main steps:

- **Extraction (dev0 – dev15).** The runtime was separated from the research code with its
  behaviour preserved (recorded-tape replay identical), given a public API, prompt and schema
  resources, storage adapters (SQLite, PostgreSQL, pgvector, Qdrant) and release checks
  (revocation, permission, validity of revisions).
- **Defaults chosen by measurement (dev16 – dev44).**
  - Writer and extractor on the Claude Code CLI with Claude Haiku (dev16).
  - A neighbour evidence window for the writer (dev17).
  - A second-stage agreement rule (dev18).
  - The scoped release fence as the default release check (dev27).
  - ACL changes arriving during a check treated as later events, and a dual draft when scope
    resolution drops the top-ranked document (dev42 – dev43).
  - Faster CLI generation (dev44).
  - Candidates that did not meet their pre-registered rules stayed off.
- **Documents (dev45 – dev53).** The guarantees-and-trade-offs document, the capability table
  matched to the code, and correction notes from later baselines (same-coverage comparison with
  selective plain RAG; date-aware RAG on real revisions).
- **Scale and product options (dev54 – dev58).**
  - Read-only float16 vector sets and a columnar catalog for a 5.23-million-passage corpus
    (dev54 – dev55).
  - The opt-in handoff packet (dev56).
  - The opt-in decision cache (dev57; adopted opt-in in dev58 by the author's decision, although
    its pre-registered confirmation rules were not all met — see `docs/GUARANTEES.md` §6).
