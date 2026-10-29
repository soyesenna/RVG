# 0001 — Initial extraction of the runtime from the research repository

The Korean original is [0001-initial-extraction.ko.md](0001-initial-extraction.ko.md). This English text is canonical; both carry the same numbers (`scripts/check_doc_numbers.py`).

- Method: selective copy. The history of this repository starts anew. The commits, reports and source texts of the original repository were not brought over (the full history is not kept).
- Source: private research repository commit `24d8625d1e137fa5b9712fcfa080e5f41096037a` (`src/deepjev/`, `configs/`, `tests/`).
- The mapping from source blobs to target blobs, file by file, is in the private research repository's `docs/RVG_MIGRATION_PROVENANCE.md`; the grounds of the classification are in its `docs/RVG_EXTRACTION_INVENTORY.md`.

## What changed

| Kind | Content |
|---|---|
| Bytes unchanged | 54 product modules (including `testing/records.py`, whose path alone changed), the policy JSON, 3 runtime schemas |
| Content changed | `universal/config.py`: reads the policy as a package resource instead of a path relative to the checkout. Policy bytes and hash unchanged |
| Split by definition | `control/grade.py` (`flat`), `grounded/engine.py` (`align_quote`), `grounded/store.py` (`PassageStore`), `temporal/engine.py` (claim extraction, judgement questions), `approval/core.py` (judgement questions), `retrieval/{data,search,store}.py` (canonical hash, BM25 and tokens, split_spans): the source bytes of the moved definitions are the same; the rest (graders, the legacy engine, SciFact tools, the ML gate) stays in the research repository |
| Only import lines changed | 28 tests, `testing/fake_openrouter.py` |
| Newly written | `api.py`, `cli.py`, `__main__.py`, `__init__.py` (lazy exports), `testing/__init__.py`, package marker docstrings, packaging and isolation tests, `scripts/check_dist.py`, documents |

## Equivalence gate

- The baseline capsule before the extraction (policy, source, prompt constants, smoke requests, schemas, storage schemas, settings) and `rvg` after the extraction must be the same.
- The N0 120-request tape replay must equal the replay of the code before the extraction, down to decision signatures, traces, floats and release records.
- The results are in the private research repository report `docs/RVG_LIBRARY_EXTRACTION_REPORT.md`.
