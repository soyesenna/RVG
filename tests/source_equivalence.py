"""Source equivalence with the extraction baseline (SEP-04 G04-PROMPT).

A module of the installed package is compared with its `018c355` source (a fixture copy) after the
same normalisation on both sides: docstrings are dropped (SEP-04 corrects old paths in them),
the `from ..prompts import prompt` import is dropped, every `prompt("id")` call becomes the
resource text it returns, literal parts of an f-string are merged, and methods declared as added
are removed. Equal ASTs mean the module does exactly what it did, with its texts now read from
resources — for every branch, observed in a run or not."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures/baseline_018c355"

# Declared before implementation (docs/sep04-06/SEP04_PROTOCOL.md, G04-PROMPT 2).
INTENDED = {
    "rvg/__init__.py": "new public names",
    "rvg/api.py": "new public names, capability fields",
    "rvg/config.py": "Settings alias RVG_MODE <-> DEEPJEV_MODE; dev11 jev-1.14 decision defaults; "
    "U21 proposer_provider claude_code; dev16 Claude Haiku generation defaults",
    # RVI-07A (candidate rvi/candidate-20260925): one-transaction events, event-id ledger,
    # explicit deny-all (docs/rvi/RVI07A_PROTOCOL.md in the parent repository).
    "rvg/storage/sqlite.py": "RVI-07A commit_event, event ledger, in-transaction helpers",
    "rvg/storage/postgres.py": "RVI-07A commit_event, event ledger, in-transaction helpers",
    "rvg/temporal/access.py": "RVI-07A reserved deny-all principal name",
    "rvg/temporal/gate.py": "RVI-02 validity plane: replacements declared outside the retrieved set",
    "rvg/temporal/retriever.py": "RVI-02 validity_documents from the authoritative state; 0.1.0 "
    "the generation view is let go of when a dense search raises (parent PUB-4)",
    "rvg/universal/engine.py": "RVI-02 the gate receives the validity plane; release-input "
    "consistency (ACL epoch at the release check, latest's replacement declarations); N6-2 "
    "tied units in document order (rank_units), writer_contract/evidence_window arm accessors; "
    "U22 consistency_partner/consistency_judge arm accessors; U23 answered_threshold and "
    "write_top arm values passed to the controller and the cascade; RVI-03 (opt-in arm) "
    "scoped dependency fence, release retry and the reads each search records; U24 (dev27) "
    "release_block: the arm's release block whole, else the policy's; PROD-1 (opt-in) "
    "handoff_packet: the handoff trail and result.handoff on a result not released; PROD-2 "
    "(opt-in) decision_cache: the lookup before writing, source.cache, the recorded entry",
    # 0.1.0.dev11 (parent docs/rvi/RVI_CYCLE3_DECISIONS_4.md §1): decision model jev-1.14.
    "rvg/testing/fake_openrouter.py": "dev11 the fake decisions endpoint answers as jev-1.14",
    # N1F-4 (parent docs/N1F4_PROTOCOL.md §2): the opt-in TurboQuant candidate stage, off by
    # default (the configuration hash of an existing deployment is unchanged).
    "rvg/storage/factory.py": "N1F-4 vectors.candidate_stage and its opt-in wiring; FIX-MATRIX-GEN "
    "layout auto (base_delta with a matrix directory) and the resolved layout in the fingerprint; "
    "N1F-5 vectors.source/external (opt-in, out of the fingerprint when off) and external_storage; "
    "0.1.0 Storage.close closes the vector index (parent PUB-4)",
    "rvg/storage/local_exact.py": "FIX-MATRIX-GEN versioned matrix directory per index layout; "
    "0.1.0 canonical scores in a fixed summation order (BLAS scan, certified rescoring); "
    "0.1.0 close() lets go of the generations (parent PUB-4)",
    # N1F-6 (parent docs/N1F6_PROTOCOL.md): the opt-in columnar catalog, batched document access
    # (`docs`, the same objects in the resident layout) and the `excluded` deny list.
    "rvg/storage/catalog.py": "N1F-6 ColumnarCatalog, CatalogCache(layout=...), docs(rows), "
    "eligible(excluded=...), revision_starts, coverage, vector_mask",
    "rvg/storage/projection.py": "N1F-6 documents of the projected rows read in one batch",
    "rvg/storage/incremental.py": "FIX-MATRIX-GEN v3/<layout> directory, parts cached by directory; "
    "0.1.0 BLAS scan of the parts, certified rescoring; 0.1.0 close() lets go of the parts",
    # 0.1.0 (parent docs/PUB_2_REPORT.md): the module description says a catch-up chases a writer
    # faster than the target (docstring only).
    "rvg/storage/migration.py": "0.1.0 catch_up note: a writer faster than the target",
    # 0.1.0 (parent docs/PUB_2B_REPORT.md): mock mode answers the product path end to end.
    "rvg/providers/mock.py": "0.1.0 extractive_draft/structured_answer (the mock writer's rules)",
    # 0.1.0 (parent docs/PUB_2_REPORT.md): the command line's help in English and `ingest`.
    "rvg/cli.py": "0.1.0 English help, the ingest command, input errors named without content",
    # 0.1.0 (parent docs/PUB_2_REPORT.md): an evaluation's collector of withheld draft values.
    "rvg/universal/trace.py": "0.1.0 HELD_DRAFTS context variable (rvg.testing.records)",
    # 0.1.0 (parent docs/PUB_1_REPORT.md): the same ranking bits on every platform's BLAS.
    "rvg/storage/remote.py": "0.1.0 query normalised in a fixed summation order (`unit`)",
    "rvg/temporal/store.py": "N1F-4 TemporalStore(candidate_stage=...) opt-in argument; 0.1.0 "
    "close() closes the vector index it owns (parent PUB-4)",
    # U21 (parent docs/UNIVERSAL_U21_PROTOCOL.md §8.1): generation through the Claude Code CLI on
    # the user's subscription by proposer_provider="claude_code" (opt-in in a995819, the default
    # since 0.1.0.dev16 after U21 ADOPT, parent docs/UNIVERSAL_U21_REPORT.md).
    "rvg/factory.py": "U21 proposer_provider claude_code; 0.1.0 OpenRouter model id check",
    "rvg/transport.py": "U21 claude-code:// requests run the Claude Code CLI",
    "rvg/providers/generation.py": "U21 ClaudeCodeGenerationProvider; 0.1.0 the mock answers the "
    "product schemas without a fixture",
    # N6-2 (parent docs/N6_2_TECHQA_PROTOCOL.md §1): opt-in arms, off by default (policy unchanged)
    "rvg/control/cascade.py": "N6-2 windowed(): the neighbour window (the policy's since dev17); "
    "writer_contract passed on; U22 partner evidence and judge of the agreement rule (arm only); "
    "U23 write_top (arm only); PROD-1 (opt-in) the handoff trail of each written stage; PROD-2 "
    "(opt-in) the decision cache's lookup and recheck before the first stage",
    "rvg/control/controller.py": "U23 ControlPolicy.answered (arm only): the profile's answered "
    "threshold, 0 drops it from the gate; PROD-1 detail() and the handoff trail (opt-in); PROD-2 "
    "reused claim judgements and no recovery for the decision cache's recheck (opt-in)",
    "rvg/control/grade.py": "U22 normalized_equal (norm-v1) and same_text (U12-b) judges",
    "rvg/control/writer.py": "N6-2 the answer-form contract (FORM) behind contract='form-v1'; "
    "PROD-1 refusal() and the writer's handoff trail (opt-in)",
    # PROD-1 (parent docs/PROD_1_PROTOCOL.md): the opt-in handoff packet and D-DIM-DECL's warning
    "rvg/universal/cli.py": "PROD-1 --handoff-packet; D-DIM-DECL warning (log only); 0.1.0 "
    "ingest, --principal/--group, English help",
    "rvg/universal/config.py": "D-DIM-DECL dimension_warning/warn_dimensions (log only; 0.1.0 "
    "INFO, once per process)",
    "rvg/testing/records.py": "PROD-1 a record keeps result.handoff when the engine gave one",
    "rvg/universal/experiment.py": "N6-2 arm keys writer_contract and evidence_window; U22 "
    "consistency_partner and consistency_judge; U23 answered_threshold and write_top; RVI-03 "
    "release mode scoped_dependency and release.retry",
}
# Non-Python resources changed on purpose: path -> ((text now, text of the baseline), ...). The
# test replaces each text back and requires the baseline bytes, so nothing else may change.
INTENDED_RESOURCES = {
    "rvg/resources/policies/universal.json": (
        # dev42 (U32) ACL moves after the check and the scope's dual draft (reverted first)
        (
            '    "evidence_window": "neighbors-v1",\n    "scope_top1": "dual-draft-v1"\n',
            '    "evidence_window": "neighbors-v1"\n',
        ),
        (
            '    "vector_moves": "after_check",\n    "acl_moves": "after_check",\n',
            '    "vector_moves": "after_check",\n',
        ),
        # dev27 (U24) the release block: scoped dependency, vectors scoped, vector moves after
        # the check, one slack retry (reverted first)
        (
            (
                '  "release": {\n    "mode": "scoped_dependency",\n    "vectors": "scoped",\n'
                '    "vector_moves": "after_check",\n    "retry": 1,\n    "retry_rule": "slack",\n'
                '    "revalidate": false\n  },\n'
            ),
            "",
        ),
        # dev11 decision jev-1.14; dev16 (U21) writer and extractor Claude Haiku on the CLI path
        (
            (
                '"decision": "typesafe/jev-1.14",\n    "writer": "claude-haiku-5-5",\n'
                '    "extractor": "claude-haiku-5-5",\n    "generation_provider": "claude_code",'
            ),
            (
                '"decision": "typesafe/jev-1.13",\n    "writer": "qwen/qwen3.8-27b",\n'
                '    "extractor": "qwen/qwen3.8-27b",'
            ),
        ),
        # dev18 (U22) the consistency rule: widened partner, same-answer judge (reverted first)
        (
            '"consistency": {"partner": "widened", "judge": "same_text", "theta": 0.7}',
            '"consistency": {"partner": "rewrite", "judge": "identical_text"}',
        ),
        # dev17 (N6-2) the neighbour evidence window
        (
            '"identical_text"},\n    "evidence_window": "neighbors-v1"\n',
            '"identical_text"}\n',
        ),
    ),
}
ADDED_METHODS = {
    "rvg/storage/sqlite.py": {"SQLiteState": {"replace_access"}},
    "rvg/storage/postgres.py": {"PostgresState": {"replace_access"}},
}
NEW_MODULES = {
    "rvg/prompts.py",
    "rvg/session.py",
    "rvg/storage/quantized.py",
    "rvg/storage/external.py",  # N1F-5 (parent docs/N1F5_PROTOCOL.md): opt-in external vectors
    "rvg/providers/claude_code.py",
    "rvg/universal/scoped.py",
    "rvg/universal/handoff.py",  # PROD-1 (parent docs/PROD_1_PROTOCOL.md): opt-in handoff packet
    "rvg/universal/cache.py",  # PROD-2 (parent docs/PROD_2_PROTOCOL.md): opt-in decision cache
    "rvg/storage/decision_cache.py",  # PROD-2: the decision cache's local SQLite store
}


def index():
    return json.loads((FIXTURE / "index.json").read_text(encoding="utf-8"))


def baseline_source(rel):
    return (FIXTURE / "src" / (rel + ".txt")).read_text(encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Resolve(ast.NodeTransformer):
    def __init__(self, text_of):
        self.text_of = text_of

    def visit_Call(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "prompt"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
        ):
            return ast.Constant(self.text_of(node.args[0].value))
        return node

    def visit_JoinedStr(self, node):
        self.generic_visit(node)
        values = []
        for value in node.values:
            if (
                isinstance(value, ast.FormattedValue)
                and isinstance(value.value, ast.Constant)
                and isinstance(value.value.value, str)
                and value.conversion == -1
                and value.format_spec is None
            ):
                value = ast.Constant(value.value.value)
            if isinstance(value, ast.Constant) and values and isinstance(values[-1], ast.Constant):
                values[-1] = ast.Constant(values[-1].value + value.value)
            else:
                values.append(value)
        node.values = values
        return node


def _drop_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                del body[0]
                if not body and not isinstance(node, ast.Module):
                    body.append(ast.Pass())


def _drop_prompt_import(tree):
    tree.body = [
        n
        for n in tree.body
        if not (
            isinstance(n, ast.ImportFrom)
            and n.level >= 1
            and n.module == "prompts"
            and [a.name for a in n.names] == ["prompt"]
        )
    ]


def _drop_methods(tree, methods):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in methods:
            node.body = [
                n
                for n in node.body
                if not (
                    isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name in methods[node.name]
                )
            ]


def normalized(source, text_of, *, added_methods=None):
    tree = ast.parse(source)
    _drop_docstrings(tree)
    _drop_prompt_import(tree)
    if added_methods:
        _drop_methods(tree, added_methods)
    tree = _Resolve(text_of).visit(tree)
    return ast.dump(tree)


def compare(rel, candidate_source, text_of):
    """(equal, detail) for one module that differs from the baseline in bytes."""
    before = normalized(baseline_source(rel), text_of)
    after = normalized(candidate_source, text_of, added_methods=ADDED_METHODS.get(rel))
    return before == after
