"""The universality contract, checked mechanically.

One policy file for every knowledge base, language and request; no per-KB profile, threshold or
retrieval choice; no labels in runtime inputs; the product never imports the offline tooling.
The research repository keeps the checks of its own CLIs (evaluation cases, entry points).
"""

import ast
import dataclasses
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from rvg.config import Settings
from rvg.control.controller import UNIVERSAL_PROFILE, ControlPolicy
from rvg.control.signals import SIGNALS
from rvg.universal import PATH, universal, universal_hash

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/rvg"
TOOLING = PACKAGE / "testing"
# The adopted policy. Changing it is a global change that goes through the research protocol
# and a new release; this pin makes that change explicit.
POLICY_SHA256 = "24777fc8dfe3a16fb6225238147dc015b498e973464c25b40fa6e1f81753bae8"


def product_files():
    for path in sorted(PACKAGE.rglob("*.py")):
        if TOOLING not in path.parents:
            yield path


def imported_names(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield (node.module or "") + "." + ".".join(a.name for a in node.names)


def test_the_policy_file_is_a_package_resource_hashed_and_naming_only_known_signals():
    assert Path(str(PATH)).parts[-4:] == ("rvg", "resources", "policies", "universal.json")
    assert PATH.is_file()
    assert len(universal_hash()) == 64 and universal()["version"].startswith("universal-")
    named = {s for own in universal()["profile"].values() for s in own}
    assert named <= set(SIGNALS)
    assert set(universal()["statuses"]) == {
        "FULL",
        "PARTIAL",
        "ABSTAIN",
        "UNRESOLVED_SCOPE",
        "ERROR",
    }


def test_the_adopted_policy_bytes_are_pinned():
    assert universal_hash() == POLICY_SHA256


def test_the_controller_takes_every_default_from_the_policy_file():
    cfg, policy = universal(), ControlPolicy()
    assert (
        policy.limits()
        == UNIVERSAL_PROFILE
        == {g: {s: t for s, t in own.items() if t > 0} for g, own in cfg["profile"].items()}
    )
    for name, value in cfg["floor"].items():
        assert getattr(policy, name) == value
    for name, value in cfg["controller"].items():
        assert getattr(policy, name) == value
    fields = {f.name for f in dataclasses.fields(ControlPolicy)}
    assert "profile" not in fields and not {f for f in fields if "threshold" in f} - set(
        cfg["floor"]
    )


def test_model_defaults_agree_with_the_policy_file():
    models, defaults = universal()["models"], Settings.model_fields
    assert defaults["openrouter_jev_model"].default == models["decision"]
    assert defaults["surface_proposer_model"].default == models["writer"]
    assert defaults["semantic_proposer_model"].default == models["extractor"]
    assert defaults["proposer_provider"].default == models["generation_provider"]
    assert defaults["embedding_model"].default == models["embedding"]
    # D-DIM-DECL: the policy's 3,072 is a declaration no code reads; the default is the 1,536
    # every measurement used (0.1.0, parent docs/PUB_2_REPORT.md)
    assert (defaults["embedding_dimensions"].default, models["embedding_dimensions"]) == (
        1536,
        3072,
    )


def test_product_modules_never_import_diagnostics_evaluation_or_the_research_package():
    offenders = [
        f"{path.relative_to(ROOT)}:{name}"
        for path in product_files()
        for name in imported_names(path)
        if "diagnostics" in name
        or name.endswith(".calibrate")
        or "evaluation" in name.split(".")
        or name.split(".")[0] in ("deepjev", "scripts", "benchmark")
    ]
    assert offenders == []


def test_product_modules_never_import_the_testing_tools():
    offenders = [
        f"{path.relative_to(ROOT)}:{name}"
        for path in product_files()
        for name in imported_names(path)
        if name.startswith(("rvg.testing", ".testing", "..testing"))
    ]
    assert offenders == []


def test_product_modules_carry_no_knowledge_base_or_language_branches():
    banned = ("korean", "finance", "fullwiki", "hotpot", "squad", "klue", "companies.json")
    offenders = []
    for path in product_files():
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}:{token}")
    assert offenders == []


def test_the_policy_file_is_json_with_no_per_knowledge_base_sections():
    text = json.loads(PATH.read_text(encoding="utf-8"))
    assert set(text) == {
        "version",
        "note",
        "models",
        "retrieval",
        "floor",
        "profile",
        "controller",
        "cascade",
        "validity",
        "release",
        "scope",
        "statuses",
    }
    assert set(text["profile"]) == {"*"}  # one map, no groups, no knowledge bases


def test_product_modules_carry_no_language_branches():
    pattern = re.compile(
        r'(Literal\[[^\]]*"(?:ko-KR|en-US)"|choices=\[[^\]]*"(?:ko-KR|en-US)"|[!=]=\s*"(?:ko-KR|en-US)")'
    )
    offenders = [
        f"{path.relative_to(ROOT)}:{m.group(0)}"
        for path in product_files()
        for m in pattern.finditer(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []  # locale is an output-language tag, never a policy switch


@pytest.mark.parametrize(
    "field",
    [
        "gold_answer",
        "relevant_doc_ids",
        "expected_status",
        "dataset_name",
        "wrong",
        # N3: a principal and its ACL epoch come from the trusted server context only.
        "principal_id",
        "acl_epoch",
    ],
)
def test_the_engine_question_rejects_labels_and_dataset_names(field):
    from rvg.temporal.models import TemporalQuestion

    base = {"text": "q?", "as_of": "2026-09-20T00:00:00Z", "known_at": "2026-09-20T00:00:00Z"}
    with pytest.raises(ValidationError):
        TemporalQuestion.model_validate(base | {field: "x"})


def test_product_entry_points_never_build_an_experiment_arm():
    """Arms (N2/N3) are handed to the engine by evaluation runners only."""
    for rel in ("src/rvg/universal/cli.py",):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "experiment" not in text and "arm=" not in text, rel
