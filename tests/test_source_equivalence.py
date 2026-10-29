"""SEP-04 G04-PROMPT/G04-API: every product module of the installed package is accounted for
against the extraction baseline `018c355` — byte-identical, AST-equivalent once its texts are
read from resources (docstrings aside), or a change declared before the work began."""

import hashlib
from pathlib import Path

import pytest
from source_equivalence import (
    ADDED_METHODS,
    INTENDED,
    INTENDED_RESOURCES,
    NEW_MODULES,
    baseline_source,
    compare,
    sha256,
)
from source_equivalence import index as baseline_index

import rvg
from rvg.prompts import prompt

PACKAGE = Path(rvg.__file__).resolve().parent
BASELINE = baseline_index()


def installed(rel):
    return PACKAGE.parent / rel


def changed_modules():
    return sorted(
        rel
        for rel, digest in BASELINE["files"].items()
        if rel.endswith(".py") and sha256(installed(rel)) != digest
    )


def test_the_fixture_is_the_extraction_baseline():
    assert BASELINE["commit"] == "018c355c6bb9cb308191a60380ae9db2458eaa2e"
    assert "rvg/universal/engine.py" in BASELINE["files"]


def test_no_baseline_module_is_missing():
    assert [rel for rel in BASELINE["files"] if not installed(rel).exists()] == []


@pytest.mark.parametrize("rel", changed_modules())
def test_a_changed_module_is_equivalent_or_declared(rel):
    if rel in INTENDED:
        pytest.skip("declared change: " + INTENDED[rel])
    assert rel in BASELINE["copies"], f"no baseline copy of {rel}: re-run snapshot_baseline_sources"
    assert compare(rel, installed(rel).read_text(encoding="utf-8"), prompt), rel


def test_declared_additions_are_the_only_difference_where_declared():
    for rel in ADDED_METHODS:
        if rel in INTENDED:
            continue  # a later declared change covers the whole module
        if sha256(installed(rel)) != BASELINE["files"][rel]:
            assert compare(rel, installed(rel).read_text(encoding="utf-8"), prompt)


def test_new_modules_are_declared():
    new = sorted(
        str(p.relative_to(PACKAGE.parent))
        for p in PACKAGE.rglob("*.py")
        if str(p.relative_to(PACKAGE.parent)) not in BASELINE["files"]
    )
    assert set(new) <= NEW_MODULES, new


def test_non_python_baseline_resources_are_unchanged():
    changed = [
        rel
        for rel, digest in BASELINE["files"].items()
        if not rel.endswith(".py") and sha256(installed(rel)) != digest
    ]
    assert sorted(changed) == sorted(INTENDED_RESOURCES)  # the runtime schemas and py.typed
    for rel, edits in INTENDED_RESOURCES.items():
        text = installed(rel).read_text(encoding="utf-8")
        for now, before in edits:
            assert text.count(now) == 1
            text = text.replace(now, before)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == BASELINE["files"][rel]


def test_a_real_change_would_be_caught():
    rel = "rvg/control/verify.py"
    source = baseline_source(rel)
    assert compare(rel, source, prompt)  # the baseline is equivalent to itself
    assert not compare(rel, source.replace("threshold=0.85", "threshold=0.8"), prompt)
    one_char = source.replace("supported, with its subject", "supported, with the subject", 1)
    assert one_char != source and not compare(rel, one_char, prompt)
