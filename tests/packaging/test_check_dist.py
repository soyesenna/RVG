"""`scripts/check_dist.py` finds what must never ship (negative controls, G06-CONTENTS) and what
must ship intact (the pinned policy, every prompt text): synthetic wheels made from the installed
package, one fault planted in each."""

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

import rvg

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(rvg.__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check_dist", ROOT / "scripts/check_dist.py")
check_dist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_dist)

MANIFEST = "rvg/resources/prompts/manifest.json"


def package_files():
    return {
        "rvg/" + str(p.relative_to(PACKAGE)): p.read_bytes()
        for p in sorted(PACKAGE.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def wheel(tmp_path, files):
    path = tmp_path / "rvg_runtime-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("rvg_runtime-0.0.0.dist-info/METADATA", "Name: rvg-runtime\n")
    return check_dist.check(path)["findings"]


def a_prompt(files):
    body = json.loads(files[MANIFEST])
    prompt_id, entry = min(body["prompts"].items())
    return prompt_id, "rvg/resources/prompts/" + entry["path"]


def test_the_installed_package_has_no_findings(tmp_path):
    assert wheel(tmp_path, package_files()) == []


def test_a_gold_file_a_tape_and_a_key_are_found(tmp_path):
    files = package_files() | {
        "rvg/labels.eval.jsonl": b'{"gold": 1}\n',
        "rvg/fixture_tape.json": b"[]",
        # assembled here so that this test file itself holds no key-shaped string (the sdist
        # ships the tests and check_dist scans it)
        "rvg/leak.py": b"API_" + b'KEY = "' + b"abcdefghij" * 3 + b'"\n',
    }
    findings = wheel(tmp_path, files)
    assert "forbidden_name:rvg/labels.eval.jsonl" in findings
    assert "forbidden_name:rvg/fixture_tape.json" in findings
    assert "secret_shaped_string:rvg/leak.py" in findings


def test_a_changed_policy_is_found(tmp_path):
    files = package_files()
    name = "rvg/resources/policies/universal.json"
    files[name] = files[name].replace(b"0.85", b"0.8", 1)
    assert "policy_bytes_changed" in wheel(tmp_path, files)


@pytest.mark.parametrize(
    "fault, expected",
    [
        ("byte", "prompt_bytes_changed"),
        ("newline", "prompt_bytes_changed"),
        ("missing", "missing_prompt"),
    ],
)
def test_a_changed_or_missing_prompt_is_found(tmp_path, fault, expected):
    files = package_files()
    prompt_id, name = a_prompt(files)
    if fault == "byte":
        files[name] = files[name][:-1] + bytes([files[name][-1] ^ 1])
    elif fault == "newline":
        files[name] = files[name] + b"\n"
    else:
        del files[name]
    assert f"{expected}:{prompt_id}" in wheel(tmp_path, files)


def test_an_unlisted_prompt_file_or_a_missing_manifest_is_found(tmp_path):
    files = package_files() | {"rvg/resources/prompts/control/extra.txt": b"Say yes."}
    assert "unlisted_prompt_file:rvg/resources/prompts/control/extra.txt" in wheel(tmp_path, files)
    files = package_files()
    del files[MANIFEST]
    assert f"missing:{MANIFEST}" in wheel(tmp_path, files)


def test_no_file_that_the_sdist_ships_holds_a_key_shaped_string():
    shipped = [
        ROOT / n
        for n in (
            "README.md",
            "README.ko.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "LICENSE",
            "NOTICE",
            ".env.example",
            "pyproject.toml",
        )
    ]
    shipped += [p for p in (ROOT / "SECURITY.ko.md",) if p.exists()]  # 0.1.0: the docs branch
    for folder in ("src", "tests", "docs", "examples", "scripts"):
        shipped += [
            p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        ]
    found = [
        str(p.relative_to(ROOT))
        for p in shipped
        if check_dist.SECRET.search(p.read_bytes().decode("utf-8", errors="ignore"))
    ]
    assert found == []
