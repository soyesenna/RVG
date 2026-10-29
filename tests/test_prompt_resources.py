"""SEP-04 A2: prompt and judgement-question texts as package resources — exact bytes, strict
checks, no fallback, nothing read by `import rvg`."""

import hashlib
import json
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest

from rvg import prompts
from rvg.prompts import PromptResourceError, read_manifest, read_prompt, verify_all

ROOT = Path(str(files("rvg") / "resources" / "prompts"))


def test_every_text_is_the_file_bytes_the_manifest_records():
    body = read_manifest()
    assert (
        body["schema"] == "rvg-prompt-manifest/1" and len(body["prompts"]) == 109
    )  # N6-2: control.writer.FORM; PROD-2: universal.cache.EQUIVALENCE
    for prompt_id, entry in body["prompts"].items():
        data = (ROOT / entry["path"]).read_bytes()
        assert len(data) == entry["bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert prompts.prompt(prompt_id).encode("utf-8") == data  # no strip, no newline


def test_the_manifest_lists_every_file_and_nothing_else():
    listed = {e["path"] for e in read_manifest()["prompts"].values()}
    on_disk = {
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    assert listed == on_disk


def test_verify_all_reports_the_installed_manifest():
    report = verify_all()
    assert report == {
        "ok": True,
        "count": 109,
        "manifest_sha256": prompts.manifest_sha256(),
        "failures": [],
    }


def test_module_texts_are_the_resources():
    from rvg.control import verify, writer
    from rvg.temporal import engine

    assert verify.SUPPORT == prompts.prompt("control.verify.SUPPORT")
    assert writer.BASE.startswith("Answer using only the supplied evidence")
    assert engine.CLAIM_CRITERIA["CONFLICTING"] == prompts.prompt(
        "temporal.engine.CLAIM_CRITERIA.CONFLICTING"
    )


def copy(tmp_path):
    target = tmp_path / "prompts"
    shutil.copytree(ROOT, target)
    return target


def test_a_changed_byte_is_an_error_not_a_fallback(tmp_path):
    root = copy(tmp_path)
    entry = read_manifest(root)["prompts"]["control.writer.BASE"]
    path = root / entry["path"]
    path.write_bytes(path.read_bytes() + b"\n")  # one trailing newline
    with pytest.raises(PromptResourceError) as caught:
        read_prompt("control.writer.BASE", root)
    assert caught.value.code == "prompt_resource_hash_mismatch"
    report = verify_all(root)
    assert not report["ok"] and report["failures"] == [
        {"id": "control.writer.BASE", "code": "prompt_resource_hash_mismatch"}
    ]


def test_a_missing_file_or_unknown_id_is_an_error(tmp_path):
    root = copy(tmp_path)
    entry = read_manifest(root)["prompts"]["control.verify.SUPPORT"]
    (root / entry["path"]).unlink()
    with pytest.raises(PromptResourceError) as caught:
        read_prompt("control.verify.SUPPORT", root)
    assert caught.value.code == "prompt_resource_missing"
    with pytest.raises(PromptResourceError) as caught:
        read_prompt("control.verify.NOT_A_PROMPT", root)
    assert caught.value.code == "prompt_id_unknown"


def test_a_missing_manifest_is_an_error(tmp_path):
    root = copy(tmp_path)
    (root / "manifest.json").unlink()
    with pytest.raises(PromptResourceError) as caught:
        read_manifest(root)
    assert caught.value.code == "prompt_manifest_missing"
    assert verify_all(root)["failures"] == [{"id": None, "code": "prompt_manifest_missing"}]


def test_a_manifest_of_another_schema_is_refused(tmp_path):
    root = copy(tmp_path)
    body = json.loads((root / "manifest.json").read_text())
    (root / "manifest.json").write_text(json.dumps(body | {"schema": "other/1"}))
    with pytest.raises(PromptResourceError):
        read_manifest(root)


PROBE = r"""
import json, sys
opened = []
sys.addaudithook(lambda e, a: opened.append(str(a[0])) if e == "open" and a else None)
import rvg
print(json.dumps([p for p in opened if "resources" in p]))
"""


def test_import_rvg_reads_no_prompt():
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []
