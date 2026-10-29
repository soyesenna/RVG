"""The external float16 example (`examples/storage/external_float16.py`) runs end to end from an
empty working directory: a set exported from a plain deployment serves a new deployment the same
answer, with no document embedded again, at precision EXACT_NATIVE."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/storage/external_float16.py"


def test_the_external_example_answers_the_same_from_the_set(tmp_path):
    out = subprocess.run(
        [sys.executable, str(EXAMPLE), str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    ).stdout
    result = json.loads(out)
    plain, external = result["plain"], result["external"]
    assert plain["status"] == "FULL" and plain["short_answer"] == "500"
    assert result["same_answer"] is True
    assert (plain["index"], external["index"]) == ("local_exact", "external_float16")
    assert external["precision"] == "EXACT_NATIVE"
    assert (plain["documents_embedded"], external["documents_embedded"]) == (2, 0)
    assert result["set"] == ["hashes.bin", "manifest.json", "shard-00000.npy"]
    config = json.loads((tmp_path / "storage.json").read_text())
    assert config["vectors"] == {
        "source": "external_float16",
        "external": {"manifest": "vectors/manifest.json"},
    }
