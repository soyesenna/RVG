"""G04-API: a consumer written against the public API only (`examples/api/public_consumer.py`) runs
every neutral operation end to end from an empty working directory."""

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/api/public_consumer.py"
STDLIB = set(sys.stdlib_module_names)


def test_the_consumer_imports_only_public_names():
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module)
    rvg_modules = {m for m in modules if m.split(".")[0] == "rvg"}
    assert rvg_modules == {"rvg", "rvg.testing.fake_openrouter"}  # the test double only
    assert {m.split(".")[0] for m in modules} - STDLIB <= {"rvg", "httpx"}
    used = {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "rvg"
    }
    import rvg

    assert used <= set(rvg.__all__), used - set(rvg.__all__)


def test_the_consumer_runs_every_operation(tmp_path):
    out = subprocess.run(
        [sys.executable, str(EXAMPLE), str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    ).stdout
    result = json.loads(out)
    assert result["ingest"] == ["STORED", "STORED"] and result["event"] == "APPLIED"
    assert result["answer"]["status"] == "FULL" and result["answer"]["short_answer"] == "500"
    assert result["answer_without_permission"] == "ABSTAIN"
    assert result["health"] == {"liveness": "OK", "readiness": "OK"}
    assert result["lineage"] == {"completeness": "COMPLETE", "kinds": ["evidence_unit", "vector"]}
    assert result["lineage_of_another_principal"] == "lineage_reference_unknown"
    # Only the deployment's own files: the store and its N1 matrices beside it (the default).
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.sqlite3", "state.sqlite3.dense"]
