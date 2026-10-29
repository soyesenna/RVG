"""Check a built wheel and sdist before anyone installs them: only the package (and, in the
sdist, its tests, docs and examples) is inside; the policy and schema resources are present
with the pinned policy bytes; every prompt text of the prompt manifest is present with its
recorded bytes and no prompt file is unlisted; no dataset, store, tape, gold/label, environment
or key-shaped file or string is inside. Prints a JSON report; exits 1 on any finding.

    uv build --out-dir dist && python scripts/check_dist.py dist/*.whl dist/*.tar.gz
"""

import hashlib
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

POLICY_SHA256 = "24777fc8dfe3a16fb6225238147dc015b498e973464c25b40fa6e1f81753bae8"
PROMPT_MANIFEST = "rvg/resources/prompts/manifest.json"
REQUIRED = (
    PROMPT_MANIFEST,
    "rvg/__init__.py",
    "rvg/py.typed",
    "rvg/resources/policies/universal.json",
    "rvg/resources/schemas/runtime_request.schema.json",
    "rvg/resources/schemas/evidence_unit.schema.json",
    "rvg/resources/schemas/trace_node.schema.json",
)
FORBIDDEN_NAME = re.compile(
    r"(\.jsonl|\.sqlite3?|\.db|\.parquet|\.npy|\.npz|\.pkl|\.pickle|\.env|\.pem|\.key)$"
    r"|(^|/)(runs|data|\.venv|__pycache__)/"
    r"|(^|/)(labels?|gold)[^/]*$|tape",
    re.IGNORECASE,
)
SECRET = re.compile(
    r"sk-or-v1-[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"""|(?i:(api[_-]?key|token|secret|password))\s*[=:]\s*["'][A-Za-z0-9_\-]{16,}["']"""
)


def members(path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as z:
            return {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    with tarfile.open(path) as t:
        return {
            m.name.split("/", 1)[1]: t.extractfile(m).read()
            for m in t.getmembers()
            if m.isfile() and "/" in m.name
        }


def check(path):
    files, findings = members(path), []
    wheel = path.suffix == ".whl"
    prefix = "" if wheel else "src/"
    for name in files:
        if wheel and not (name.startswith("rvg/") or ".dist-info/" in name):
            findings.append(f"outside_package:{name}")
        if (
            not wheel
            and not name.startswith(("src/rvg/", "tests/", "docs/", "examples/", "scripts/"))
            and name
            not in (
                "README.md",
                "README.ko.md",
                "CHANGELOG.md",
                "CONTRIBUTING.md",
                "SECURITY.md",
                "SECURITY.ko.md",
                "LICENSE",
                "NOTICE",
                ".env.example",
                "pyproject.toml",
                "PKG-INFO",
                ".gitignore",
            )
        ):
            findings.append(f"unexpected_sdist_file:{name}")
        if FORBIDDEN_NAME.search(name):
            findings.append(f"forbidden_name:{name}")
        text = files[name].decode("utf-8", errors="ignore")
        if SECRET.search(text):
            findings.append(f"secret_shaped_string:{name}")
    for name in REQUIRED:
        if prefix + name not in files:
            findings.append(f"missing:{prefix + name}")
    policy = files.get(prefix + "rvg/resources/policies/universal.json")
    if policy is not None and hashlib.sha256(policy).hexdigest() != POLICY_SHA256:
        findings.append("policy_bytes_changed")
    findings += prompt_findings(files, prefix)
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "entries": len(files),
        "findings": findings,
        "listing": sorted(files),
    }


def prompt_findings(files, prefix):
    """Every prompt text the manifest lists is inside with its recorded size and sha256, and no
    prompt file is outside the manifest (SEP-04: the texts a model receives)."""
    manifest = files.get(prefix + PROMPT_MANIFEST)
    if manifest is None:
        return []  # reported as missing:… above
    root = prefix + "rvg/resources/prompts/"
    entries = json.loads(manifest)["prompts"]
    findings = []
    for prompt_id, entry in entries.items():
        data = files.get(root + entry["path"])
        if data is None:
            findings.append(f"missing_prompt:{prompt_id}")
        elif len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            findings.append(f"prompt_bytes_changed:{prompt_id}")
    listed = {root + e["path"] for e in entries.values()} | {prefix + PROMPT_MANIFEST}
    findings += [
        f"unlisted_prompt_file:{n}" for n in files if n.startswith(root) and n not in listed
    ]
    return findings


def main():
    reports = [check(Path(p)) for p in sys.argv[1:]]
    print(json.dumps(reports, indent=1))
    return 1 if any(r["findings"] for r in reports) or not reports else 0


if __name__ == "__main__":
    raise SystemExit(main())
