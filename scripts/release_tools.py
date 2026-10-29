"""SEP-06 release tools (docs/sep04-06/SEP06_PROTOCOL.md): hash-pinned locks from a wheelhouse,
a static lock check, and an installed-package identity check.

    python scripts/release_tools.py lock --wheelhouse WH --requirements REQS --out LOCK \
        [--add 'rvg-runtime[postgres]==0.1.0.dev1' WHEEL]
    python scripts/release_tools.py check-lock LOCK --wheelhouse WH
    python -m ... release_tools.py identity --expect-version V   # run with the installed Python

`lock` pins every requirement of REQS (exact `name==version` lines, e.g. a target's `uv pip
compile` output) to the sha256 of the one wheel of that name and version in WH — the file that
will be installed, not every file the index lists. `check-lock` refuses anything but
`name[extras]==version --hash=sha256:…` lines and every lock entry must be a file in WH.
`identity` reports where `rvg` was imported from and whether the installed files are the wheel's
(RECORD hashes), and fails on an editable, shadowed or foreign installation."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<extras>\[[A-Za-z0-9,._-]+\])?==(?P<version>[^\s;]+)"
    r"(?:\s*;\s*(?P<marker>[^\\]+?))?\s*(?P<hashes>(?:\s*--hash=sha256:[0-9a-f]{64})+)\s*$"
)
FORBIDDEN = (
    "-e ",
    "--editable",
    "--index-url",
    "--extra-index-url",
    "--find-links",
    "-f ",
    "git+",
    "://",
    "file:",
)


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wheels(wheelhouse):
    """(normalised name, version) -> wheel path, for every wheel in a wheelhouse."""
    out = {}
    for path in sorted(Path(wheelhouse).glob("*.whl")):
        name, version = path.name.split("-")[:2]
        key = (norm(name), version)
        if key in out:
            raise SystemExit(f"two wheels for {key}: {out[key].name}, {path.name}")
        out[key] = path
    return out


def requirement_lines(path):
    """Exact pins of a requirements file (comments, blanks and hash continuations dropped)."""
    text = Path(path).read_text(encoding="utf-8").replace("\\\n", " ")
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.lstrip().startswith("--hash") else ""
        if not line:
            continue
        if any(bad in line for bad in FORBIDDEN):
            raise SystemExit(f"refused requirement: {line}")
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]+\])?==([^\s;]+)", line)
        if not match:
            raise SystemExit(f"not an exact pin: {line}")
        yield match.group(1), match.group(2) or "", match.group(3)


def lock(args):
    available = wheels(args.wheelhouse)
    lines = []
    for name, extras, version in requirement_lines(args.requirements):
        path = available.get((norm(name), version))
        if path is None:
            raise SystemExit(f"missing wheel: {name}=={version}")
        lines.append(f"{norm(name)}{extras}=={version} --hash=sha256:{sha(path)}")
    for spec, wheel in args.add or ():
        lines.append(f"{spec} --hash=sha256:{sha(wheel)}")
    if Path(args.out).exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    Path(args.out).write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    print(json.dumps({"lock": str(args.out), "entries": len(lines)}))


def check_lock(args):
    findings, entries = [], 0
    available = {sha(p): p.name for p in Path(args.wheelhouse).glob("*.whl")}
    for n, raw in enumerate(Path(args.lock).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        entries += 1
        if any(bad in raw for bad in FORBIDDEN):
            findings.append(f"line {n}: forbidden form")
            continue
        match = LINE.match(raw)
        if not match:
            findings.append(f"line {n}: not name==version --hash=sha256:…")
            continue
        for digest in re.findall(r"sha256:([0-9a-f]{64})", match["hashes"]):
            if digest not in available:
                findings.append(f"line {n}: {match['name']} has no wheel in the wheelhouse")
    report = {"lock": str(args.lock), "entries": entries, "findings": findings}
    print(json.dumps(report))
    return 1 if findings or not entries else 0


def identity(args):
    """Run with the Python that has the release installed."""
    import importlib.metadata as md
    import site
    import sysconfig

    import rvg

    here = Path(rvg.__file__).resolve().parent
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    dist = md.distribution("rvg-runtime")
    owners = md.packages_distributions().get("rvg", [])
    direct = dist.read_text("direct_url.json")
    record, mismatched, missing = {}, [], []
    for f in dist.files or []:
        if str(f).startswith("rvg/") and f.hash is not None:
            record[str(f)] = f
    for rel, f in record.items():
        path = Path(f.locate())
        if not path.exists():
            missing.append(rel)
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=")
        if digest.decode() != f.hash.value:
            mismatched.append(rel)
    on_disk = {
        str(p.relative_to(here.parent))
        for p in here.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    unrecorded = sorted(on_disk - set(record))
    report = {
        "python": sys.version.split()[0],
        "prefix": sys.prefix,
        "rvg_file": str(here / "__init__.py"),
        "in_site_packages": here.is_relative_to(purelib),
        "distribution": dist.metadata["Name"],
        "version": dist.version,
        "owners_of_rvg": owners,
        "direct_url": json.loads(direct) if direct else None,
        "record_files": len(record),
        "record_mismatched": mismatched,
        "record_missing": missing,
        "unrecorded_files": {"count": len(unrecorded), "first": unrecorded[:5]},
        "policy_sha256": rvg.policy_hash(),
        "site_packages": [str(p) for p in site.getsitepackages()],
    }
    problems = []
    if not report["in_site_packages"]:
        problems.append("not_imported_from_site_packages")
    if owners != ["rvg-runtime"]:
        problems.append("rvg_module_owned_by_other_distribution")
    if report["direct_url"] is not None:
        problems.append("installed_from_a_url_or_editable")
    if mismatched or missing or unrecorded:
        problems.append("installed_files_differ_from_the_wheel_record")
    if args.expect_version and dist.version != args.expect_version:
        problems.append("unexpected_version")
    report["problems"] = problems
    print(json.dumps(report, indent=1))
    return 1 if problems else 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("lock")
    a.add_argument("--wheelhouse", required=True)
    a.add_argument("--requirements", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--add", nargs=2, action="append", metavar=("SPEC", "WHEEL"))
    b = sub.add_parser("check-lock")
    b.add_argument("lock")
    b.add_argument("--wheelhouse", required=True)
    c = sub.add_parser("identity")
    c.add_argument("--expect-version")
    args = p.parse_args()
    return {"lock": lock, "check-lock": check_lock, "identity": identity}[args.command](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
