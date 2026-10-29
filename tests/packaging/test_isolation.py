"""RVG stands alone: no module imports the research repository, a benchmark, its scripts or its
tests, and a full answer reads no file outside the installed package, the interpreter and the
temporary directory — in particular nothing next to a checkout (a research repository's
`configs/`, `data/`, `runs/` or gold files)."""

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/rvg"
FORBIDDEN_ROOTS = ("deepjev", "scripts", "benchmark", "tests", "conftest")

PROBE = r"""
import json, os, site, sys, sysconfig, tempfile
from pathlib import Path
opened = []
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        opened.append(os.fsdecode(args[0]))
    elif event in ("socket.connect", "socket.getaddrinfo"):
        opened.append("socket:" + str(args[1] if len(args) > 1 else args))
sys.addaudithook(hook)
import importlib, pkgutil
import rvg
for m in pkgutil.walk_packages(rvg.__path__, "rvg."):
    if not m.name.endswith("__main__"):
        importlib.import_module(m.name)
sys.path.insert(0, sys.argv[1])
from smoke import main
main()
allowed = {Path(p).resolve() for p in (
    Path(rvg.__file__).parent, sys.prefix, sys.base_prefix, sys.exec_prefix,
    tempfile.gettempdir(), sys.argv[1], *site.getsitepackages(), sysconfig.get_path("stdlib"),
)}
system = ("/dev/", "/etc/", "/proc/", "/usr/share/zoneinfo/", "/System/Library/CoreServices/")
outside = sorted({
    p for p in opened
    if p.startswith("socket:") or not (
        p.startswith(system) or any(Path(p).resolve().is_relative_to(a) for a in allowed)
    )
})
checkout, around = Path(sys.argv[2]).resolve(), Path(sys.argv[2]).resolve().parent
beside = sorted({
    p for p in opened
    if not p.startswith("socket:")
    and Path(p).resolve().is_relative_to(around)
    and not Path(p).resolve().is_relative_to(checkout)
    and not any(Path(p).resolve().is_relative_to(a) for a in allowed)
})
print(json.dumps({"opened": len(opened), "outside": outside, "beside": beside, "rvg": rvg.__file__}))
"""


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            yield node.module


def test_no_module_imports_the_research_repository_or_a_benchmark():
    offenders = [
        f"{path.relative_to(ROOT)}:{name}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for name in imports(path)
        if name.split(".")[0] in FORBIDDEN_ROOTS
    ]
    assert offenders == []


def test_no_module_names_a_path_outside_its_package():
    offenders = [
        f"{path.relative_to(ROOT)}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if "parents[" in path.read_text(encoding="utf-8")
        or "sys.path" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_an_answer_reads_nothing_outside_the_package_interpreter_and_temp(tmp_path):
    out = (
        subprocess.run(
            [sys.executable, "-c", PROBE, str(Path(__file__).resolve().parent), str(ROOT)],
            capture_output=True,
            text=True,
            check=True,
            cwd=tmp_path,
        )
        .stdout.strip()
        .splitlines()
    )
    smoke, report = json.loads(out[0]), json.loads(out[-1])
    assert smoke["status"] == "FULL"
    assert report["opened"] > 0 and report["outside"] == []
    assert report["beside"] == []  # nothing next to the checkout, e.g. a research repository
