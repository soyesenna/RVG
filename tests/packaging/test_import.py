"""`import rvg` is light and side-effect free: it names its version and public API, loads no
runtime module, reads no file outside the interpreter and opens no socket or database."""

import json
import subprocess
import sys

import rvg

PROBE = r"""
import json, sys
events = []
def hook(event, args):
    if event in ("open", "socket.connect", "socket.getaddrinfo", "sqlite3.connect"):
        events.append([event, str(args[0]) if args else ""])
sys.addaudithook(hook)
before = set(sys.modules)
import rvg
loaded = sorted(m for m in set(sys.modules) - before if m.split(".")[0] != "encodings")
print(json.dumps({"loaded": loaded, "events": events, "version": rvg.__version__}))
"""


def probe():
    out = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)


def test_the_package_has_a_version_and_a_public_api():
    assert rvg.__version__ == "0.1.0"
    assert "UniversalEngine" in rvg.__all__ and "capabilities" in rvg.__all__


def test_importing_the_package_loads_no_runtime_module():
    loaded = probe()["loaded"]
    assert [m for m in loaded if m.startswith("rvg.")] == []
    assert not {"numpy", "httpx", "pydantic", "sqlite3", "psycopg"} & set(loaded)


def test_importing_the_package_opens_no_socket_database_or_data_file():
    events = probe()["events"]
    assert [e for e in events if e[0] != "open"] == []
    opened = [path for _, path in events]
    assert all(p.endswith((".py", ".pyc")) or "__pycache__" in p for p in opened), opened
