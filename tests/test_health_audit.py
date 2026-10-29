"""G04-OBS 1: liveness opens no socket and no database connection, reads nothing outside the
installed package, and sends nothing to a provider; readiness adds only reads of the already open
store. Measured by an audit hook in a fresh process, not by the report's own fields."""

import json
import subprocess
import sys
from pathlib import Path

PROBE = r"""
import asyncio, json, os, sys
from pathlib import Path
import httpx
import rvg
from rvg.testing.fake_openrouter import FakeOpenRouter, providers

async def main(root):
    fake = FakeOpenRouter()
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
        p = providers(rvg.JsonTransport(client, timeout=5, max_attempts=3))
        config = {"schema_version": 1, "state": {"backend": "sqlite", "path": "s.sqlite3"}}
        async with rvg.open_session(config, rvg.Settings(_env_file=None), root=root, providers=p) as session:
            ctx = rvg.RequestContext(tenant="t")
            await session.health("readiness", ctx)  # warm imports and caches first
            await session.health("liveness")
            sent = len(fake.bodies)
            events = []
            def hook(event, args):
                if event in ("socket.connect", "socket.getaddrinfo", "sqlite3.connect"):
                    events.append([event, ""])
                elif event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
                    events.append([event, os.fsdecode(args[0])])
            sys.addaudithook(hook)
            live = await session.health("liveness")
            live_events = list(events)
            events.clear()
            ready = await session.health("readiness", ctx)
            ready_events = list(events)
            print(json.dumps({
                "live": live["status"], "ready": ready["status"],
                "live_events": live_events, "ready_events": ready_events,
                "provider_requests": len(fake.bodies) - sent,
                "package": str(Path(rvg.__file__).resolve().parent),
            }))

asyncio.run(main(Path(sys.argv[1])))
"""


def test_liveness_and_readiness_touch_nothing_they_should_not(tmp_path):
    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    ).stdout
    report = json.loads(out.strip().splitlines()[-1])
    assert report["live"] == "OK" and report["ready"] == "OK"
    assert report["provider_requests"] == 0
    package = Path(report["package"])
    for key in ("live_events", "ready_events"):
        kinds = {e[0] for e in report[key]}
        assert not kinds & {"socket.connect", "socket.getaddrinfo", "sqlite3.connect"}, key
        opened = [Path(p).resolve() for e, p in report[key] if e == "open"]
        assert all(p.is_relative_to(package) for p in opened), opened  # package resources only
