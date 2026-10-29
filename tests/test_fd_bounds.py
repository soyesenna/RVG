"""0.1.0 (parent docs/PUB_4_REPORT.md): open files stay bounded. A long-lived local exact index
lets go of a superseded generation when it catches up, however many times it does, and closing a
deployment closes every generation's pinned file and memory map at once — not when the garbage
collector reaches the cycle a store and its storage form (a clone's suite ran out of descriptors
under the macOS default `ulimit -n 256`). The collector is off while these count, so a release
that only a collection would make fails here."""

import asyncio
import gc
import os
import shutil
import tempfile
from pathlib import Path

import httpx
import numpy as np
import pytest
from matrix_worker import DIMS, doc
from test_incremental_layouts import FP, QUERIES, deployment, reference, search

import rvg
from rvg.temporal.store import TemporalRetriever, TemporalStore

ROUNDS = 50
# Open files one deployment may hold while it serves: its state store and, per layout, the
# generation it reads (E2 segments and E3 pages are several files). A leak of one descriptor per
# catch-up or per session passes it well before the last round.
BOUND = 24
RECIPES = {
    "E0H": None,
    "base_delta": {"theta": 0.25},
    "segments": {"base_rows": 4, "fanout": 2},
    "paged": {"page_rows": 8},
}
AS_OF = "2026-09-20T00:00:00Z"


def open_files():
    return len(os.listdir("/dev/fd"))


@pytest.fixture
def no_collector():
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


@pytest.mark.parametrize("layout", list(RECIPES))
def test_fifty_catch_ups_of_one_index_keep_its_open_files_bounded(no_collector, layout):
    root = Path(tempfile.mkdtemp())
    try:
        before = open_files()
        store = TemporalStore(root / "s.sqlite3", dense_memmap_dir=None)
        rng = np.random.default_rng(1)
        docs = [doc(i) for i in range(30)]
        for d in docs:
            store.upsert("t", d)
            store.put_vector("t", FP, TemporalRetriever.input_text(d), rng.normal(size=DIMS))
        arm = deployment(store, root, layout, RECIPES[layout])
        held = []
        for i in range(ROUNDS):  # a new vector each round: a new generation to catch up to
            store.put_vector(
                "t", FP, TemporalRetriever.input_text(docs[i % 30]), rng.normal(size=DIMS)
            )
            assert search(arm, QUERIES[0]) == reference(store, QUERIES[0])
            held.append(open_files() - before)
        assert arm.index.stats["builds"] == ROUNDS
        if layout != "E0H":
            assert arm.index.stats["catch_ups"] == ROUNDS - 1
        assert max(held) <= BOUND, held
        arm.close()
        assert open_files() == before
        arm.close()  # idempotent
    finally:
        shutil.rmtree(root, ignore_errors=True)


def revision(i):
    text = f"The budget of project {i} is {100 + i}."
    return {
        "id": f"p{i:03d}",
        "family": f"p{i:03d}",
        "revision": 1,
        "title": f"P{i}",
        "text": text,
        "source": f"example://{i}",
        "published_at": "2026-09-01T00:00:00Z",
        "observed_at": "2026-09-01T00:00:00Z",
        "units": [{"id": "u0", "start": 0, "end": len(text)}],
    }


def test_fifty_sessions_on_a_local_exact_deployment_close_what_they_open(no_collector, monkeypatch):
    for name in ("DEEPJEV_MODE", "RVG_MODE"):
        monkeypatch.delenv(name, raising=False)

    async def refuse(self, request, **kwargs):
        raise AssertionError(f"mock mode sent a request to {request.url.host}")

    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)
    storage = {"schema_version": 1, "state": {"backend": "sqlite", "path": "state.sqlite3"}}
    settings = rvg.Settings(_env_file=None)
    root = Path(tempfile.mkdtemp())

    async def round_(i):  # one more document each round: the next session catches up to it
        start = open_files()  # with the event loop's own descriptors
        session = rvg.open_session(storage, settings, root=root)
        async with session:
            context = rvg.RequestContext(tenant="example")
            await session.ingest([revision(i)], context)
            envelope = await session.answer(
                {"text": f"What is the budget of project {i}?", "as_of": AS_OF, "known_at": AS_OF},
                context,
            )
            during = open_files()
        # Closed, and still referenced: `aclose` let go of what the session opened.
        return envelope["result"]["status"], during, open_files() - start

    try:
        before = open_files()
        held, closed, left = [], [], []
        for i in range(ROUNDS):
            status, during, after = asyncio.run(round_(i))
            assert status == "FULL"
            held.append(during - before)
            closed.append(after)
            left.append(open_files() - before)
        assert (root / "state.sqlite3.dense").is_dir()  # the local exact matrix in files
        assert max(held) <= BOUND, held
        assert closed == [0] * ROUNDS, closed
        assert left == [0] * ROUNDS, left
    finally:
        shutil.rmtree(root, ignore_errors=True)
