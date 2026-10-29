"""N1F-1: the dense matrix of a file store is built once per generation by all processes of a
host together, a stalled builder cannot publish over its successor, a crashed one is taken over,
readers keep the generation they mapped, and failures are errors — never a half-written matrix."""

import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pytest
from matrix_worker import (
    DIMS,
    QUERY,
    SPACE,
    doc,
    first_queries,
    open_store,
    populate,
    run,
)

from rvg.storage.local_exact import MatrixDirectory, canonical_scores, unit
from rvg.temporal.store import TemporalRetriever

CTX = multiprocessing.get_context("spawn")


def reference(path, top_k=5):
    """The canonical exact ranking computed directly from the stored vectors (no index code):
    row-normalised float64 matrix times the normalised query in the canonical summation order
    (`canonical_scores`, `unit`: no BLAS), stable descending order."""
    store = open_store(path)
    rows = []
    for d in store.documents("t"):
        v = store.vector("t", SPACE.fingerprint, TemporalRetriever.input_text(d))
        rows.append((d.key, v))
    m = np.asarray([v for _, v in rows])
    q = unit(QUERY)
    scores = canonical_scores(m / np.linalg.norm(m, axis=1, keepdims=True), q)
    order = np.argsort(-scores, kind="stable")[:top_k]
    store.close()
    return [rows[i][0] for i in order], [float(scores[i]) for i in order]


def files(path):
    return [f for f in Path(f"{path}.dense").rglob("*") if f.is_file()]


def spawn(path, out, concurrency, **kw):
    p = CTX.Process(target=run, args=(str(path), str(out), concurrency), kwargs=kw)
    p.start()
    return p


def read(out):
    return json.loads(Path(out).read_text())


def matrix_dir(path):
    return MatrixDirectory(Path(f"{path}.dense"), "t", SPACE.fingerprint)


def test_processes_publish_one_matrix_per_generation(tmp_path):
    path = tmp_path / "s.sqlite3"
    populate(path)
    barrier = CTX.Barrier(4)
    outs = [tmp_path / f"p{i}.json" for i in range(4)]
    procs = [spawn(path, out, 8, barrier=barrier) for out in outs]
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    reports = [read(o) for o in outs]
    assert sum(r["stats"]["builds"] for r in reports) == 1  # one publish for 4 x 8 first queries
    assert sum(r["stats"]["adopted"] for r in reports) == 3
    seeds, scores = reference(path)
    for r in reports:
        assert all(x == {"seed_ids": seeds, "scores": scores} for x in r["results"])
    assert len([f for f in files(path) if f.suffix == ".rvgm"]) == 1
    assert not [f for f in files(path) if f.name.endswith(".tmp")]


def test_a_stalled_builder_is_fenced_and_adopts_its_successor(tmp_path):
    path = tmp_path / "s.sqlite3"
    populate(path)
    stalled = spawn(
        path,
        tmp_path / "a.json",
        1,
        fault={"point": "after_data_written", "action": "sleep", "seconds": 3.0},
        lease=1.0,
    )
    directory = matrix_dir(path)
    deadline = time.time() + 30
    while directory.read_lease().get("pid") != stalled.pid and time.time() < deadline:
        time.sleep(0.02)
    assert directory.read_lease().get("pid") == stalled.pid  # A holds the lease, then stalls
    successor = spawn(path, tmp_path / "b.json", 2, lease=1.0)
    successor.join(60)
    stalled.join(60)
    a, b = read(tmp_path / "a.json"), read(tmp_path / "b.json")
    assert b["stats"]["builds"] == 1 and b["stats"]["takeovers"] == 1
    assert a["stats"]["fenced"] == 1 and a["stats"]["builds"] == 0 and a["stats"]["adopted"] == 1
    seeds, scores = reference(path)
    assert a["results"] == [{"seed_ids": seeds, "scores": scores}]
    assert directory.read_manifest()["token"] == 2  # B's token; A's (1) was never published
    assert not [f for f in files(path) if f.name.endswith(".tmp")]


@pytest.mark.parametrize("point", ["after_data_written", "after_rename"])
def test_a_crashed_builder_is_taken_over_and_its_leftovers_removed(tmp_path, point):
    path = tmp_path / "s.sqlite3"
    populate(path)
    crashed = spawn(path, tmp_path / "a.json", 1, fault={"point": point, "action": "exit"})
    crashed.join(60)
    assert crashed.exitcode == 3 and not (tmp_path / "a.json").exists()
    assert matrix_dir(path).read_manifest() is None  # nothing was published
    import asyncio

    store = open_store(path)
    results = asyncio.run(first_queries(store, 2))
    seeds, scores = reference(path)
    assert results == [{"seed_ids": seeds, "scores": scores}] * 2
    # The dead owner's lease is taken over at once (its process is gone, not expired).
    assert store.index.stats["takeovers"] == 1 and store.index.stats["builds"] == 1
    names = [f.name for f in files(path)]
    assert len([n for n in names if n.endswith(".rvgm")]) == 1
    assert not [n for n in names if n.endswith(".tmp")]


def test_a_reader_keeps_its_generation_while_another_process_publishes(tmp_path):
    import asyncio

    path = tmp_path / "s.sqlite3"
    populate(path)
    store = open_store(path)
    before = asyncio.run(first_queries(store, 1))
    catalog = asyncio.run(store.storage.catalogs.get("t"))
    held = asyncio.run(store.index.view("t", SPACE.fingerprint, DIMS, catalog))  # pins gen 1
    old_file = held._snapshot.file
    writer = open_store(path)
    extra = doc(999, "A new, very relevant fact.")
    writer.upsert("t", extra)
    writer.put_vector("t", SPACE.fingerprint, TemporalRetriever.input_text(extra), QUERY)
    writer.close()
    other = spawn(path, tmp_path / "b.json", 1)
    other.join(60)
    b = read(tmp_path / "b.json")
    assert b["stats"]["builds"] == 1 and b["stats"]["gc_pinned"] == 1  # gen 1 is pinned here
    assert b["results"][0]["seed_ids"][0] == "d999@1"
    assert (Path(f"{path}.dense") / "v2").exists() and any(f.name == old_file for f in files(path))
    mask = np.ones(len(catalog), dtype=bool)
    rows, _ = held.rank(mask, QUERY, 5)  # still the old generation, still exact
    assert [catalog.documents[i].key for i in rows] == before[0]["seed_ids"]
    del held
    after = asyncio.run(first_queries(store, 1))  # adopts gen 2, drops gen 1, collects it
    assert after[0]["seed_ids"][0] == "d999@1"
    assert not any(f.name == old_file for f in files(path))


def test_a_full_disk_is_an_error_and_leaves_nothing_behind(tmp_path):
    import asyncio

    path = tmp_path / "s.sqlite3"
    populate(path)
    store = open_store(path, fault={"point": "during_write", "action": "enospc"})
    results = asyncio.run(first_queries(store, 3))
    assert results == [{"error": "dense_build_io"}] * 3  # one failed build, one error each
    assert matrix_dir(path).read_manifest() is None
    assert not [f for f in files(path) if f.suffix in (".rvgm", ".tmp")]
    assert matrix_dir(path).read_lease()["token"] is None  # released, not left to expire
    healthy = open_store(path)
    seeds, scores = reference(path)
    assert asyncio.run(first_queries(healthy, 1)) == [{"seed_ids": seeds, "scores": scores}]


def test_a_damaged_published_matrix_is_rebuilt(tmp_path):
    import asyncio

    path = tmp_path / "s.sqlite3"
    populate(path)
    asyncio.run(first_queries(open_store(path), 1))
    published = next(f for f in files(path) if f.suffix == ".rvgm")
    published.write_bytes(b"not a matrix")
    store = open_store(path)
    seeds, scores = reference(path)
    assert asyncio.run(first_queries(store, 1)) == [{"seed_ids": seeds, "scores": scores}]
    assert store.index.stats["builds"] == 1


def test_publication_is_monotonic_and_fenced(tmp_path):
    directory = MatrixDirectory(tmp_path, "t", "space")
    own = directory.acquire((2,), now=0, ttl=10)
    assert own[0] == "own" and own[1] == 1
    assert directory.acquire((2,), now=1, ttl=10)[0] == "wait"  # same generation: wait
    assert directory.acquire((1,), now=1, ttl=10)[0] == "wait"  # a newer one is being built
    assert directory.acquire((3,), now=1, ttl=10)[0] == "wait"  # a live builder is waited for
    newer = directory.acquire((3,), now=11, ttl=10)  # expired: taken over, token 1 fenced
    assert newer[0] == "own" and newer[1] == 2
    assert not directory.renew(1, now=2, ttl=10)
    tmp = directory.dir / "m-2-1.rvgm.tmp"
    tmp.write_bytes(b"x")
    assert directory.publish(1, (2,), tmp, "m-2-1.rvgm", {}, lambda p: None) is None
    tmp3 = directory.dir / "m-3-2.rvgm.tmp"
    tmp3.write_bytes(b"y")
    manifest = directory.publish(2, (3,), tmp3, "m-3-2.rvgm", {}, lambda p: None)
    done = directory.acquire((2,), now=3, ttl=10)  # a newer published generation serves it
    assert manifest["token"] == 2 and done[0] == "done" and done[1]["stamp"] == [3]
    assert directory.gc() == (1, 0)  # the fenced builder's leftover temporary file
