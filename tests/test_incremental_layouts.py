"""N1F-2: the incremental exact layouts (E1 base+delta, E2 segments, E3 paged COW) give the same
ranking as the full-rebuild matrix (E0H) and as an independent reference, bit for bit, through
random sequences of new vectors, rewrites, vectors turning invalid and valid again, revocations
and pending documents — with small recipes so merges, compactions and page boundaries happen
often — and another index adopting what one published continues from it."""

import asyncio
import gc
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
from matrix_worker import DIMS, SPACE, Queries, doc

from rvg.errors import ContractError
from rvg.runtime import CallContext
from rvg.storage.catalog import CatalogCache
from rvg.storage.factory import Storage
from rvg.storage.incremental import IncrementalExactIndex
from rvg.storage.local_exact import LocalExactIndex, canonical_scores, unit
from rvg.storage.release import ReleaseCoordinator
from rvg.temporal.models import TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore

FP = SPACE.fingerprint
RECIPES = {
    "base_delta": {"theta": 0.25},
    "segments": {"base_rows": 2, "fanout": 2},
    "paged": {"page_rows": 3},
}
QUERIES = [np.cos(np.arange(DIMS) * k) + 0.1 * k for k in range(1, 4)]


def deployment(store, root, layout, recipe=None):
    if layout == "E0H":
        index = LocalExactIndex(store, matrix_dir=root / layout)
    else:
        index = IncrementalExactIndex(
            store, matrix_dir=root / layout, layout=layout, recipe=recipe or RECIPES[layout]
        )
    return Storage(store, store, index, CatalogCache(store), ReleaseCoordinator(store))


class Q(Queries):
    def __init__(self, vector):
        self.vector = vector

    async def embed(self, texts, *, purpose, context):
        from rvg.contracts import EmbeddingBatch

        return EmbeddingBatch(tuple(tuple(self.vector) for _ in texts), SPACE, ("f",), ())


def search(storage, query):
    q = TemporalQuestion(text="fact", as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z")

    async def go():
        retriever = TemporalRetriever(storage, Q(query), max_documents=200)
        selected, info = await retriever.search(
            "t", q, CallContext.with_timeout(30, max_embedding_calls=4), top_k=5
        )
        return (
            [d.key for d in selected],
            info["seed_ids"],
            [r["score"] for r in info.get("ranking", [])],
            info["pending"],
        )

    try:
        return asyncio.run(go())
    except ContractError as exc:
        return ("error", exc.code)


def reference(store, query):
    """Eligible documents from the state store, their stored vectors, the canonical
    matrix-vector product, stable order, pending documents by BM25 (the common path)."""
    from rvg.retrieval.search import BM25

    docs = store.documents("t")
    if not docs:
        return ([], [], [], 0)
    vectors = [store.vector("t", FP, TemporalRetriever.input_text(d)) for d in docs]
    if any(v is not None and v.shape != (DIMS,) for v in vectors):
        return ("error", "temporal_cached_vector_invalid")
    ready = [(d, v) for d, v in zip(docs, vectors, strict=True) if v is not None]
    pending = [d for d, v in zip(docs, vectors, strict=True) if v is None]
    seeds, scores = [], []
    if ready:
        m = np.asarray([v for _, v in ready])
        s = canonical_scores(m / np.linalg.norm(m, axis=1, keepdims=True), unit(query))
        order = np.argsort(-s, kind="stable")[:5]
        seeds, scores = [ready[i][0].key for i in order], [float(s[i]) for i in order]
    lexical = (
        [
            k
            for k, _ in BM25(
                [{"id": d.key, "title": d.title, "text": d.text} for d in pending]
            ).search("fact", 5)
        ]
        if pending
        else []
    )
    fallback = [k for k in lexical if k not in set(seeds)]
    scores += [None] * len(fallback)
    seeds = list(dict.fromkeys(seeds + fallback))
    by_key = {d.key: d for d in docs}
    families = {by_key[k].family for k in seeds}
    return ([d.key for d in docs if d.family in families], seeds, scores, len(pending))


class Layouts(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.root = Path(tempfile.mkdtemp())
        self.store = TemporalStore(self.root / "s.sqlite3", dense_memmap_dir=None)
        self.arms = {
            name: deployment(self.store, self.root, name)
            for name in ("E0H", "base_delta", "segments", "paged")
        }
        self.docs = {}
        self.rng = np.random.default_rng(0)

    def _vector(self, k, scale):
        v = np.sin(np.arange(DIMS) + k) * scale
        v[k % DIMS] += 1.5
        return v

    @rule(k=st.integers(0, 14), with_vector=st.booleans(), scale=st.sampled_from([1.0, 0.5]))
    def insert(self, k, with_vector, scale):
        if k in self.docs:
            return
        d = doc(k)
        self.docs[k] = d
        self.store.upsert("t", d)
        if with_vector:
            self.store.put_vector("t", FP, TemporalRetriever.input_text(d), self._vector(k, scale))

    @rule(k=st.integers(0, 14), scale=st.sampled_from([1.0, 2.0, -1.0]))
    def write(self, k, scale):  # a new vector or a rewrite of an existing one
        if k in self.docs:
            self.store.put_vector(
                "t", FP, TemporalRetriever.input_text(self.docs[k]), self._vector(k + 7, scale)
            )

    @rule(k=st.integers(0, 14))
    def invalidate(self, k):  # a stored vector of the wrong dimension: fails only if ranked
        if k in self.docs:
            self.store.put_vector("t", FP, TemporalRetriever.input_text(self.docs[k]), [1.0, 2.0])

    @rule(k=st.integers(0, 14))
    def revoke(self, k):
        if k in self.docs:
            self.store.revoke("t", self.docs[k].id)

    @invariant()
    def every_layout_equals_the_reference(self):
        for query in QUERIES:
            expected = reference(self.store, query)
            for name, storage in self.arms.items():
                assert search(storage, query) == expected, name

    @invariant()
    def open_files_stay_bounded(self):  # 0.1.0 (parent PUB-4): one generation per layout
        assert len(os.listdir("/dev/fd")) - self.open_before <= 64

    def teardown(self):
        for storage in self.arms.values():  # the shared store, then each layout's generations
            storage.close()
        shutil.rmtree(self.root, ignore_errors=True)


def test_random_update_sequences_keep_every_layout_exact():
    """With the garbage collector off (0.1.0, parent PUB-4): each example's teardown closes
    every file it opened, so the count is back where it started after sixty examples."""
    run = Layouts.TestCase
    run.settings = settings(
        max_examples=60,
        stateful_step_count=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    gc.collect()
    gc.disable()
    try:
        Layouts.open_before = before = len(os.listdir("/dev/fd"))
        run().runTest()
        assert len(os.listdir("/dev/fd")) == before
    finally:
        gc.enable()


@pytest.mark.parametrize("layout", ["base_delta", "segments", "paged"])
def test_a_catch_up_writes_only_what_changed_and_another_index_continues_from_it(tmp_path, layout):
    store = TemporalStore(tmp_path / "s.sqlite3", dense_memmap_dir=None)
    rng = np.random.default_rng(4)
    docs = [doc(i) for i in range(40)]
    for d in docs:
        store.upsert("t", d)
        store.put_vector("t", FP, TemporalRetriever.input_text(d), rng.normal(size=DIMS))
    recipe = {
        "base_delta": {"theta": 0.5},
        "segments": {"base_rows": 64, "fanout": 4},
        "paged": {"page_rows": 16},
    }[layout]
    first = deployment(store, tmp_path, layout, recipe)
    query = QUERIES[0]
    assert search(first, query) == reference(store, query)
    built = first.index.stats["rows_written"]
    assert first.index.stats["full_builds"] == 1 and built == 40
    for i in (3, 17):  # two rewrites and one new document with its vector
        store.put_vector("t", FP, TemporalRetriever.input_text(docs[i]), rng.normal(size=DIMS))
    extra = doc(99)
    store.upsert("t", extra)
    store.put_vector("t", FP, TemporalRetriever.input_text(extra), query)
    assert search(first, query) == reference(store, query)
    stats = first.index.stats
    assert stats["catch_ups"] == 1 and stats["full_builds"] == 1
    written = stats["rows_written"] - built
    assert written < 40  # not a full rewrite: E1 delta 3, E2 segment 3, E3 touched pages
    other = deployment(store, tmp_path, layout, recipe)  # another process's index
    assert search(other, query) == reference(store, query)
    assert other.index.stats["adopted"] == 1 and other.index.stats["builds"] == 0
    first.close()
    other.close()


def test_a_fenced_catch_up_publishes_nothing_and_leaves_no_parts(tmp_path):
    store = TemporalStore(tmp_path / "s.sqlite3", dense_memmap_dir=None)
    for i in range(10):
        store.upsert("t", doc(i))
        store.put_vector("t", FP, TemporalRetriever.input_text(doc(i)), np.ones(DIMS) + i)
    arm = deployment(store, tmp_path, "segments", {"base_rows": 4, "fanout": 2})
    search(arm, QUERIES[0])
    store.put_vector("t", FP, TemporalRetriever.input_text(doc(1)), -np.ones(DIMS))

    def crash(point):
        if point == "after_data_written":
            raise RuntimeError("builder died")

    arm.index.fault = crash
    with pytest.raises(RuntimeError):
        search(arm, QUERIES[0])
    directory = next((tmp_path / "segments" / "v3" / "segments").iterdir())  # FIX-MATRIX-GEN
    manifest_files = set(
        __import__("json").loads((directory / "manifest.json").read_text())["files"]
    )
    parts = {p.name for p in directory.glob("*.rvgm")}
    assert parts == manifest_files  # the failed builder's parts were discarded
    arm.index.fault = lambda point: None
    assert search(arm, QUERIES[0]) == reference(store, QUERIES[0])
    arm.close()


@pytest.mark.parametrize("layout", ["E0H", "base_delta", "segments", "paged"])
def test_queries_keep_answering_while_vectors_are_written(tmp_path, layout):
    """A query needs vectors at least as new as when it started: when a newer generation is
    already published or being built, it takes that one instead of chasing its own (N1F-2 pilot:
    `dense_generation_unstable` under a steady writer)."""
    import threading

    store = TemporalStore(tmp_path / "s.sqlite3", dense_memmap_dir=None)
    rng = np.random.default_rng(8)
    docs = [doc(i) for i in range(30)]
    for d in docs:
        store.upsert("t", d)
        store.put_vector("t", FP, TemporalRetriever.input_text(d), rng.normal(size=DIMS))
    arm = deployment(
        store,
        tmp_path,
        layout,
        {
            "base_delta": {"theta": 0.2},
            "segments": {"base_rows": 4, "fanout": 2},
            "paged": {"page_rows": 8},
        }.get(layout),
    )
    stop = threading.Event()

    def write():
        writer = TemporalStore(tmp_path / "s.sqlite3", dense_memmap_dir=None)
        k = 0
        while not stop.is_set():
            d = docs[k % len(docs)]
            writer.put_vector("t", FP, TemporalRetriever.input_text(d), rng.normal(size=DIMS))
            k += 1
        writer.close()

    thread = threading.Thread(target=write)
    thread.start()

    async def many():
        q = TemporalQuestion(
            text="fact", as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
        )

        async def one():
            for _ in range(15):
                await TemporalRetriever(arm, Q(QUERIES[0]), max_documents=200).search(
                    "t", q, CallContext.with_timeout(30, max_embedding_calls=4), top_k=5
                )

        await asyncio.gather(*[one() for _ in range(8)])

    try:
        asyncio.run(many())
    finally:
        stop.set()
        thread.join()
    assert search(arm, QUERIES[0]) == reference(store, QUERIES[0])
    arm.close()
