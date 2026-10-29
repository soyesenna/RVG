"""FIX-MATRIX-GEN (docs/rvi/FIX_MATRIX_GEN_PROTOCOL.md in the parent repository): a local exact
index with a matrix directory uses the N1F-2 `base_delta` layout by default ("auto"), so a vector
write appends to a small delta instead of rewriting the whole matrix. The layout keeps its files in
`v3/base_delta/`, apart from the rebuild layout's `v2/`; search results are the same bit for bit;
a builder that dies mid-way leaves the published generation readable."""

import asyncio
import multiprocessing
from pathlib import Path

import numpy as np
from matrix_worker import DIMS, SPACE, Queries, doc

from rvg.runtime import CallContext
from rvg.storage.factory import load_config, open_storage, resolved_layout
from rvg.storage.incremental import IncrementalExactIndex
from rvg.storage.local_exact import LocalExactIndex
from rvg.temporal.models import TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore

FP = SPACE.fingerprint
CTX = multiprocessing.get_context("spawn")
Q = TemporalQuestion(text="q", as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z")


def put(store, tenant, i, vector, text=None):
    d = doc(i, text)
    store.upsert(tenant, d)
    store.put_vector(tenant, FP, TemporalRetriever.input_text(d), vector)


def search(store, tenant="t", top_k=5):
    async def go():
        _, out = await TemporalRetriever(store, Queries(), max_documents=64).search(
            tenant, Q, CallContext.with_timeout(60, max_embedding_calls=4), top_k=top_k
        )
        return out["seed_ids"], [float(r["score"]).hex() for r in out["ranking"]]

    return asyncio.run(go())


def opened(path, make):
    """The file store at `path` searched by `make(store)` (an index over its own connection)."""
    store = TemporalStore(path)
    store.storage.index = make(store)
    return store


# --- the default ------------------------------------------------------------------------------


def test_a_file_store_uses_base_delta_and_a_resident_one_the_rebuild(tmp_path):
    store = TemporalStore(tmp_path / "s.sqlite3")
    assert type(store.index) is IncrementalExactIndex
    assert store.index.layout == "base_delta" and store.index.recipe == {"theta": 0.10}
    assert type(TemporalStore(":memory:").index) is LocalExactIndex
    assert type(TemporalStore(tmp_path / "r.sqlite3", dense_memmap_dir=None).index) is (
        LocalExactIndex
    )
    store.close()


def test_open_storage_resolves_auto_and_keeps_the_rebuild_on_request(tmp_path):
    base = {"schema_version": 1, "state": {"backend": "sqlite", "path": "s.sqlite3"}}
    auto = open_storage(base, root=tmp_path)
    assert type(auto.index) is IncrementalExactIndex and auto.index.layout == "base_delta"
    rebuild = open_storage({**base, "vectors": {"layout": "rebuild"}}, root=tmp_path)
    assert type(rebuild.index) is LocalExactIndex
    resident = open_storage({**base, "vectors": {"matrix_dir": None}}, root=tmp_path)
    assert type(resident.index) is LocalExactIndex
    assert resolved_layout(load_config(base).vectors) == "base_delta"
    staged = load_config({**base, "vectors": {"candidate_stage": "turboquant_mse4"}})
    assert resolved_layout(staged.vectors) == "rebuild"
    remote = load_config(
        {
            "schema_version": 1,
            "state": {"backend": "postgres", "dsn_env": "RVG_DSN"},
            "vectors": {"index": "pgvector", "dsn_env": "RVG_DSN"},
        }
    )
    assert resolved_layout(remote.vectors) == "rebuild"
    for s in (auto, rebuild, resident):
        s.close()


# --- directories and results --------------------------------------------------------------------


def test_the_layouts_keep_apart_and_answer_the_same(tmp_path):
    path = tmp_path / "s.sqlite3"
    rng = np.random.default_rng(7)
    store = TemporalStore(path)
    for i in range(40):
        put(store, "t", i, rng.normal(size=DIMS))
    dense = Path(f"{path}.dense")
    rebuild = opened(path, lambda s: LocalExactIndex(s, matrix_dir=dense))
    for step in range(6):
        for k in range(3):
            put(store, "t", 100 + 3 * step + k, rng.normal(size=DIMS))
        assert search(store) == search(rebuild)
    assert any((dense / "v3" / "base_delta").rglob("*.rvgm"))
    assert any((dense / "v2").rglob("*.rvgm"))
    v3_manifest = next((dense / "v3" / "base_delta").rglob("manifest.json"))
    v2_manifest = next((dense / "v2").rglob("manifest.json"))
    assert '"format": 3' in v3_manifest.read_text() and '"format": 2' in v2_manifest.read_text()
    store.close()
    rebuild.close()


def test_two_tenants_with_the_same_generation_names_keep_their_own_matrices(tmp_path):
    """Before the fix the part cache was keyed by file name only: two tenants with as many
    vector writes publish `base-<seq>-0-t1.rvgm` each, and the second read the first's."""
    path = tmp_path / "s.sqlite3"
    rng = np.random.default_rng(11)
    store = TemporalStore(path)
    for tenant in ("a", "b"):
        for i in range(30):
            put(store, tenant, i, rng.normal(size=DIMS), text=f"{tenant} fact {i}.")
    reference = opened(path, lambda s: LocalExactIndex(s, matrix_dir=None))  # resident
    for tenant in ("a", "b", "a", "b"):
        assert search(store, tenant) == search(reference, tenant)
    names = {
        tenant: sorted(
            p.name
            for p in (Path(f"{path}.dense") / "v3" / "base_delta").rglob("*.rvgm")
            if store.index._directory(tenant, FP).dir in p.parents
        )
        for tenant in ("a", "b")
    }
    assert names["a"] and names["a"] == names["b"]  # the colliding names are really there
    store.close()
    reference.close()


# --- a builder that dies mid-way -----------------------------------------------------------------


def test_a_builder_killed_mid_way_leaves_the_store_readable_and_exact(tmp_path):
    from fix_matrix_gen_worker import die_at

    path = tmp_path / "s.sqlite3"
    rng = np.random.default_rng(13)
    store = TemporalStore(path)
    for i in range(30):
        put(store, "t", i, rng.normal(size=DIMS))
    assert search(store)  # the first generation is published
    store.close()
    for n, point in enumerate(("after_data_written", "after_rename", "after_manifest")):
        writer = TemporalStore(path)
        put(writer, "t", 200 + n, rng.normal(size=DIMS))
        writer.close()
        p = CTX.Process(target=die_at, args=(str(path), point))
        p.start()
        p.join(60)
        assert p.exitcode == 3, point
        again = TemporalStore(path)  # the default index takes the dead builder's lease over
        reference = opened(path, lambda s: LocalExactIndex(s, matrix_dir=None))
        assert search(again) == search(reference), point
        assert not list(Path(f"{path}.dense").rglob("*.tmp"))
        again.close()
        reference.close()
