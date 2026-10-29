"""N1: the exact dense matrix search gives the row-by-row search's results (commit 1060c27,
kept verbatim in tests/row_by_row_reference.py) on every filter, tie, pending document and
store change, and never answers from a stale snapshot. Since N1F-1 the matrix sits behind the
vector index port (`rvg.storage.local_exact`) and only a vector change rebuilds it."""

import asyncio
import random

import numpy as np
import pytest
import row_by_row_reference as bench

from rvg.contracts import EmbeddingBatch, EmbeddingSpace
from rvg.errors import ContractError
from rvg.runtime import CallContext
from rvg.storage.local_exact import canonical_scores, unit
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore

DIMS = 6
SPACE = EmbeddingSpace(
    provider="fixture",
    model="fixture",
    base_url="fixture://",
    dimensions=DIMS,
    normalize=False,
    similarity_metric="cosine",
    encoding_format="float",
    query_prefix="",
    document_prefix="",
    query_input_type=None,
    document_input_type=None,
    version=1,
)


class Queries:
    mode = "fixture"

    def __init__(self, vectors):
        self.vectors = vectors

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        return EmbeddingBatch(tuple(tuple(self.vectors[t]) for t in texts), SPACE, ("f",), ())


def doc(identifier, *, family=None, revision=1, day=1, text=None):
    text = text or f"Fact {identifier} rev {revision}."
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family or identifier,
            "revision": revision,
            "title": f"T {identifier}",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": f"2026-09-{day:02d}T00:00:00Z",
            "observed_at": f"2026-09-{day:02d}T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
        }
    )


def tenant_store(rng, *, tenant="t", n=40, path=":memory:", memmap=None):
    store = TemporalStore(path, dense_memmap_dir=memmap)
    docs = []
    for i in range(n):
        family = f"f{i // 3}" if i % 5 == 0 else None
        docs.append(doc(f"d{i:03d}", family=family, day=1 + i % 20))
        if i % 7 == 0:
            docs.append(doc(f"d{i:03d}", family=family, revision=2, day=21))
    for d in docs:
        store.upsert(tenant, d)
        if rng.random() < 0.85:  # the rest wait for indexing (BM25 path)
            store.base.put_vector(
                tenant, SPACE.fingerprint, TemporalRetriever.input_text(d), rng.normal(size=DIMS)
            )
    return store, docs


def summary(hit):
    return bench.summary(hit)


def search(cls, store, emb, text, *, known="2026-09-30T00:00:00Z", allowed=None, top_k=5):
    retriever = cls(store, emb, allowed_ids=allowed, max_documents=64)
    q = TemporalQuestion(text=text, as_of="2026-09-30T00:00:00Z", known_at=known)
    return asyncio.run(
        retriever.search("t", q, CallContext.with_timeout(30, max_embedding_calls=4), top_k=top_k)
    )


def agree(reference, product):
    """N1F-2: the N1 row-by-row reference multiplies in BLAS's order; the product returns
    canonical scores (a fixed summation order, 0.1.0). Rankings, selections and pending counts
    are equal; scores agree to the last bits."""
    if reference.keys() != product.keys():
        return False
    for key in reference:
        if key != "scores" and reference[key] != product[key]:
            return False
    pairs = list(zip(reference["scores"], product["scores"], strict=False))
    return len(reference["scores"]) == len(product["scores"]) and all(
        (x is None) == (y is None) and (x is None or abs(x - y) <= 1e-12) for x, y in pairs
    )


def canonical(store, emb, text, product):
    """Every dense score of the product equals the canonical computation of that document's
    stored vector, bit for bit."""
    q = unit(emb.vectors[text])  # the canonical query normalisation
    for key, score in zip(product["seed_ids"], product["scores"], strict=False):
        if score is None:
            continue
        d = next(x for x in store.documents("t") if x.key == key)
        v = store.vector("t", SPACE.fingerprint, TemporalRetriever.input_text(d))[None, :]
        row = v / np.linalg.norm(v, axis=1, keepdims=True)  # the canonical row normalisation
        if canonical_scores(row, q)[0] != score:
            return False
    return True


def both(store, emb, text, **kw):
    return summary(search(bench.RowByRowRetriever, store, emb, text, **kw)), summary(
        search(TemporalRetriever, store, emb, text, **kw)
    )


@pytest.mark.parametrize("memmap", [False, True])
def test_matrix_search_equals_row_by_row_on_filters_known_at_and_pending(tmp_path, memmap):
    rng = np.random.default_rng(7)
    store, docs = tenant_store(rng, memmap=tmp_path / "mm" if memmap else None)
    store.revoke("t", "d004")
    ids = sorted({d.id for d in docs})
    vectors = {f"q{i}": rng.normal(size=DIMS).tolist() for i in range(25)}
    emb = Queries(vectors)
    r = random.Random(3)
    for text in vectors:
        for known in ("2026-09-30T00:00:00Z", "2026-09-10T00:00:00Z"):
            for allowed in (None, sorted(r.sample(ids, 20))):
                a, b = both(store, emb, text, known=known, allowed=allowed)
                assert agree(a, b) and canonical(store, emb, text, b)
    if memmap:  # one directory per (tenant, space) holding the published matrix only
        files = [f for f in (tmp_path / "mm").rglob("*") if f.is_file()]
        assert len([f for f in files if f.suffix == ".rvgm"]) == 1
        assert not [f for f in files if f.name.endswith(".tmp")]


def test_exact_ties_keep_store_order():
    store = TemporalStore(":memory:")
    same = np.ones(DIMS)
    for identifier in ("b", "a", "c"):
        d = doc(identifier)
        store.upsert("t", d)
        store.base.put_vector("t", SPACE.fingerprint, TemporalRetriever.input_text(d), same)
    emb = Queries({"q": np.ones(DIMS).tolist()})
    a, b = both(store, emb, "q", top_k=2)
    assert agree(a, b) and b["seed_ids"] == ["a@1", "b@1"]


def test_every_store_change_moves_the_stamp_and_is_seen():
    rng = np.random.default_rng(11)
    store, _ = tenant_store(rng)
    emb = Queries({"q": rng.normal(size=DIMS).tolist()})
    fp = SPACE.fingerprint
    stamp = lambda: (store.generations("t"), store.vector_seq("t", fp))
    stamps = [stamp()]
    assert agree(*both(store, emb, "q"))
    late = doc("zz9", day=29)
    store.upsert("t", late)  # new document, no vector yet: BM25 pending path
    stamps.append(stamp())
    a, b = both(store, emb, "q")
    assert agree(a, b) and b["pending"] >= 1
    store.base.put_vector("t", fp, TemporalRetriever.input_text(late), np.asarray(emb.vectors["q"]))
    stamps.append(stamp())
    a, b = both(store, emb, "q")
    assert agree(a, b) and b["seed_ids"][0] == "zz9@1"  # indexed and now the best match
    store.base.put_vector(
        "t", fp, TemporalRetriever.input_text(late), -np.asarray(emb.vectors["q"])
    )
    stamps.append(stamp())
    a, b = both(store, emb, "q")
    assert agree(a, b) and "zz9@1" not in b["seed_ids"][:1]  # a rewritten vector is not stale
    store.revoke("t", "zz9")
    stamps.append(stamp())
    a, b = both(store, emb, "q")
    assert agree(a, b) and "zz9@1" not in b["seed_ids"]
    assert len(set(stamps)) == len(stamps)


def test_tenants_never_share_a_snapshot():
    rng = np.random.default_rng(5)
    store, _ = tenant_store(rng)
    other = doc("d001", text="Another tenant's secret 999.")
    store.upsert("u", other)
    store.base.put_vector(
        "u", SPACE.fingerprint, TemporalRetriever.input_text(other), np.ones(DIMS)
    )
    emb = Queries({"q": np.ones(DIMS).tolist()})
    selected, _ = search(TemporalRetriever, store, emb, "q")
    assert all("secret" not in d.text for d in selected)


def test_a_malformed_vector_fails_only_when_it_could_be_ranked():
    store = TemporalStore(":memory:")
    good, bad = doc("a"), doc("b")
    for d in (good, bad):
        store.upsert("t", d)
    store.base.put_vector("t", SPACE.fingerprint, TemporalRetriever.input_text(good), np.ones(DIMS))
    store.base.put_vector("t", SPACE.fingerprint, TemporalRetriever.input_text(bad), np.ones(3))
    emb = Queries({"q": np.ones(DIMS).tolist()})
    for cls in (bench.RowByRowRetriever, TemporalRetriever):
        with pytest.raises(ContractError, match="temporal_cached_vector_invalid"):
            search(cls, store, emb, "q")
        assert search(cls, store, emb, "q", allowed=["a"])[1]["seed_ids"] == ["a@1"]


def test_family_overflow_is_the_same_error():
    store = TemporalStore(":memory:")
    for i in range(70):
        d = doc(f"x{i:02d}", family="one")
        store.upsert("t", d)
        store.base.put_vector(
            "t", SPACE.fingerprint, TemporalRetriever.input_text(d), np.ones(DIMS)
        )
    emb = Queries({"q": np.ones(DIMS).tolist()})
    for cls in (bench.RowByRowRetriever, TemporalRetriever):
        with pytest.raises(ContractError, match="temporal_context_overflow"):
            search(cls, store, emb, "q")


def test_nothing_eligible_returns_the_empty_result():
    store = TemporalStore(":memory:")
    d = doc("a", day=20)
    store.upsert("t", d)
    store.base.put_vector("t", SPACE.fingerprint, TemporalRetriever.input_text(d), np.ones(DIMS))
    emb = Queries({"q": np.ones(DIMS).tolist()})
    a, b = both(store, emb, "q", known="2026-09-01T00:00:00Z")
    assert a == b == {"seed_ids": [], "scores": [], "selected": [], "pending": 0, "documents": None}


def test_a_file_store_keeps_its_snapshots_beside_it_and_drops_superseded_ones(tmp_path):
    """N1 arm rule (C): matrices are memory-mapped files next to the store, one directory per
    tenant and space; publishing a new generation removes the older, unpinned file."""
    path = tmp_path / "s.sqlite3"
    store = TemporalStore(path)
    emb = Queries({"q": np.ones(DIMS).tolist()})
    for i in range(3):
        d = doc(f"a{i}")
        store.upsert("t", d)
        store.base.put_vector(
            "t", SPACE.fingerprint, TemporalRetriever.input_text(d), np.ones(DIMS) + i
        )
        search(TemporalRetriever, store, emb, "q")
        files = list((tmp_path / "s.sqlite3.dense").rglob("*.rvgm"))
        assert len(files) == 1  # only the published generation's matrix is kept
    assert TemporalStore(":memory:").index.matrix_dir is None  # in-memory stores stay resident


def test_concurrent_first_queries_share_one_snapshot_build():
    """N1 R3: queries that arrive while a tenant's snapshot is being built wait for that build."""
    rng = np.random.default_rng(2)
    store, _ = tenant_store(rng)
    emb = Queries({"q": rng.normal(size=DIMS).tolist()})

    async def many():
        q = TemporalQuestion(
            text="q", as_of="2026-09-30T00:00:00Z", known_at="2026-09-30T00:00:00Z"
        )
        return await asyncio.gather(
            *[
                TemporalRetriever(store, emb, max_documents=64).search(
                    "t", q, CallContext.with_timeout(30, max_embedding_calls=4), top_k=5
                )
                for _ in range(8)
            ]
        )

    results = asyncio.run(many())
    assert store.index.stats["builds"] == 1 and store.storage.catalogs.builds == 1
    assert len({tuple(r[1]["seed_ids"]) for r in results}) == 1
