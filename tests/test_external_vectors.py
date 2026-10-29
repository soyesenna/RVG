"""N1F-5 (parent docs/N1F5_PROTOCOL.md): the opt-in external float16 vector set. Rows are ranked
in place, bit for bit as an independent fixed-order numpy computation (0.1.0: canonical scores);
a row's score never depends on the mask or the threads; catalog rows reach external rows only
through the RVG input hash; the set's files are never written; the default storage configuration
hashes as before; a query of a wider space of the same model is cut and normalised (E9, opt-in)."""

import asyncio
import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from rvg.contracts import EmbeddingBatch, EmbeddingSpace
from rvg.errors import ConfigurationError, ContractError
from rvg.retrieval.data import canonical_hash
from rvg.runtime import CallContext
from rvg.storage import load_config, open_storage
from rvg.storage.external import (
    FORMAT,
    ExternalCandidateIndex,
    ExternalFloat16Index,
    ExternalVectorSet,
)
from rvg.storage.factory import external_storage, resolved_layout
from rvg.storage.identity import EMBEDDING_INPUT, embedding_input, input_hash
from rvg.storage.local_exact import BLOCK_ROWS, scan_bound
from rvg.storage.ports import APPROXIMATE, EXACT_NATIVE
from rvg.storage.sqlite import SQLiteState
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever

DIMS = 64
SPACE = EmbeddingSpace(
    provider="fixture",
    model="fixture",
    base_url="fixture://",
    dimensions=DIMS,
    normalize=True,
    similarity_metric="cosine",
    encoding_format="float",
    query_prefix="",
    document_prefix="",
    query_input_type=None,
    document_input_type=None,
    version=1,
)
FP = SPACE.fingerprint
SHARD_ROWS = BLOCK_ROWS + 1000  # two blocks in a full shard: the second one short


def bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc(i):
    text = f"Fact number {i} about topic {i % 7}."
    return TemporalDocument.model_validate(
        {
            "id": f"d{i:05d}",
            "family": f"d{i:05d}",
            "revision": 1,
            "title": f"T {i}",
            "text": text,
            "source": f"fixture://{i}",
            "published_at": "2026-09-01T00:00:00Z",
            "observed_at": "2026-09-01T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
        }
    )


def vectors(n, dims=DIMS, seed=5):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((9, dims))
    x = centers[rng.integers(0, 9, n)] + 1.1 * rng.standard_normal((n, dims))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def write_set(folder, matrix, inputs, *, space=SPACE, shard_rows=SHARD_ROWS):
    """A manifest over float16 shards; `inputs[r]` is the RVG input row r embeds, or None."""
    folder.mkdir(parents=True, exist_ok=True)
    shards = []
    for i, start in enumerate(range(0, len(matrix), shard_rows)):
        path = folder / f"shard-{i:04d}.npy"
        np.save(path, matrix[start : start + shard_rows])
        shards.append({"file": path.name, "rows": len(matrix[start : start + shard_rows])})
        shards[-1]["sha256"] = sha(path)
    raw = np.zeros((len(matrix), 32), dtype=np.uint8)
    for row, text in enumerate(inputs):
        if text is not None:
            raw[row] = np.frombuffer(bytes.fromhex(input_hash(text)), dtype=np.uint8)
    (folder / "hashes.bin").write_bytes(raw.tobytes())
    manifest = {
        "format": FORMAT,
        "dtype": "float16",
        "dims": space.dimensions,
        "input_spec": EMBEDDING_INPUT,
        "space": asdict(space),
        "space_fingerprint": space.fingerprint,
        "shards": shards,
        "hashes": {"file": "hashes.bin", "sha256": sha(folder / "hashes.bin")},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder / "manifest.json"


def reference(matrix, q):
    """Independent canonical scores of every row, whatever the scan's dtype: the float16 row
    raised exactly to float64, its products rounded one by one and summed by numpy's pairwise
    sum (no BLAS), divided by its norm summed the same way; the query normalised the same way."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.sqrt(np.add.reduce(q * q))
    wide = matrix.astype(np.float32).astype(np.float64)
    return np.add.reduce(wide * q, axis=1) / np.sqrt(np.add.reduce(wide * wide, axis=1))


def ranked(rows, scores, mask, top_k):
    """The reference ranking: eligible linked catalog rows, stable descending order."""
    ready = np.flatnonzero(mask & (rows >= 0))
    s = scores[rows[ready]]
    order = np.argsort(-s, kind="stable")[:top_k]
    return ready[order], s[order]


def catalog_of(texts, generation=1):
    return SimpleNamespace(input_hashes=tuple(input_hash(t) for t in texts), generation=generation)


def fixture(tmp_path, n=SHARD_ROWS + 700, dropped=()):
    texts = [f"T {i}\nFact number {i} about topic {i % 7}." for i in range(n)]
    inputs = [None if i in dropped else t for i, t in enumerate(texts)]
    matrix = vectors(n)
    manifest = write_set(tmp_path / "set", matrix, inputs)
    return texts, matrix, manifest


def rank(index, catalog, mask, query, top_k, space=FP, dims=DIMS):
    view = asyncio.run(index.view("t", space, dims, catalog))
    return view, view.rank(mask, query, top_k)


# --- the exact path -----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_ranking_equals_an_independent_computation_bit_for_bit(tmp_path, dtype):
    texts, matrix, manifest = fixture(tmp_path, dropped={3, 9000})
    index = ExternalFloat16Index(
        ExternalVectorSet(manifest), SQLiteState(":memory:"), score_dtype=dtype
    )
    order = np.random.default_rng(1).permutation(len(texts))  # catalog order is not row order
    catalog = catalog_of([texts[i] for i in order])
    rows = np.where(np.isin(order, [3, 9000]), -1, order)
    rng = np.random.default_rng(2)
    for trial in range(12):
        q = rng.standard_normal(DIMS)
        mask = rng.random(len(texts)) < (0.9 if trial % 2 else 0.05)  # full scan / sparse blocks
        _, got = rank(index, catalog, mask, q, 10)
        want = ranked(rows, reference(matrix, q), mask, 10)
        assert np.array_equal(got[0], want[0])
        assert np.array_equal(bits(got[1]), bits(want[1]))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_rows_score_depends_on_neither_the_mask_nor_the_threads(tmp_path, dtype):
    texts, _, manifest = fixture(tmp_path)
    vset = ExternalVectorSet(manifest)
    one = ExternalFloat16Index(vset, SQLiteState(":memory:"), score_dtype=dtype)
    many = ExternalFloat16Index(vset, SQLiteState(":memory:"), score_dtype=dtype, threads=3)
    catalog = catalog_of(texts)
    q = np.random.default_rng(3).standard_normal(DIMS)
    full = np.ones(len(texts), dtype=bool)
    _, (rows_a, scores_a) = rank(one, catalog, full, q, len(texts))
    _, (rows_b, scores_b) = rank(many, catalog, full, q, len(texts))
    assert np.array_equal(rows_a, rows_b) and np.array_equal(bits(scores_a), bits(scores_b))
    score_of = dict(zip(rows_a.tolist(), bits(scores_a).tolist(), strict=True))
    half = np.zeros(len(texts), dtype=bool)
    half[::2] = True
    _, (rows_c, scores_c) = rank(one, catalog, half, q, len(texts))
    assert set(rows_c.tolist()) == set(np.flatnonzero(half).tolist())
    assert all(
        score_of[r] == s for r, s in zip(rows_c.tolist(), bits(scores_c).tolist(), strict=True)
    )
    _, again = rank(one, catalog, half, q, len(texts))
    assert np.array_equal(bits(again[1]), bits(scores_c))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_ranking_does_not_move_when_the_scan_moves_within_its_bound(
    tmp_path, monkeypatch, dtype
):
    """0.1.0: another platform's BLAS sums the scan in another order. Scan scores moved within
    `scan_bound` of the scan's dtype change neither the ranking nor its scores."""
    texts, matrix, manifest = fixture(tmp_path, n=3000)
    vset = ExternalVectorSet(manifest)
    index = ExternalFloat16Index(vset, SQLiteState(":memory:"), score_dtype=dtype)
    catalog = catalog_of(texts)
    rng = np.random.default_rng(13)
    cases = [(rng.random(3000) < 0.6, rng.standard_normal(DIMS), k) for k in (1, 10, 50, 10)]
    before = [rank(index, catalog, m, q, k)[1] for m, q, k in cases]
    for (m, q, k), (rows, scores) in zip(cases, before, strict=True):
        want = ranked(np.arange(3000), reference(matrix, q), m, k)
        assert np.array_equal(rows, want[0]) and np.array_equal(bits(scores), bits(want[1]))
    real, noise = ExternalFloat16Index.external_scores, np.random.default_rng(14)
    bound = scan_bound(DIMS, np.float32 if dtype == "float32" else np.float64)

    def moved(self, rows, q):
        scores = real(self, rows, q)
        return scores + noise.uniform(-0.9, 0.9, scores.shape) * bound

    monkeypatch.setattr(ExternalFloat16Index, "external_scores", moved)
    for (m, q, k), (rows, scores) in zip(cases, before, strict=True):
        _, got = rank(index, catalog, m, q, k)
        assert np.array_equal(got[0], rows) and np.array_equal(bits(got[1]), bits(scores))


def test_catalog_rows_link_only_through_the_rvg_input_hash(tmp_path):
    texts, _, manifest = fixture(tmp_path, n=300, dropped={5})
    vset = ExternalVectorSet(manifest)
    rows = vset.rows_of([input_hash(texts[7]), input_hash(texts[5]), input_hash("other"), "0" * 64])
    assert rows.tolist() == [7, -1, -1, -1]
    # the same input twice in the set: the first row
    twice = write_set(tmp_path / "twice", vectors(4), ["a", "b", "a", None], shard_rows=2)
    assert ExternalVectorSet(twice).rows_of([input_hash("a")]).tolist() == [0]
    index = ExternalFloat16Index(vset, SQLiteState(":memory:"))
    view, _ = rank(index, catalog_of(texts + ["late"]), np.ones(301, bool), np.ones(DIMS), 3)
    assert not view.has_vector[5] and not view.has_vector[300] and view.has_vector.sum() == 299


def test_state_vectors_of_inputs_the_set_lacks_rank_in_the_same_ranking(tmp_path):
    texts, matrix, manifest = fixture(tmp_path, n=400, dropped={11})
    state = SQLiteState(":memory:")
    late = np.random.default_rng(4).standard_normal(DIMS)
    state.put_vector("t", FP, texts[11], late)
    index = ExternalFloat16Index(ExternalVectorSet(manifest), state)
    view, (rows, scores) = rank(index, catalog_of(texts), np.ones(400, bool), late, 400)
    assert view.has_vector.all() and rows[0] == 11
    assert scores[0] == pytest.approx(1.0, abs=1e-12)
    others = reference(matrix, late)
    assert np.array_equal(bits(scores[1:]), bits(np.sort(np.delete(others, 11))[::-1][:399]))


def test_blobs_answer_from_the_set_and_never_write_it(tmp_path):
    texts, matrix, manifest = fixture(tmp_path, n=50)
    before = {p.name: sha(p) for p in (tmp_path / "set").iterdir()}
    state = SQLiteState(":memory:")
    blobs = ExternalFloat16Index(ExternalVectorSet(manifest), state).blobs
    assert blobs.has_vectors("t", FP, [input_hash(texts[2]), input_hash("new")]) == {
        input_hash(texts[2])
    }
    assert np.array_equal(blobs.vector("t", FP, texts[2]), matrix[2].astype(np.float64))
    assert blobs.vector("t", "another space", texts[2]) is None
    assert blobs.put_vector("t", FP, "new", np.ones(DIMS))
    assert blobs.has_vectors("t", FP, [input_hash("new")]) == {input_hash("new")}
    assert state.vector_seq("t", FP) == 1
    assert {p.name: sha(p) for p in (tmp_path / "set").iterdir()} == before
    with pytest.raises(ValueError):
        ExternalVectorSet(manifest).shards[0][0, 0] = 1  # mapped read-only


def test_a_malformed_stored_row_is_invalid_not_ranked(tmp_path):
    texts = [f"x{i}" for i in range(20)]
    matrix = vectors(20)
    matrix[4] = 0
    index = ExternalFloat16Index(
        ExternalVectorSet(write_set(tmp_path / "s", matrix, texts)), SQLiteState(":memory:")
    )
    view, (rows, _) = rank(index, catalog_of(texts), np.ones(20, bool), np.ones(DIMS), 20)
    assert view.invalid.tolist() == [i == 4 for i in range(20)] and 4 not in rows.tolist()


def test_another_space_or_a_broken_manifest_is_refused(tmp_path):
    texts, _, manifest = fixture(tmp_path, n=30)
    index = ExternalFloat16Index(ExternalVectorSet(manifest), SQLiteState(":memory:"))
    with pytest.raises(ContractError, match="external_vectors_space_mismatch"):
        rank(index, catalog_of(texts), np.ones(30, bool), np.ones(DIMS), 3, space="other")
    with pytest.raises(ContractError, match="external_vectors_space_mismatch"):
        rank(index, catalog_of(texts), np.ones(30, bool), np.ones(DIMS), 3, dims=DIMS * 2)
    body = json.loads(manifest.read_text())
    for change, code in (
        ({"dtype": "float32"}, "external_vectors_manifest_unsupported"),
        ({"space_fingerprint": "0" * 64}, "external_vectors_space_description_mismatch"),
        ({"shards": [body["shards"][0] | {"rows": 31}]}, "external_vectors_shard_mismatch"),
    ):
        manifest.write_text(json.dumps(body | change))
        with pytest.raises(ContractError, match=code):
            ExternalVectorSet(manifest)
    manifest.write_text(json.dumps(body))
    (tmp_path / "set" / "hashes.bin").write_bytes(b"\0" * 32 * 30)
    with pytest.raises(ContractError, match="external_vectors_sha256_mismatch"):
        ExternalVectorSet(manifest).verify()


# --- the retriever and the deployment -----------------------------------------------------------
class Provider:
    mode = "fixture"

    def __init__(self, vector, space=SPACE):
        self.vector, self.space, self.calls = np.asarray(vector, dtype=float), space, 0

    def describe(self):
        return self.space

    async def embed(self, texts, *, purpose, context):
        self.calls += len(texts)
        return EmbeddingBatch(tuple(tuple(self.vector) for _ in texts), self.space, ("f",), ())


def question():
    return TemporalQuestion(
        text="fact topic", as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
    )


def deployment(tmp_path, n=120, **options):
    state = SQLiteState(tmp_path / "s.sqlite3")
    docs = [doc(i) for i in range(n)]
    for d in docs:
        state.upsert("t", d)
    matrix = vectors(n, seed=8)
    inputs = [None if i == 0 else embedding_input(d) for i, d in enumerate(docs)]
    manifest = write_set(tmp_path / "set", matrix, inputs, space=options.pop("space", SPACE))
    return state, docs, matrix, manifest


def test_the_retriever_indexes_nothing_the_set_has_and_ranks_through_it(tmp_path):
    state, docs, matrix, manifest = deployment(tmp_path)
    storage = external_storage(state, manifest)
    # The unlinked doc gets another row's vector: two rows of one vector tie up to rounding.
    provider = Provider(matrix[7].astype(np.float64))
    retriever = TemporalRetriever(storage, provider, max_documents=50)
    context = CallContext.with_timeout(30, max_embedding_calls=4)
    report = asyncio.run(retriever.index("t", context))
    assert report["pending_at_start"] == 1 and report["indexed"] == 1  # only the unlinked doc
    provider.vector = matrix[42].astype(np.float64)  # the query: row 42's vector
    _, info = asyncio.run(retriever.search("t", question(), context, top_k=5))
    assert info["seed_ids"][0] == docs[42].key and info["pending"] == 0
    assert all(r["via"] == "dense" for r in info["ranking"])
    assert storage.describe()["precision"] == EXACT_NATIVE
    assert storage.describe()["index"] == "external_float16"
    again = TemporalRetriever(storage, provider, max_documents=50)
    _, second = asyncio.run(again.search("t", question(), context, top_k=5))
    assert second["ranking"] == info["ranking"]


def test_open_storage_builds_the_external_deployment_and_defaults_hash_as_before(tmp_path):
    state, *_ = deployment(tmp_path, n=20)
    state.close()
    plain = load_config({"schema_version": 1, "state": {"backend": "sqlite", "path": "s.sqlite3"}})
    body = plain.model_dump(mode="json")
    for key in ("candidate_stage", "source", "external"):
        del body["vectors"][key]
    del body["catalog"]  # N1F-6
    body["vectors"]["layout"] = resolved_layout(plain.vectors)
    assert plain.fingerprint() == canonical_hash(body)  # the fields of N1F-5/6 do not enter it
    config = {
        "schema_version": 1,
        "state": {"backend": "sqlite", "path": "s.sqlite3"},
        "vectors": {
            "source": "external_float16",
            "external": {"manifest": "set/manifest.json", "score_dtype": "float32"},
        },
    }
    storage = open_storage(config, root=tmp_path)
    assert isinstance(storage.index, ExternalFloat16Index)
    assert storage.index.capabilities()["score_dtype"] == "float32"
    assert load_config(config).fingerprint() != plain.fingerprint()
    storage.close()
    staged = open_storage(
        config | {"vectors": config["vectors"] | {"candidate_stage": "turboquant_mse4"}},
        root=tmp_path,
    )
    assert isinstance(staged.index, ExternalCandidateIndex)
    assert staged.describe()["precision"] == APPROXIMATE
    staged.close()
    with pytest.raises(ConfigurationError, match="external_vectors_manifest_required"):
        open_storage(config | {"vectors": {"source": "external_float16"}}, root=tmp_path)
    with pytest.raises(ConfigurationError, match="external_vectors_need_source"):
        open_storage(config | {"vectors": {"external": {"manifest": "x"}}}, root=tmp_path)


# --- query projection (E9) ----------------------------------------------------------------------
def test_a_wider_query_of_the_same_space_is_cut_and_normalised(tmp_path):
    texts, matrix, manifest = fixture(tmp_path, n=500)
    wide = replace(SPACE, dimensions=2 * DIMS)
    projection = {"method": "truncate_l2", "query_dimensions": 2 * DIMS}
    vset = ExternalVectorSet(manifest)
    assert vset.space_with(2 * DIMS) == wide.fingerprint
    index = ExternalFloat16Index(vset, SQLiteState(":memory:"), projection=projection)
    q = np.random.default_rng(6).standard_normal(2 * DIMS)
    _, got = rank(index, catalog_of(texts), np.ones(500, bool), q, 10, wide.fingerprint, 2 * DIMS)
    want = ranked(np.arange(500), reference(matrix, q[:DIMS]), np.ones(500, bool), 10)
    assert np.array_equal(got[0], want[0]) and np.array_equal(bits(got[1]), bits(want[1]))
    other = replace(SPACE, dimensions=2 * DIMS, model="other")
    with pytest.raises(ContractError, match="external_vectors_space_mismatch"):
        rank(index, catalog_of(texts), np.ones(500, bool), q, 10, other.fingerprint, 2 * DIMS)
    plain = ExternalFloat16Index(vset, SQLiteState(":memory:"))
    with pytest.raises(ContractError, match="external_vectors_space_mismatch"):
        rank(plain, catalog_of(texts), np.ones(500, bool), q, 10, wide.fingerprint, 2 * DIMS)
    blobs = index.blobs
    assert blobs.has_vectors("t", wide.fingerprint, [input_hash(texts[1])]) == {
        input_hash(texts[1])
    }
    assert blobs.vector("t", wide.fingerprint, texts[1]) is None  # no cross-dimension watermark
    assert index.capabilities()["query_projection"]["stored_dimensions"] == DIMS
    with pytest.raises(ValueError, match="query_projection_needs_more_dimensions"):
        ExternalFloat16Index(
            vset, SQLiteState(":memory:"), projection=projection | {"query_dimensions": DIMS}
        )


def test_projected_state_vectors_are_cut_the_same_way(tmp_path):
    texts, _, manifest = fixture(tmp_path, n=200, dropped={9})
    wide = replace(SPACE, dimensions=2 * DIMS)
    state = SQLiteState(":memory:")
    late = np.random.default_rng(7).standard_normal(2 * DIMS)
    state.put_vector("t", wide.fingerprint, texts[9], late)
    index = ExternalFloat16Index(
        ExternalVectorSet(manifest),
        state,
        projection={"method": "truncate_l2", "query_dimensions": 2 * DIMS},
    )
    _, (rows, scores) = rank(
        index, catalog_of(texts), np.ones(200, bool), late, 3, wide.fingerprint, 2 * DIMS
    )
    assert rows[0] == 9 and scores[0] == pytest.approx(1.0, abs=1e-12)


# --- the candidate stage ------------------------------------------------------------------------
def test_the_candidate_stage_needs_built_codes_and_agrees_with_the_exact_path(tmp_path):
    texts, matrix, manifest = fixture(tmp_path)
    exact = ExternalFloat16Index(ExternalVectorSet(manifest), SQLiteState(":memory:"))
    staged = ExternalCandidateIndex(exact, ladder=(40, 160))
    catalog = catalog_of(texts)
    with pytest.raises(ContractError, match="quantized_codes_missing"):
        rank(staged, catalog, np.ones(len(texts), bool), np.ones(DIMS), 10)
    report = staged.build(tmp_path / "codes.npy")
    assert report["rows"] == len(texts) and (tmp_path / "codes.npy.json").is_file()
    staged.load(tmp_path / "codes.npy")
    rng = np.random.default_rng(9)
    for _ in range(10):
        q = matrix[rng.integers(len(texts))].astype(np.float64) + 0.05 * rng.standard_normal(DIMS)
        mask = np.ones(len(texts), bool)
        _, got = rank(staged, catalog, mask, q, 10)
        _, want = rank(exact, catalog, mask, q, 10)
        assert np.array_equal(got[0], want[0])
        assert np.allclose(got[1], want[1], rtol=0, atol=1e-12)
    assert staged.stats["queries"] == 10 and staged.capabilities()["search"] == APPROXIMATE


def test_threaded_candidate_scores_have_the_same_bits(tmp_path):
    from rvg.storage.quantized import quantizer

    *_, manifest = fixture(tmp_path, n=3 * BLOCK_ROWS + 77)
    vset = ExternalVectorSet(manifest)
    one = ExternalCandidateIndex(ExternalFloat16Index(vset, SQLiteState(":memory:")))
    many = ExternalCandidateIndex(ExternalFloat16Index(vset, SQLiteState(":memory:"), threads=3))
    one.build()
    quant = quantizer(DIMS)
    prepared = quant.prepare(np.random.default_rng(11).standard_normal(DIMS))
    a = one.scores(quant, one.codes.packed, prepared)
    b = many.scores(quant, one.codes.packed, prepared)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert np.array_equal(
        a.view(np.uint32), quant.scores(one.codes.packed, prepared).view(np.uint32)
    )
