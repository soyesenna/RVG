"""N1F-4 (parent docs/N1F4_PROTOCOL.md): the opt-in TurboQuant candidate stage in front of the
local exact index. The quantizer is the N1F-3 one (seeded rotation, N(0, 1) Lloyd-Max codebook);
candidates are rescored from the canonical matrix, so the ranking equals the exact path's bit for
bit, or the stage falls back to the exact path itself. Codes exist only when built explicitly;
rows changed after them are scored exactly (overlay)."""

import ast
import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pytest

from rvg.contracts import EmbeddingBatch, EmbeddingSpace
from rvg.errors import ConfigurationError, ContractError
from rvg.runtime import CallContext
from rvg.storage import load_config, local_exact, open_storage
from rvg.storage.factory import local_storage
from rvg.storage.local_exact import LocalExactIndex, rank_rows, scan_bound
from rvg.storage.ports import APPROXIMATE, EXACT_CANONICAL
from rvg.storage.quantized import (
    LADDER,
    QuantizedCandidateIndex,
    TurboQuant,
    build_codes,
    distortion,
    encode_matrix,
    lloyd_max_gaussian,
    rank_quantized,
    rotation,
)
from rvg.storage.sqlite import SQLiteState
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore

ROOT = Path(__file__).resolve().parents[1]
DIMS = 64
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
FP = SPACE.fingerprint
SHORT = (24, 96)  # a short ladder so that a few hundred rows reach the candidate stage


def bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def same(a, b):
    return np.array_equal(a[0], b[0]) and np.array_equal(bits(a[1]), bits(b[1]))


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def clustered(n, dims, seed, spread=1.2):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((12, dims))
    return unit(centers[rng.integers(0, 12, n)] + spread * rng.standard_normal((n, dims))), rng


# --- the quantizer ------------------------------------------------------------------------------
def test_the_codebook_and_its_distortion_are_the_known_gaussian_values():
    c1, b1 = lloyd_max_gaussian(1)
    c2, _ = lloyd_max_gaussian(2)
    c4, b4 = lloyd_max_gaussian(4)
    assert np.allclose(c1, [-0.7979, 0.7979], atol=2e-4)
    assert np.allclose(c2, [-1.5104, -0.4528, 0.4528, 1.5104], atol=2e-4)
    assert distortion(c1, b1) == pytest.approx(1 - 2 / math.pi, abs=1e-9)
    assert distortion(c4, b4) == pytest.approx(0.009501, abs=2e-6)
    q = TurboQuant(1536)
    assert q.margin == pytest.approx(2 * q.distortion + 10 * math.sqrt(q.distortion / 1536))
    assert q.margin == pytest.approx(0.04387, abs=1e-5)
    assert q.width == 768


def test_the_rotation_is_orthogonal_seeded_and_reproducible():
    r = rotation(32)
    assert np.allclose(r.astype(np.float64) @ r.T.astype(np.float64), np.eye(32), atol=1e-5)
    g = np.random.default_rng(20261017).standard_normal((32, 32))
    q, s = np.linalg.qr(g)
    assert np.array_equal(r, (q * np.sign(np.diag(s))[None, :]).astype(np.float32))
    assert TurboQuant(32).fingerprint == TurboQuant(32).fingerprint
    assert TurboQuant(32).fingerprint != TurboQuant(33).fingerprint


@pytest.mark.parametrize("dims", [64, 65])
def test_codes_pack_two_coordinates_per_byte_and_scores_approximate_the_cosine(dims):
    x, rng = clustered(300, dims, 1)
    quant = TurboQuant(dims)
    packed = quant.encode(x)
    assert packed.shape == (300, (dims + 1) // 2) and packed.dtype == np.uint8
    y = math.sqrt(dims) * (x.astype(np.float32) @ quant.rotation)
    want = np.searchsorted(quant.boundaries, y, side="left")
    assert np.array_equal(packed & 15, want[:, 0::2])
    assert np.array_equal((packed >> 4)[:, : dims // 2], want[:, 1::2])
    q = rng.standard_normal(dims)
    exact = x @ (q / np.linalg.norm(q))
    approx = quant.scores(packed, quant.prepare(q))
    assert approx.dtype == np.float32
    assert np.abs(approx - exact).max() < 0.1 and np.corrcoef(approx, exact)[0, 1] > 0.98
    many = quant.scores_many(packed, [q, -q])
    assert np.allclose(many[:, 0], approx, atol=1e-5) and np.allclose(
        many[:, 1], -approx, atol=1e-5
    )
    assert np.array_equal(encode_matrix(quant, x), packed)


# --- ranking ------------------------------------------------------------------------------------
def rank_both(matrix, codes, quant, mask, query, top_k, ladder, code_row=None, stats=None):
    rows = np.arange(len(matrix))
    code_row = rows if code_row is None else code_row

    def count(event, n=1):
        if stats is not None:
            stats[event] = stats.get(event, 0) + n

    exact = rank_rows(rows, matrix, mask, query, top_k)
    port = rank_quantized(
        rows, matrix, code_row, codes, quant, mask, query, top_k, ladder=ladder, count=count
    )
    return exact, port


def test_the_candidate_stage_equals_the_exact_path_bit_for_bit():
    matrix, rng = clustered(3000, DIMS, 2)
    quant = TurboQuant(DIMS)
    codes = encode_matrix(quant, matrix)
    stats = {}
    for i in range(60):
        query = matrix[rng.integers(0, len(matrix))] + 0.3 * rng.standard_normal(DIMS)
        mask = np.ones(len(matrix), dtype=bool)
        if i % 3 == 1:
            mask[rng.choice(len(matrix), 1800, replace=False)] = False  # the gather path
        if i % 3 == 2:
            mask[rng.choice(len(matrix), 30, replace=False)] = False
        for top_k in (1, 5, 20):
            exact, port = rank_both(matrix, codes, quant, mask, query, top_k, SHORT, stats=stats)
            assert same(exact, port)
    assert stats.get("accepted_24", 0) + stats.get("accepted_96", 0) > 0  # the stage did decide


def test_the_stage_still_equals_the_exact_path_when_the_blas_moves_within_its_bound(monkeypatch):
    """0.1.0: another platform's BLAS sums the exact path's scan in another order. Scan scores
    moved within `scan_bound` change neither path: both return canonical scores."""
    matrix, rng = clustered(3000, DIMS, 4)
    quant = TurboQuant(DIMS)
    codes = encode_matrix(quant, matrix)
    cases = []
    for i in range(30):
        query = matrix[rng.integers(0, len(matrix))] + 0.3 * rng.standard_normal(DIMS)
        mask = rng.random(len(matrix)) < (0.9 if i % 2 else 0.3)
        cases.append((mask, query, (1, 5, 20)[i % 3]))
    before = [rank_both(matrix, codes, quant, m, q, k, SHORT) for m, q, k in cases]
    real, noise = local_exact.scan_scores, np.random.default_rng(5)

    def moved(block, q):
        scores = real(block, q)
        return scores + noise.uniform(-0.9, 0.9, scores.shape) * scan_bound(q.size)

    monkeypatch.setattr(local_exact, "scan_scores", moved)
    for (m, q, k), (exact, port) in zip(cases, before, strict=True):
        assert same(exact, port)
        again = rank_both(matrix, codes, quant, m, q, k, SHORT)
        assert same(again[0], exact) and same(again[1], port)


def test_crowded_neighbours_fall_back_to_the_exact_path():
    rng = np.random.default_rng(3)
    target = rng.standard_normal(DIMS)
    near = unit(target + 0.002 * rng.standard_normal((400, DIMS)))  # 400 near-ties
    matrix = np.vstack([near, unit(rng.standard_normal((600, DIMS)))])
    quant = TurboQuant(DIMS)
    stats = {}
    exact, port = rank_both(
        matrix,
        encode_matrix(quant, matrix),
        quant,
        np.ones(1000, bool),
        target,
        20,
        SHORT,
        stats=stats,
    )
    assert same(exact, port) and stats == {"overlay_rows": 0, "exact_fallback": 1}


def test_small_sets_and_large_k_use_the_exact_path():
    matrix, rng = clustered(300, DIMS, 4)
    quant = TurboQuant(DIMS)
    codes = encode_matrix(quant, matrix)
    stats = {}
    q = rng.standard_normal(DIMS)
    assert same(*rank_both(matrix, codes, quant, np.ones(300, bool), q, 20, LADDER, stats=stats))
    assert same(*rank_both(matrix, codes, quant, np.ones(300, bool), q, 100, SHORT, stats=stats))
    empty = np.zeros(300, bool)
    assert same(*rank_both(matrix, codes, quant, empty, q, 5, SHORT, stats=stats))
    assert stats == {"exact_small": 3}


def test_rows_without_a_current_code_are_scored_exactly():
    matrix, rng = clustered(2000, DIMS, 5)
    quant = TurboQuant(DIMS)
    codes = encode_matrix(quant, matrix)
    code_row = np.arange(len(matrix))
    stale = rng.choice(len(matrix), 300, replace=False)
    code_row[stale] = -1
    stats = {}
    for _ in range(20):
        query = matrix[rng.choice(stale)]  # its best rows have no code
        exact, port = rank_both(
            matrix, codes, quant, np.ones(len(matrix), bool), query, 10, SHORT, code_row, stats
        )
        assert same(exact, port) and exact[0][0] in stale
    assert stats["overlay_rows"] == 20 * 300


# --- the index port -----------------------------------------------------------------------------
class Q:
    mode = "fixture"

    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=float)

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        return EmbeddingBatch(tuple(tuple(self.vector) for _ in texts), SPACE, ("f",), ())


def doc(i, text=None):
    text = text or f"Fact number {i} about topic {i % 7}."
    return TemporalDocument.model_validate(
        {
            "id": f"d{i:04d}",
            "family": f"d{i:04d}",
            "revision": 1,
            "title": f"T {i}",
            "text": text,
            "source": f"fixture://{i}",
            "published_at": "2026-09-01T00:00:00Z",
            "observed_at": "2026-09-01T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
        }
    )


def fill(state, n, seed):
    vectors, _ = clustered(n, DIMS, seed)
    docs = [doc(i) for i in range(n)]
    for d, v in zip(docs, vectors, strict=True):
        state.upsert("t", d)
        state.put_vector("t", FP, TemporalRetriever.input_text(d), v)
    return docs, vectors


def search(storage, query, *, allowed=None, top_k=10):
    q = TemporalQuestion(
        text="fact topic", as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
    )

    async def go():
        retriever = TemporalRetriever(storage, Q(query), max_documents=500, allowed_ids=allowed)
        selected, info = await retriever.search(
            "t", q, CallContext.with_timeout(30, max_embedding_calls=4), top_k=top_k
        )
        return [d.key for d in selected], info["seed_ids"], [r["score"] for r in info["ranking"]]

    try:
        return asyncio.run(go())
    except ContractError as exc:
        return ("error", exc.code)


def quantized(state, matrix_dir=None, ladder=SHORT):
    exact = LocalExactIndex(state, matrix_dir=matrix_dir)
    return local_storage(state, index=QuantizedCandidateIndex(exact, ladder=ladder))


def test_a_search_needs_explicitly_built_codes(tmp_path):
    state = SQLiteState(tmp_path / "s.sqlite3")
    _, vectors = fill(state, 600, 6)
    storage = quantized(state, tmp_path / "s.sqlite3.dense")
    assert search(storage, vectors[3]) == ("error", "quantized_codes_missing")
    report = asyncio.run(build_codes(storage, "t", FP, DIMS))
    assert report["rows"] == 600 and report["bytes"] == 600 * DIMS // 2
    path = tmp_path / "s.sqlite3.dense" / "quantized-v1"
    assert [p.name for p in path.rglob("*") if p.is_file()] == ["codes.rvgq"]
    assert search(storage, vectors[3]) == search(local_storage(state), vectors[3])
    with pytest.raises(ContractError, match="candidate_stage_off"):
        asyncio.run(build_codes(local_storage(state), "t", FP, DIMS))


def test_the_public_builder_takes_the_space_or_its_fingerprint(tmp_path):
    """0.1.0: `rvg.build_candidate_codes` (the documented name) reads the dimensions from an
    `EmbeddingSpace`; a fingerprint needs `dims` (a developer passed the space object to the
    internal `build_codes`, which wants the fingerprint string)."""
    import rvg

    state = SQLiteState(tmp_path / "s.sqlite3")
    _, vectors = fill(state, 600, 6)
    storage = quantized(state, tmp_path / "s.sqlite3.dense")
    report = asyncio.run(rvg.build_candidate_codes(storage, "t", SPACE))
    assert report["rows"] == 600 and report["bytes"] == 600 * DIMS // 2
    assert search(storage, vectors[3]) == search(local_storage(state), vectors[3])
    again = asyncio.run(rvg.build_candidate_codes(storage, "t", FP, DIMS))
    assert again["rows"] == 600
    with pytest.raises(ConfigurationError, match="build_candidate_codes"):
        asyncio.run(rvg.build_candidate_codes(storage, "t", FP))


def test_the_port_equals_the_local_exact_index_and_counts_its_steps(tmp_path):
    state = SQLiteState(tmp_path / "s.sqlite3")
    docs, vectors = fill(state, 800, 7)
    storage = quantized(state, tmp_path / "s.sqlite3.dense")
    asyncio.run(build_codes(storage, "t", FP, DIMS))
    local = local_storage(state)
    ids = [d.id for d in docs]
    rng = np.random.default_rng(8)
    for i in range(30):
        query = vectors[rng.integers(0, 800)] + 0.2 * rng.standard_normal(DIMS)
        allowed = None if i % 2 else ids[::3]
        assert search(storage, query, allowed=allowed) == search(local, query, allowed=allowed)
    stats = storage.index.stats
    assert stats["queries"] == 30 and stats["accepted_24"] + stats["accepted_96"] > 0
    capabilities = storage.index.capabilities()
    assert capabilities["search"] == APPROXIMATE and capabilities["index"] == "local_exact"
    assert capabilities["candidate_stage"]["ladder"] == list(SHORT)
    assert storage.describe()["precision"] == APPROXIMATE
    assert local.describe()["precision"] == EXACT_CANONICAL


def test_codes_are_shared_through_the_file_and_rows_changed_after_them_are_overlay(tmp_path):
    state = SQLiteState(tmp_path / "s.sqlite3")
    docs, vectors = fill(state, 600, 9)
    builder = quantized(state, tmp_path / "s.sqlite3.dense")
    asyncio.run(build_codes(builder, "t", FP, DIMS))
    reader = quantized(state, tmp_path / "s.sqlite3.dense")  # another process: reads the file
    assert search(reader, vectors[5]) == search(local_storage(state), vectors[5])
    assert reader.index.stats["codes_loaded"] == 1 and reader.index.stats["codes_built"] == 0
    # after the codes: a vector rewritten (its old code must not decide) and a new document
    state.put_vector("t", FP, TemporalRetriever.input_text(docs[5]), -vectors[5])
    late = doc(900, "A late fact.")
    state.upsert("t", late)
    state.put_vector("t", FP, TemporalRetriever.input_text(late), vectors[5])
    for query in (vectors[5], -vectors[5], vectors[11]):
        assert search(reader, query) == search(local_storage(state), query)
    assert reader.index.stats["overlay_rows"] >= 2 * 3
    asyncio.run(build_codes(builder, "t", FP, DIMS))  # rebuilt: the reader re-reads the file
    before = reader.index.stats["overlay_rows"]
    assert search(reader, vectors[5]) == search(local_storage(state), vectors[5])
    assert reader.index.stats["overlay_rows"] == before and reader.index.stats["codes_loaded"] == 2


def test_codes_of_another_quantizer_or_with_damaged_bytes_are_refused(tmp_path):
    state = SQLiteState(tmp_path / "s.sqlite3")
    _, vectors = fill(state, 500, 10)
    storage = quantized(state, tmp_path / "s.sqlite3.dense")
    asyncio.run(build_codes(storage, "t", FP, DIMS))
    path = next((tmp_path / "s.sqlite3.dense" / "quantized-v1").rglob("codes.rvgq"))
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    assert search(quantized(state, tmp_path / "s.sqlite3.dense"), vectors[0]) == (
        "error",
        "quantized_codes_mismatch",
    )
    raw[-1] ^= 0xFF
    length = int.from_bytes(raw[8:16], "little")
    header = json.loads(raw[16 : 16 + length])
    header["seed"] = 20261018  # same length: only the seed differs
    text = json.dumps(header).encode()
    assert len(text) == length
    raw[16 : 16 + length] = text
    path.write_bytes(bytes(raw))
    assert search(quantized(state, tmp_path / "s.sqlite3.dense"), vectors[0]) == (
        "error",
        "quantized_codes_mismatch",
    )


def test_a_resident_store_keeps_its_codes_in_memory():
    store = TemporalStore(":memory:", candidate_stage="turboquant_mse4")
    _, vectors = fill(store, 450, 11)
    assert isinstance(store.index, QuantizedCandidateIndex) and store.index.ladder == LADDER
    assert search(store, vectors[1]) == ("error", "quantized_codes_missing")
    asyncio.run(build_codes(store, "t", FP, DIMS))
    assert search(store, vectors[1], top_k=20) == search(local_storage(store), vectors[1], top_k=20)
    assert store.index.stats["accepted_400"] + store.index.stats["exact_fallback"] == 1


# --- configuration and the default ---------------------------------------------------------------
def test_the_default_is_off_and_keeps_every_configuration_hash():
    assert type(TemporalStore(":memory:").index) is LocalExactIndex
    sqlite = load_config({"schema_version": 1, "state": {"backend": "sqlite", "path": "s.sqlite3"}})
    remote = load_config(
        {
            "schema_version": 1,
            "state": {"backend": "postgres", "dsn_env": "RVG_DSN"},
            "vectors": {"index": "pgvector", "dsn_env": "RVG_DSN", "precision": "native"},
        }
    )
    # the values before N1F-4 added the field (RVG 25572bf); a local exact index with a matrix
    # directory moved to the base_delta layout in 0.1.0.dev41 (FIX-MATRIX-GEN): its hash changed,
    # and naming the rebuild layout gives the old one back
    assert (
        sqlite.fingerprint() == "7bf616fa6ac1ce21d76968f6d932e8f9405240d261dd6424511c083cd88d85b6"
    )
    rebuild = load_config(
        {
            "schema_version": 1,
            "state": {"backend": "sqlite", "path": "s.sqlite3"},
            "vectors": {"layout": "rebuild"},
        }
    )
    assert (
        rebuild.fingerprint() == "91687431772cfe6b5075aa5170d946369cc1105ceceec3246bd844af1493b943"
    )
    assert (
        remote.fingerprint() == "b5c7b0986fb8eaab8f0f035a5a248a58670c48b266a044e09d76ae251c0172df"
    )
    on = load_config(
        {
            "schema_version": 1,
            "state": {"backend": "sqlite", "path": "s.sqlite3"},
            "vectors": {"candidate_stage": "turboquant_mse4"},
        }
    )
    assert on.fingerprint() != sqlite.fingerprint()


def test_the_stage_is_opened_from_storage_json_and_only_on_the_local_exact_index(tmp_path):
    storage = open_storage(
        {
            "schema_version": 1,
            "state": {"backend": "sqlite", "path": "s.sqlite3"},
            "vectors": {"candidate_stage": "turboquant_mse4"},
        },
        root=tmp_path,
    )
    assert isinstance(storage.index, QuantizedCandidateIndex)
    assert storage.index.exact.matrix_dir == tmp_path / "s.sqlite3.dense"
    storage.close()
    with pytest.raises(ConfigurationError, match="candidate_stage_requires_local_exact"):
        open_storage(
            {
                "schema_version": 1,
                "state": {"backend": "sqlite", "path": "s.sqlite3"},
                "vectors": {
                    "index": "qdrant",
                    "url_env": "Q",
                    "candidate_stage": "turboquant_mse4",
                },
            },
            root=tmp_path,
            environ={"Q": "http://127.0.0.1:1"},
        )
    with pytest.raises(ConfigurationError, match="candidate_stage_requires_local_exact"):
        open_storage(
            {
                "schema_version": 1,
                "state": {"backend": "sqlite", "path": "t.sqlite3"},
                "vectors": {"layout": "segments", "candidate_stage": "turboquant_mse4"},
            },
            root=tmp_path,
        )


def test_the_policy_resource_is_untouched():
    import hashlib

    policy = ROOT / "src/rvg/resources/policies/universal.json"
    # the N1F-4 stage left it as it was (4681b686…); U21 (dev16) changed only its models
    # (caef8ed5…), N6-2 (dev17) only the cascade's evidence window (49b9367f…), U22 (dev18) only
    # the cascade's consistency rule (5b7d666b…), U24 (dev27) only its release block (b85003cd…),
    # U32 (dev42) only the release block's ACL moves and the cascade's scope_top1
    assert hashlib.sha256(policy.read_bytes()).hexdigest().startswith("24777fc8")


def test_the_stage_never_reaches_a_backend():
    tree = ast.parse((ROOT / "src/rvg/storage/quantized.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("conn", "cursor", "base")
        if isinstance(node, ast.ImportFrom) and node.level:
            assert node.module not in ("sqlite", "postgres", "pgvector", "qdrant")
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in ("sqlite3", "psycopg") for a in node.names)
