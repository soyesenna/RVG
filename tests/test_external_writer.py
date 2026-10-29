"""The public builder of external float16 vector sets (`write_external_set`, `export_external_set`;
docs/external-vectors.md). A written set is the format `ExternalVectorSet` reads — the same files
as a set written by hand — rows link through the RVG input hash, a deployment over the set ranks
as the resident local exact index over the same (float16) vectors, the bytes are a function of the
rows, and a refused row or a broken manifest leaves nothing a deployment could open."""

import asyncio
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_external_vectors import (
    DIMS,
    FP,
    SPACE,
    Provider,
    bits,
    doc,
    question,
    rank,
    ranked,
    reference,
    vectors,
    write_set,
)

from rvg.errors import ContractError
from rvg.runtime import CallContext
from rvg.storage.external import (
    HASHES_FILE,
    MANIFEST_FILE,
    ExternalFloat16Index,
    ExternalVectorSet,
    export_external_set,
    write_external_set,
)
from rvg.storage.factory import external_storage, local_storage
from rvg.storage.identity import embedding_input, input_hash
from rvg.storage.local_exact import BLOCK_ROWS
from rvg.storage.ports import EXACT_NATIVE
from rvg.storage.sqlite import SQLiteState
from rvg.temporal.store import TemporalRetriever

DOC = Path(__file__).resolve().parents[1] / "docs" / "external-vectors.md"


class Catalog:
    """The catalog rows an index links: input hashes, a generation and a length."""

    def __init__(self, texts):
        self.input_hashes, self.generation = tuple(input_hash(t) for t in texts), 1

    def __len__(self):
        return len(self.input_hashes)


def files(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def texts_of(n):
    return [(f"T {i}", f"Fact number {i} about topic {i % 7}.") for i in range(n)]


# --- the format ---------------------------------------------------------------------------------
def test_a_written_set_is_the_set_written_by_hand(tmp_path):
    """Same rows, same shard size: the same shards and hash file, so the same set fingerprint."""
    n, shard = BLOCK_ROWS + 300, 4000
    pairs, matrix = texts_of(n), vectors(BLOCK_ROWS + 300)
    inputs = [None if i in (2, 5000) else f"{t}\n{x}" for i, (t, x) in enumerate(pairs)]
    rows = [(None, matrix[i]) if inputs[i] is None else (*pairs[i], matrix[i]) for i in range(n)]
    manifest = write_external_set(tmp_path / "w", rows, SPACE, shard_rows=shard)
    by_hand = write_set(tmp_path / "h", matrix, inputs, shard_rows=shard)
    assert manifest.name == MANIFEST_FILE
    written, hand = ExternalVectorSet(manifest), ExternalVectorSet(by_hand)
    assert written.fingerprint == hand.fingerprint  # rows and sha256 of every shard and hash file
    assert written.describe() == hand.describe()
    assert [s["rows"] for s in written.manifest["shards"]] == [4000, 4000, n - 8000]
    assert written.verify() == {"files": 4, "ok": True}
    body = json.loads(manifest.read_text())
    assert body["space"] == asdict(SPACE) and body["space_fingerprint"] == FP
    assert (tmp_path / "w" / HASHES_FILE).read_bytes() == (
        tmp_path / "h" / "hashes.bin"
    ).read_bytes()


def test_rows_link_through_the_rvg_input_hash_in_every_row_form(tmp_path):
    matrix = vectors(5)
    doc3 = doc(3)
    rows = [
        ("A", "first", matrix[0]),
        (input_hash("B\nsecond"), matrix[1]),
        (None, matrix[2]),
        (doc3.title, doc3.text, matrix[3]),
        ("A", "first", matrix[4]),  # the same input again: the first row serves it
    ]
    vset = ExternalVectorSet(write_external_set(tmp_path / "s", rows, SPACE))
    found = vset.rows_of(
        [
            input_hash("A\nfirst"),
            input_hash("B\nsecond"),
            input_hash(embedding_input(doc3)),
            input_hash("nothing"),
        ]
    )
    assert found.tolist() == [0, 1, 3, -1]
    assert np.array_equal(vset.gather([1])[0], matrix[1].astype(np.float64))


def test_vectors_are_rounded_to_float16_and_a_float16_set_round_trips_exactly(tmp_path):
    rng = np.random.default_rng(21)
    wide = rng.standard_normal((40, DIMS))
    first = ExternalVectorSet(write_external_set(tmp_path / "a", [(None, v) for v in wide], SPACE))
    stored = first.gather(np.arange(40))
    assert np.array_equal(stored, wide.astype(np.float16).astype(np.float64))
    again = ExternalVectorSet(
        write_external_set(tmp_path / "b", [(None, v) for v in stored], SPACE)
    )
    assert np.array_equal(bits(again.gather(np.arange(40))), bits(stored))


def test_the_bytes_are_a_function_of_the_rows(tmp_path):
    rows = [(*t, v) for t, v in zip(texts_of(300), vectors(300), strict=True)]
    write_external_set(tmp_path / "one", rows, SPACE, shard_rows=128)
    write_external_set(tmp_path / "two", rows, asdict(SPACE), shard_rows=128)  # a dict space
    assert files(tmp_path / "one") == files(tmp_path / "two")


def test_the_documented_hash_rules_are_the_code(tmp_path):
    """docs/external-vectors.md lets another tool write a set: its input hash recipe, check value
    and space fingerprint recipe are RVG's."""
    text = DOC.read_text(encoding="utf-8")
    check = input_hash("Budget\nThe budget of project A is 500. It ends soon.")
    assert f'"{check}"' in text
    for title, body in (("Budget", "It ends soon."), ("Café 예산", "500원\n둘째 줄"), ("", "x")):
        value = title + "\n" + body
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        documented = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        assert documented == input_hash(embedding_input(SimpleNamespace(title=title, text=body)))
    description = asdict(SPACE)
    documented = json.dumps(description, sort_keys=True, ensure_ascii=False).encode()
    assert hashlib.sha256(documented).hexdigest() == FP


# --- a deployment over a written set ------------------------------------------------------------
def test_the_written_set_ranks_as_the_resident_index_over_the_same_vectors(tmp_path):
    """Index level: every query's ranking over the written set is the resident local exact
    index's ranking over the same float16 vectors; scores agree to rounding."""
    n = 2 * BLOCK_ROWS + 50
    pairs, matrix = texts_of(n), vectors(n, seed=31)
    state = SQLiteState(":memory:")
    for (title, text), v in zip(pairs, matrix, strict=True):
        state.put_vector("t", FP, f"{title}\n{text}", v.astype(np.float64))
    manifest = export_external_set(state, "t", SPACE, tmp_path / "set", shard_rows=BLOCK_ROWS)
    external = ExternalFloat16Index(ExternalVectorSet(manifest), SQLiteState(":memory:"))
    resident = local_storage(state).index
    catalog = Catalog([f"{t}\n{x}" for t, x in pairs])
    rng = np.random.default_rng(32)
    for trial in range(8):
        q = rng.standard_normal(DIMS)
        mask = rng.random(n) < (0.9 if trial % 2 else 0.1)
        _, got = rank(external, catalog, mask, q, 10)
        _, want = rank(resident, catalog, mask, q, 10)
        assert np.array_equal(got[0], want[0])
        assert np.allclose(got[1], want[1], rtol=0, atol=1e-12)
        exact = ranked(np.arange(n), reference(matrix, q), mask, 10)
        assert np.array_equal(got[0], exact[0]) and np.array_equal(bits(got[1]), bits(exact[1]))


def test_a_retriever_over_the_exported_set_embeds_no_document_and_ranks_as_before(tmp_path):
    """Deployment level: documents embedded once into a store; the store's vectors exported as a
    set; a fresh store of the same documents beside the set embeds no document and returns the
    resident deployment's ranking for the same query."""
    docs = [doc(i) for i in range(150)]
    matrix = vectors(150, seed=33).astype(np.float64)
    source = SQLiteState(tmp_path / "source.sqlite3")
    for d, v in zip(docs, matrix, strict=True):
        source.upsert("t", d)
        source.put_vector("t", FP, embedding_input(d), v)
    context = CallContext.with_timeout(30, max_embedding_calls=4)
    provider = Provider(matrix[42])
    before = TemporalRetriever(local_storage(source), provider, max_documents=20)
    _, want = asyncio.run(before.search("t", question(), context, top_k=5))
    manifest = export_external_set(source, "t", SPACE, tmp_path / "set")
    fresh = SQLiteState(tmp_path / "fresh.sqlite3")
    for d in docs:
        fresh.upsert("t", d)
    storage = external_storage(fresh, manifest)
    after = TemporalRetriever(storage, provider, max_documents=20)
    provider.calls = 0
    report = asyncio.run(after.index("t", context))
    assert report["pending_at_start"] == 0 and report["indexed"] == 0
    _, got = asyncio.run(after.search("t", question(), context, top_k=5))
    assert provider.calls == 1  # the query only
    assert [r["key"] for r in got["ranking"]] == [r["key"] for r in want["ranking"]]
    assert np.allclose(
        [r["score"] for r in got["ranking"]], [r["score"] for r in want["ranking"]], atol=1e-12
    )
    assert storage.describe()["precision"] == EXACT_NATIVE
    source.close()
    fresh.close()


def test_export_writes_the_store_vectors_of_one_space_in_input_hash_order(tmp_path):
    state = SQLiteState(":memory:")
    texts = [f"x {i}" for i in range(30)]
    matrix = vectors(30, seed=34).astype(np.float64)
    for text, v in zip(texts, matrix, strict=True):
        state.put_vector("t", FP, text, v)
    other = replace(SPACE, model="other")
    state.put_vector("t", other.fingerprint, "elsewhere", np.ones(DIMS))
    state.put_vector("u", FP, "another tenant", np.ones(DIMS))
    vset = ExternalVectorSet(export_external_set(state, "t", SPACE, tmp_path / "s"))
    order = sorted(range(30), key=lambda i: input_hash(texts[i]))
    assert vset.rows == 30
    assert vset.rows_of([input_hash(texts[i]) for i in order]).tolist() == list(range(30))
    assert np.array_equal(vset.gather(np.arange(30)), matrix[order].astype(np.float16))


# --- refusals -----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("row", "code"),
    [
        (("T", "x", np.ones(DIMS + 1)), "external_vectors_row_dimensions"),
        (("T", "x", np.full(DIMS, np.nan)), "external_vectors_row_not_representable"),
        (("T", "x", np.full(DIMS, 1e6)), "external_vectors_row_not_representable"),  # overflow
        (("T", "x", np.full(DIMS, 1e-9)), "external_vectors_row_not_representable"),  # zero
        (("T", "x", np.zeros(DIMS)), "external_vectors_row_not_representable"),
        (("ABC" * 21 + "D", np.ones(DIMS)), "external_vectors_row_input_hash_invalid"),
        ((b"\0" * 32, np.ones(DIMS)), "external_vectors_row_input_hash_invalid"),
        (("T", 3, np.ones(DIMS)), "external_vectors_row_input_not_text"),
        ((np.ones(DIMS),), "external_vectors_row_shape"),
    ],
)
def test_a_refused_row_leaves_no_file(tmp_path, row, code):
    good = [("T", f"ok {i}", np.ones(DIMS)) for i in range(5)]
    with pytest.raises(ContractError, match=code) as raised:
        write_external_set(tmp_path / "s", [*good, row], SPACE, shard_rows=2)
    assert raised.value.detail.get("row") == 5
    assert not (tmp_path / "s").exists()
    (tmp_path / "kept").mkdir()  # a directory that existed before stays, empty
    with pytest.raises(ContractError, match=code):
        write_external_set(tmp_path / "kept", [*good, row], SPACE, shard_rows=2)
    assert list((tmp_path / "kept").iterdir()) == []


def test_bad_arguments_are_refused_before_anything_is_written(tmp_path):
    rows = [("T", "x", np.ones(DIMS))]
    with pytest.raises(ContractError, match="external_vectors_empty"):
        write_external_set(tmp_path / "empty", [], SPACE)
    assert not (tmp_path / "empty").exists()
    with pytest.raises(ValueError, match="shard_rows_must_be_positive"):
        write_external_set(tmp_path / "s", rows, SPACE, shard_rows=0)
    with pytest.raises(ValueError, match="external_vectors_space_dimensions_required"):
        write_external_set(tmp_path / "s", rows, {"model": "m"})
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "other.txt").write_text("not ours")
    with pytest.raises(FileExistsError):
        write_external_set(tmp_path / "busy", rows, SPACE)
    assert [p.name for p in (tmp_path / "busy").iterdir()] == ["other.txt"]
    assert not (tmp_path / "s").exists()


def test_a_broken_written_manifest_or_file_is_refused(tmp_path):
    rows = [(*t, v) for t, v in zip(texts_of(50), vectors(50), strict=True)]
    manifest = write_external_set(tmp_path / "s", rows, SPACE, shard_rows=20)
    body = json.loads(manifest.read_text())
    for change, code in (
        ({"format": "rvg-external-vectors/0"}, "external_vectors_manifest_unsupported"),
        ({"input_spec": "text-only/1"}, "external_vectors_manifest_unsupported"),
        ({"dims": DIMS * 2}, "external_vectors_space_description_mismatch"),
        (
            {"space": body["space"] | {"model": "other"}},
            "external_vectors_space_description_mismatch",
        ),
        ({"shards": body["shards"][:1]}, "external_vectors_hashes_mismatch"),
    ):
        manifest.write_text(json.dumps(body | change))
        with pytest.raises(ContractError, match=code):
            ExternalVectorSet(manifest)
    manifest.write_text("{not json")
    with pytest.raises(ContractError, match="external_vectors_manifest_unreadable"):
        ExternalVectorSet(manifest)
    manifest.write_text(json.dumps(body))
    shard = tmp_path / "s" / "shard-00001.npy"
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1  # one flipped bit in a vector
    shard.write_bytes(bytes(data))
    with pytest.raises(ContractError, match="external_vectors_sha256_mismatch") as raised:
        ExternalVectorSet(manifest).verify()
    assert raised.value.detail == {"files": ["shard-00001.npy"]}
