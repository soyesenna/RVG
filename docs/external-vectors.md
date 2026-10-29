# External float16 vector sets

An external vector set lets a deployment rank document vectors that already exist, without copying them into the state store. The vectors are float16 shards (one `.npy` file each), a file of row input hashes and a manifest. A deployment maps these files read-only and ranks them in place. Typical uses:

- A corpus of millions of passages that was embedded once, elsewhere.
- Vectors produced by your own embedding pipeline.

Status: opt-in. `rvg.capabilities()` lists it as `vector_indexes.external_float16` (`TESTED_OFFLINE`).

- Search precision is `EXACT_NATIVE`: an exhaustive search over the stored float16 values.
- With the TurboQuant candidate stage in front, precision is `APPROXIMATE`.
- The default storage configuration and its hash do not change.

This page covers three things:

- the file format (`rvg-external-vectors/1`), so other tools can write a set;
- the builder in `rvg.storage.external`;
- the storage configuration that opens a set.

The code is `src/rvg/storage/external.py`. A complete offline run is `examples/storage/external_float16.py`.

## Writing a set

```python
from rvg.storage.external import write_external_set

manifest = write_external_set(
    "vectors",                         # a new or empty directory
    rows,                              # (title, text, vector) or (input_hash, vector)
    space,                             # the EmbeddingSpace the vectors were computed in
    shard_rows=65536,                  # rows per shard file (the default)
)
```

**Rows.** Each row is one of:

- `(title, text, vector)`: RVG computes the input hash of that title and text, the same way it does for a stored document (see "Input hash" below).
- `(input_hash, vector)`: the hash as 64 lowercase hex characters.
- `(None, vector)`: the row is not the embedding of any RVG input, so no document reaches it.

The order of `rows` is the order of the set. If two rows have the same input, a deployment uses the first.

**Vectors.**

- Each vector has the space's `dimensions`.
- Each is stored as float16, rounded to the nearest value.
- A vector that already holds float16 values is stored exactly.

**Space.** `space` is an `EmbeddingSpace` (`rvg.contracts`), or its description as a dict. It must be exactly the space of the deployment's embedding provider (`provider.describe()`). Otherwise a query fails with `external_vectors_space_mismatch`.

**Shards and memory.**

- A shard holds `shard_rows` rows; the last shard may hold fewer.
- Each shard is assembled in memory before it is written. This needs `shard_rows × dims × 2` bytes: 192 MiB at 1,536 dimensions with the default size.
- Use a multiple of 8,192 (`BLOCK_ROWS`) so that every scan block is full.

**The write.**

- `out_dir` must not exist yet, or must be empty. The builder never writes into an existing set.
- Files are written in one pass, and the manifest is written last.
- The result is read back with `ExternalVectorSet`, the same reader a deployment uses. The function returns the manifest's path.
- The same rows produce the same bytes.

**Refused rows.** A refused row stops the write and removes every file written so far. If the builder created `out_dir`, it removes that too. The errors are `rvg.ContractError` codes, and `detail.row` is the 0-based row index:

| Code | Cause |
|---|---|
| `external_vectors_row_dimensions` | the vector's length is not `dimensions` |
| `external_vectors_row_not_representable` | a value is not finite or overflows float16, or the vector rounds to all zeros |
| `external_vectors_row_input_hash_invalid` | the hash is not 64 lowercase hex characters (or None) |
| `external_vectors_row_input_not_text` | the title or text is not a string |
| `external_vectors_row_shape` | the row has neither 2 nor 3 items |
| `external_vectors_empty` | `rows` yielded no row |

A `shard_rows` below 1, or a space without a positive integer `dimensions`, raises `ValueError` before anything is written. A non-empty `out_dir` raises `FileExistsError`.

### From a deployment's state store

```python
from rvg.storage.external import export_external_set

export_external_set(storage.state, "my-tenant", space, "vectors")
```

This writes every vector the state store holds for one tenant in one space, in input-hash order. A row is found by input hash, not by tenant, so the set serves every tenant of the deployment that opens it.

## Opening a set

In `storage.json`:

```json
{
  "schema_version": 1,
  "state": {"backend": "sqlite", "path": "state.sqlite3"},
  "vectors": {
    "source": "external_float16",
    "external": {"manifest": "vectors/manifest.json"}
  }
}
```

- A relative `manifest` path is relative to the deployment root (`open_storage(..., root=...)`, or `open_session(..., root=...)`).
- The index must be `local_exact`, which is the default.

Options under `vectors.external`:

| Option | Meaning |
|---|---|
| `score_dtype` | `float64` (default) or `float32`. This is the dtype of the first scan. Returned scores are float64 canonical scores either way. |
| `threads` | scan threads, 1 to 64 (default 1). Rankings and scores do not depend on it. |
| `query_projection` | `{"method": "truncate_l2", "query_dimensions": N}`. Queries of a space that differs from the set's only in `dimensions` (N larger) are cut to the set's first coordinates and normalised again. The release check then gets no vector from the set, so a change to the knowledge base during a question leaves its scope unknown. |

`vectors.candidate_stage: "turboquant_mse4"` puts the TurboQuant candidate stage in front. You build its codes explicitly. Precision is then `APPROXIMATE`.

**What a deployment does with the set.**

- A document's catalog row reaches a set row only through its input hash.
- Documents the set has are not embedded again (`ingest` and `index` find their vectors in the set).
- Documents the set lacks are embedded into the state store as usual. Their vectors are ranked in the same ranking, from the resident matrix of the local exact index.
- The set's files are never written.
- A row whose norm is zero or not finite is marked invalid and never ranked.

**Updating a set.** The files are memory-mapped. Do not change them while a deployment has them open. To update, write a new set into a new directory and point `storage.json` at its manifest.

**Checking the files.** `ExternalVectorSet(manifest).verify()` checks every shard and the hash file against the manifest's sha256. It reads every byte. Opening a set checks only shapes, dtypes and the space, not the sha256.

## Format `rvg-external-vectors/1`

A set is a directory:

```
vectors/
  manifest.json
  shard-00000.npy
  shard-00001.npy
  ...
  hashes.bin
```

The file names are free: the manifest names the files. Paths in the manifest are relative to the manifest's directory, or absolute.

### manifest.json

| Field | Value |
|---|---|
| `format` | `"rvg-external-vectors/1"` |
| `dtype` | `"float16"` |
| `dims` | number of dimensions; equals `space.dimensions` |
| `input_spec` | `"title-newline-text/1"` (see Input hash) |
| `space` | the embedding space description: the 12 fields of `rvg.contracts.EmbeddingSpace` (`provider`, `model`, `base_url`, `dimensions`, `normalize`, `similarity_metric`, `encoding_format`, `query_prefix`, `document_prefix`, `query_input_type`, `document_input_type`, `version`) |
| `space_fingerprint` | the space fingerprint (below) |
| `shards` | list of `{"file", "rows", "sha256"}` in row order; global row numbers run through the shards in this order |
| `hashes` | `{"file", "sha256"}` of the row hash file |

`sha256` values are lowercase hex digests of the whole file.

### Shards

Each shard is a NumPy `.npy` file. It holds one two-dimensional array of shape `(rows, dims)`, dtype little-endian float16 (`<f2`), in C order. Any `.npy` format version that `numpy.load` reads with `mmap_mode="r"` works. `numpy.save` writes such a file.

### hashes.bin

`rows × 32` bytes, where `rows` is the total over all shards. For each global row it holds the 32-byte input hash of the RVG input that row is the embedding of. A row of 32 zero bytes is the embedding of no RVG input (for example, a passage embedded with an empty title or a truncated text), and no document reaches it.

### Input hash (`title-newline-text/1`)

A document's embedding input is its title, a newline, and its text: `title + "\n" + text`, for the whole revision. The input hash is the SHA-256 of the input's JSON encoding: a JSON string with the quotes, non-ASCII characters kept as UTF-8, and no extra spaces.

```python
import hashlib, json

def input_hash(title, text):
    value = title + "\n" + text
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

# check value:
# input_hash("Budget", "The budget of project A is 500. It ends soon.")
#   == "7a2d71ec5a557fcaff22e102aa50653c7b5ef88f3840307d9c76bfbd73f801d2"
```

This is `rvg.storage.identity.input_hash(embedding_input(document))`. The vector is the provider's embedding of that input. The space's `document_prefix` and `document_input_type` describe what the provider was asked; the input hash does not include them.

### Space fingerprint

The SHA-256 of the space description as JSON: keys sorted, non-ASCII kept, and Python's default separators (`", "` and `": "`).

```python
hashlib.sha256(json.dumps(space, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
```

This is `EmbeddingSpace.fingerprint`. A query is ranked over the set only if its space's fingerprint equals `space_fingerprint`. With `query_projection`, the query space may differ from the set's in `dimensions` only.

### Errors when opening a set

| Code | Cause |
|---|---|
| `external_vectors_manifest_unreadable` | the manifest is missing or not JSON |
| `external_vectors_manifest_unsupported` | `format`, `dtype` or `input_spec` is not the value above |
| `external_vectors_space_description_mismatch` | `space` is not a description whose fingerprint is `space_fingerprint` and whose `dimensions` is `dims` |
| `external_vectors_shard_unreadable` | a shard cannot be loaded |
| `external_vectors_shard_mismatch` | a shard is not float16, not `(rows, dims)`, or not C order |
| `external_vectors_hashes_mismatch` / `external_vectors_hashes_unreadable` | the hash file's size is not `rows × 32`, or it cannot be read |
| `external_vectors_sha256_mismatch` | `verify()`: a file's sha256 differs (`detail.files`) |
| `external_vectors_space_mismatch` | a query of another space (or of other dimensions without `query_projection`) |

## Ranking

The scan works on fixed blocks: each shard is cut at multiples of 8,192 rows. For each block:

- the float16 rows are raised exactly to float64 (or float32 with `score_dtype: float32`);
- one matrix-vector product is taken against the query, normalised in float64;
- each product is divided by the row's float64 norm.

Rows that could reach the top k are then rescored in float64 in a fixed summation order. So a ranking and its scores depend on none of the following: the eligibility mask, the other rows, the number of threads, `score_dtype`, or the platform's BLAS.

Eligible rows only are ranked. Ties are broken in catalog order, and every row tied with the k-th score is kept.

## Tests

- `tests/test_external_writer.py` covers the builder:
  - a written set is byte-for-byte the set written by hand;
  - every row form links through the input hash;
  - float16 rounding and exact round trips;
  - deterministic bytes;
  - the ranking equals the resident local exact index over the same vectors, and a deployment over an exported set embeds no document;
  - refused rows leave no file;
  - broken manifests and files are refused.
- `tests/test_external_vectors.py` covers the reader and the index.
- `tests/packaging/test_external_example.py` runs the example.
