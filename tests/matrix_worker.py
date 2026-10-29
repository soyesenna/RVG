"""Child-process side of the matrix publication tests: open the shared file store, optionally
inject a fault into the matrix builder, run concurrent first queries, report what happened."""

import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np

from rvg.contracts import EmbeddingBatch, EmbeddingSpace
from rvg.errors import ContractError
from rvg.runtime import CallContext
from rvg.storage.local_exact import LocalExactIndex
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore

DIMS = 8
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
QUERY = np.linspace(1.0, 2.0, DIMS)


class Queries:
    mode = "fixture"

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        return EmbeddingBatch(tuple(tuple(QUERY) for _ in texts), SPACE, ("f",), ())


def doc(i, text=None):
    text = text or f"Fact number {i}."
    return TemporalDocument.model_validate(
        {
            "id": f"d{i:03d}",
            "family": f"d{i:03d}",
            "revision": 1,
            "title": f"T {i}",
            "text": text,
            "source": f"fixture://{i}",
            "published_at": "2026-09-01T00:00:00Z",
            "observed_at": "2026-09-01T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
        }
    )


def populate(path, n=40, seed=0):
    store = TemporalStore(path)
    rng = np.random.default_rng(seed)
    for i in range(n):
        d = doc(i)
        store.upsert("t", d)
        store.put_vector(
            "t", SPACE.fingerprint, TemporalRetriever.input_text(d), rng.normal(size=DIMS)
        )
    store.close()


def fault_of(spec):
    if not spec:
        return None

    def fault(point):
        if point != spec["point"]:
            return
        if spec["action"] == "exit":
            os._exit(3)
        if spec["action"] == "sleep":
            time.sleep(spec["seconds"])
        if spec["action"] == "enospc":
            raise OSError(28, "No space left on device")

    return fault


def open_store(path, *, fault=None, lease=120.0):
    store = TemporalStore(path)
    store.storage.index = LocalExactIndex(
        store, matrix_dir=Path(f"{path}.dense"), fault=fault_of(fault), lease_seconds=lease
    )
    return store


async def first_queries(store, concurrency):
    q = TemporalQuestion(text="q", as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z")

    async def one():
        try:
            _, out = await TemporalRetriever(store, Queries(), max_documents=64).search(
                "t", q, CallContext.with_timeout(60, max_embedding_calls=4), top_k=5
            )
            return {"seed_ids": out["seed_ids"], "scores": [r["score"] for r in out["ranking"]]}
        except ContractError as exc:
            return {"error": exc.code}

    return await asyncio.gather(*[one() for _ in range(concurrency)])


def run(path, out, concurrency, barrier=None, fault=None, lease=120.0):
    store = open_store(path, fault=fault, lease=lease)
    if barrier is not None:
        barrier.wait()
    results = asyncio.run(first_queries(store, concurrency))
    Path(out).write_text(
        json.dumps({"pid": os.getpid(), "stats": store.index.stats, "results": results})
    )
