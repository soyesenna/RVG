"""Child-process side of the FIX-MATRIX-GEN crash test: open the file store with its default
index (base_delta) and die at a fault point of the build."""

import asyncio
import os

from matrix_worker import Queries

from rvg.runtime import CallContext
from rvg.temporal.models import TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore


def die_at(path, point):
    store = TemporalStore(path)

    def fault(where):
        if where == point:
            os._exit(3)

    store.index.fault = fault
    q = TemporalQuestion(text="q", as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z")
    asyncio.run(
        TemporalRetriever(store, Queries(), max_documents=64).search(
            "t", q, CallContext.with_timeout(60, max_embedding_calls=4), top_k=5
        )
    )
