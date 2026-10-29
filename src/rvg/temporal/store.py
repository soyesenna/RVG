"""`TemporalStore`: the SQLite + local exact deployment under its historical name, so the N0-N3
runners and tests keep working. The product path does not depend on it: the retriever and the
engine use the storage ports (`rvg.storage`), and any deployment `open_storage` builds —
SQLite or PostgreSQL state — runs the same code."""

from __future__ import annotations

from pathlib import Path

from ..storage.factory import local_storage
from ..storage.sqlite import SQLiteState
from .retriever import INDEX_BATCH, TemporalRetriever

__all__ = ["INDEX_BATCH", "TemporalRetriever", "TemporalStore"]


class TemporalStore(SQLiteState):
    """Immutable source events and reusable vectors; no time-unsafe answer cache. Dense search
    uses the local exact index; `dense_memmap_dir` keeps its matrices in files (N1: next to the
    store, `<store>.dense/`; an in-memory store keeps them in memory). `candidate_stage` turns on
    the opt-in TurboQuant candidate stage (N1F-4; its codes are built explicitly)."""

    def __init__(self, path, *, dense_memmap_dir="beside", candidate_stage="none"):
        super().__init__(path)
        if dense_memmap_dir == "beside":
            dense_memmap_dir = None if str(path) == ":memory:" else Path(f"{path}.dense")
        self.storage = local_storage(
            self, matrix_dir=dense_memmap_dir, candidate_stage=candidate_stage
        )

    @property
    def index(self):
        return self.storage.index

    def close(self):
        """The store and the vector index it owns (0.1.0, parent PUB-4: its generations' files
        and memory maps were held until the garbage collector reached the store)."""
        super().close()
        close = getattr(self.storage.index, "close", None)
        if close is not None:
            close()
