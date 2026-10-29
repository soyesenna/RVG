"""The pre-N1 row-by-row dense search, a test oracle only: the N1 matrix search and every later
index layout must return what it returns. Copied verbatim from the research repository's
`scripts/benchmark_dense_n1.py` (itself `TemporalRetriever.search` at commit 1060c27); it is not
a product implementation and nothing outside the tests imports it."""

import numpy as np

from rvg.errors import ContractError
from rvg.retrieval.search import BM25
from rvg.temporal.store import TemporalRetriever


class RowByRowRetriever(TemporalRetriever):
    """N1-A: the dense branch of `TemporalRetriever.search` at commit 1060c27, verbatim."""

    async def search(self, tenant, question, context, *, strategy="dense", top_k=4, expand=True):
        docs = self.store.documents(
            tenant, known_at=question.known_at, allowed_ids=self.allowed_ids
        )
        by_id = {d.key: d for d in docs}
        if not docs:
            return [], {"seed_ids": [], "pending": 0, "expanded": expand}

        def lexical(items):
            return (
                [
                    key
                    for key, _ in BM25(
                        [{"id": d.key, "title": d.title, "text": d.text} for d in items]
                    ).search(question.text, top_k)
                ]
                if items
                else []
            )

        pending, ranking = [], []
        if strategy != "dense":
            raise ValueError("row_by_row_reference_is_dense_only")
        space = self.embedding.describe()
        ready, vectors, ranking = [], [], []
        for doc in docs:
            vector = self.store.base.vector(tenant, space.fingerprint, self.input_text(doc))
            if vector is None:
                pending.append(doc)
            else:
                if (
                    vector.shape != (space.dimensions,)
                    or not np.isfinite(vector).all()
                    or np.linalg.norm(vector) == 0
                ):
                    raise ContractError("temporal_cached_vector_invalid")
                ready.append(doc)
                vectors.append(vector)
        seeds = []
        if ready:
            batch = await self.embedding.embed([question.text], purpose="query", context=context)
            if (
                batch.space.fingerprint != space.fingerprint
                or len(batch.vectors) != 1
                or len(batch.vectors[0]) != space.dimensions
            ):
                raise ContractError("temporal_query_space_mismatch")
            q = np.asarray(batch.vectors[0], dtype=float)
            if not np.isfinite(q).all() or np.linalg.norm(q) == 0:
                raise ContractError("temporal_query_vector_invalid")
            m = np.asarray(vectors)
            scores = (m / np.linalg.norm(m, axis=1, keepdims=True)) @ (q / np.linalg.norm(q))
            order = np.argsort(-scores, kind="stable")[:top_k]
            seeds = [ready[i].key for i in order]
            ranking = [
                {"key": ready[i].key, "score": float(scores[i]), "via": "dense"} for i in order
            ]
        fallback = [k for k in lexical(pending) if k not in set(seeds)]
        ranking += [{"key": k, "score": None, "via": "lexical_pending"} for k in fallback]
        seeds = list(dict.fromkeys(seeds + fallback))
        families = {by_id[k].family for k in seeds}
        selected = (
            [d for d in docs if d.family in families] if expand else [by_id[k] for k in seeds]
        )
        if len(selected) > self.max_documents:
            raise ContractError("temporal_context_overflow")
        return selected, {
            "seed_ids": seeds,
            "ranking": ranking,
            "pending": len(pending),
            "expanded": expand,
            "documents": len(selected),
        }


def summary(hit):
    selected, info = hit
    return {
        "seed_ids": info["seed_ids"],
        "scores": [r["score"] for r in info.get("ranking", [])],
        "selected": [d.key for d in selected],
        "pending": info["pending"],
        "documents": info.get("documents"),
    }
