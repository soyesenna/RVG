"""Time-scoped retrieval over the storage ports: one logical search for every deployment.

The retriever reads the catalog and the masks from the knowledge state, ranks through the
deployment's vector index and falls back to BM25 for eligible documents that have no vector yet
(the pending path). It never sees a connection, a row id or a file: which database holds the
documents and which index holds the vectors is the deployment's choice (`rvg.storage`)."""

from __future__ import annotations

import asyncio

import numpy as np

from ..errors import ContractError
from ..retrieval.search import BM25
from ..storage.factory import as_storage
from ..storage.identity import embedding_input, input_hash
from ..storage.segments import aligned, catalog_entries, merged

INDEX_BATCH = 100  # documents per embedding call when indexing a tenant


class TemporalRetriever:
    def __init__(
        self,
        store,
        embedding=None,
        *,
        allowed_ids=None,
        max_documents=24,
        principal=None,
        excluded_ids=None,
    ):
        """`store` is a deployment's storage (or a store that carries one). `principal` is a
        `TrustedPrincipal` from the server context (N3): documents it may not see are never
        retrieved. `allowed_ids` only narrows what remains; `excluded_ids` (N1F-6, a deny list)
        removes every revision of those ids from retrieval — on a knowledge base that does not
        change during a question, `excluded_ids=H` retrieves exactly what `allowed_ids=all - H`
        does."""
        self.storage = as_storage(store)
        self.store = self.storage.state
        self.embedding = embedding
        self.allowed_ids = None if allowed_ids is None else frozenset(allowed_ids)
        self.excluded_ids = frozenset(excluded_ids) if excluded_ids else None
        self.max_documents = max_documents
        self.principal = principal
        self._query_vectors = {}  # question text -> vector, reused within one request

    @staticmethod
    def input_text(doc):
        return embedding_input(doc)

    async def index(self, tenant, context):
        if self.embedding is None:
            raise ContractError("embedding_provider_required")
        space = self.embedding.describe()
        blobs = self.storage.blobs
        docs = self.store.documents(tenant, allowed_ids=self.allowed_ids)
        inputs, blocked = [], 0  # (embedding input, revision) per segment (K8-1)
        for d in docs:
            try:
                inputs += [(text, d) for *_, text in catalog_entries(d)]
            except ContractError:
                blocked += 1  # listed by the catalog; never embedded, never retrieved
        have = blobs.has_vectors(tenant, space.fingerprint, {input_hash(t) for t, _ in inputs})
        pending = {t: d for t, d in inputs if input_hash(t) not in have}
        saved = 0
        items = list(pending)
        # The authorised set is read once: re-reading every document per batch made indexing a
        # large tenant quadratic (U18, 12,000 documents).
        authorized = {d.key for d in docs}
        for offset in range(0, len(items), INDEX_BATCH):
            texts = items[offset : offset + INDEX_BATCH]
            batch = await self.embedding.embed(texts, purpose="document", context=context)
            if (
                batch.space.fingerprint != space.fingerprint
                or len(batch.vectors) != len(texts)
                or any(len(v) != space.dimensions for v in batch.vectors)
            ):
                raise ContractError("temporal_embedding_contract")
            for text, vector in zip(texts, batch.vectors, strict=True):
                if pending[text].key in authorized:
                    blobs.put_vector(tenant, space.fingerprint, text, vector)
                    saved += 1
        return {
            "indexed": saved,
            "pending_at_start": len(items),
            "documents": len(docs),
            "space": space.fingerprint,
        } | ({"blocked": blocked} if blocked else {})

    async def search(
        self, tenant, question, context, *, strategy="dense", top_k=4, expand=True, at=None
    ):
        """Documents of the seed families and the ranking. Dense: the deployment's vector index
        over the eligible documents — revoked, not yet known, hidden and not allowed documents
        are removed before ranking, ties keep canonical key order `(id, revision)`, documents
        without a vector follow the dense seeds through BM25."""
        if strategy == "dense":
            return await self._dense(tenant, question, context, top_k=top_k, expand=expand, at=at)
        if strategy != "lexical":
            raise ValueError("unknown_temporal_strategy")
        hidden = self.store.hidden(tenant, self.principal)
        excluded = self.excluded_ids or ()
        docs = [
            d
            for d in self.store.documents(
                tenant, known_at=question.known_at, allowed_ids=self.allowed_ids, at=at
            )
            if d.id not in hidden and d.id not in excluded
        ]
        by_id = {d.key: d for d in docs}
        if not docs:
            return [], {"seed_ids": [], "pending": 0, "expanded": expand}
        seeds = self._lexical(question, docs, top_k)
        ranking = [{"key": k, "score": None, "via": "lexical"} for k in seeds]
        families = {by_id[k].family for k in seeds}
        selected = (
            [d for d in docs if d.family in families] if expand else [by_id[k] for k in seeds]
        )
        return self._selected(selected, seeds, ranking, 0, expand)

    def validity_documents(self, tenant, docs, *, at=None):
        """RVI-02, the validity plane: revisions outside `docs` that can decide whether `docs`
        are in force — replacement declarations aimed at them and other revisions of the same
        ids — read from the authoritative state whatever their ACL, revocation or the user's
        document selection. Only their metadata is used, by the validity gate's code rules; they
        are never evidence, never shown to a model and never named in a public output."""
        store = self.store
        if not docs or not hasattr(store, "supersession_sources"):
            return []
        have = {d.key for d in docs}
        found = store.supersession_sources(tenant, {d.id for d in docs}, at=at)
        return [d for d in found if d.key not in have]

    @staticmethod
    def _lexical(question, items, top_k):
        """BM25 over rows: a segment row ranks by its own text under its row key."""
        return (
            [
                key
                for key, _ in BM25(
                    [
                        {
                            "id": getattr(d, "row_key", d.key),
                            "title": d.title,
                            "text": getattr(d, "search_text", d.text),
                        }
                        for d in items
                    ]
                ).search(question.text, top_k)
            ]
            if items
            else []
        )

    def _selected(self, selected, seeds, ranking, pending, expand):
        if len(selected) > self.max_documents:
            raise ContractError("temporal_context_overflow")
        return selected, {
            "seed_ids": seeds,
            "ranking": ranking,
            "pending": pending,
            "expanded": expand,
            "documents": len(selected),
        }

    async def _dense(self, tenant, question, context, *, top_k, expand, at=None):
        if self.embedding is None:
            raise ContractError("embedding_provider_required")
        space = self.embedding.describe()
        catalog = await self.storage.catalogs.get(tenant, run=asyncio.to_thread)
        mask = catalog.eligible(
            known_at=question.known_at,
            at=at,
            revoked=self.store.revoked(tenant, at=at),
            hidden=self.store.hidden(tenant, self.principal),
            allowed=self.allowed_ids,
            excluded=self.excluded_ids or (),
        )
        if not mask.any():
            return [], {"seed_ids": [], "pending": 0, "expanded": expand}
        view = await self.storage.index.view(
            tenant, space.fingerprint, space.dimensions, catalog, run=asyncio.to_thread
        )
        try:
            if (mask & view.invalid).any():
                raise ContractError("temporal_cached_vector_invalid")
            pending = catalog.docs(np.flatnonzero(mask & ~view.has_vector))
            seeds, ranking = [], []
            if (mask & view.has_vector).any():
                vector = self._query_vectors.get(question.text)
                if vector is None:
                    batch = await self.embedding.embed(
                        [question.text], purpose="query", context=context
                    )
                    if (
                        batch.space.fingerprint != space.fingerprint
                        or len(batch.vectors) != 1
                        or len(batch.vectors[0]) != space.dimensions
                    ):
                        raise ContractError("temporal_query_space_mismatch")
                    vector = self._query_vectors[question.text] = batch.vectors[0]
                q = np.asarray(vector, dtype=float)
                if not np.isfinite(q).all() or np.linalg.norm(q) == 0:
                    raise ContractError("temporal_query_vector_invalid")
                rows, scores = await asyncio.to_thread(view.rank, mask, q, top_k)
                seeds = [catalog.row_keys[i] for i in rows]
                ranking = [
                    {"key": catalog.row_keys[i], "score": float(s), "via": "dense"}
                    for i, s in zip(rows, scores, strict=True)
                ]
        finally:
            # 0.1.0 (parent PUB-4): an error raised above keeps this frame in its traceback,
            # which can sit in a reference cycle (an asyncio task holds its error) until the
            # garbage collector runs; the generation's pinned file and memory map go now.
            view = None
        fallback = [k for k in self._lexical(question, pending, top_k) if k not in set(seeds)]
        ranking += [{"key": k, "score": None, "via": "lexical_pending"} for k in fallback]
        seeds = list(dict.fromkeys(seeds + fallback))
        seed_rows = [catalog.row_of_key[k] for k in seeds]
        rows = self._expanded(catalog, seed_rows, mask) if expand else seed_rows
        selected = merged(catalog.docs(rows))
        return self._selected(selected, seeds, ranking, len(pending), expand)

    @staticmethod
    def _expanded(catalog, seed_rows, mask):
        """K8 R2: the seed rows and, from every other revision of their families, the eligible
        row in the same place as a seed of that family — a revision that is one row (today's
        data) is taken whole, so this is the seed families' eligible rows as before. Other
        segments of a seed's own revision are not expanded (they rank on their own)."""
        families = {catalog.families[i] for i in seed_rows}
        family = catalog.family_rows(families)
        family = family[mask[family]]
        need = list(dict.fromkeys([*seed_rows, *family.tolist()]))
        doc = dict(zip(need, catalog.docs(need), strict=True))  # N1F-6: one batched read
        seeds_of = {}
        for i in seed_rows:
            seeds_of.setdefault(catalog.families[i], []).append(doc[i])
        seeded = {doc[i].key for i in seed_rows}
        by_revision = {}
        for i in family.tolist():
            by_revision.setdefault(doc[i].key, []).append(i)
        chosen = set(seed_rows)
        for key, rows in by_revision.items():
            if len(catalog.rows_of_revision[key]) == 1:
                chosen.update(rows)
            elif key not in seeded:
                views = [doc[i] for i in rows]
                for seed in seeds_of[catalog.families[rows[0]]]:
                    chosen.add(rows[views.index(aligned(views, seed))])
        return sorted(chosen)
