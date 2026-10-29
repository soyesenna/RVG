"""The scoped dependency release fence (RVI-03): the release check of every engine since
0.1.0.dev27 (U24), through the policy's `release` block.

`current` withholds an answer whenever the tenant's knowledge number moved while the answer was
made, whatever moved. The scoped fence asks instead, at the release instant, whether any change
since the attempt began could have changed what the attempt read. It reads the change records
the registered writers leave in the authoritative state store — never a scope guessed from
search results alone — and classifies every change as CLEAR, RELATED or UNKNOWN:

- a new revision (every insert path: ingest, revision events, upsert, catch-up): RELATED when its
  id or a document it `supersedes` is in the read-set (whatever its ACL or observation time: the
  validity plane reads those revisions as they are); then CLEAR when it was observed after the
  question's `known_at`; RELATED when its family is in the read-set; CLEAR when the principal
  cannot retrieve it at the fence; otherwise RELATED when one of its rows scores at or above the
  k-th score of one of the attempt's searches (the watermark), CLEAR below it;
- a revocation: RELATED when the document is in the read-set, otherwise CLEAR (removing a row
  that was neither a seed, a family member nor a validity source changes none of them);
- an ACL change (outbox `acl` rows): RELATED when the document is in the read-set; otherwise
  its revisions the principal can now retrieve go through the family and watermark steps;
- a vector write (outbox `vector` rows; only with `release.vectors = scoped`, RVI-03 F2): CLEAR
  in another embedding space or when no catalog row has its input; then, for the revisions whose
  rows have that input, the id, family, retrieval and watermark steps above (the watermark of
  the attempt's dense searches only); and RELATED when a search that read rows without vectors
  would now get another BM25 fallback (rows leaving the pending set move its statistics);
- anything it cannot map is UNKNOWN: an event without a kind, an ACL epoch that moved by more
  than its outbox rows say (a writer that bypasses them, e.g. a tenant import), a changed policy
  hash, an ACL epoch that moved during the check (with `release.acl_moves = after_check`, only a
  row on a document or family the attempt read: the check then reads visibility as of the fence
  and other rows are the next event), a new row without a vector, an index without
  exact scores, the lexical strategy, an attempt that read rows without vectors (their BM25
  ranking depends on every document), or a vector write that arrived while the check read them.
  UNKNOWN falls back to the global rule.

Guarantee, limited to authority · watermark · cutoff: when every change is CLEAR, a fresh run at
the fence's state (knowledge number and ACL epoch the release check read) with the same
principal, question and selection gets the same seeds and documents in every search of the
attempt and gives the validity gate the same documents and validity plane; with
`release.vectors = scoped` the fence's state also holds the vectors up to the outbox mark the
check read. Not covered: group membership (the request's trusted principal), changes outside the
state store, model outputs, a conflicting document below the watermark (a fresh run does not
retrieve it either), vector writes when `release.vectors = ignore` (the global rule ignores them
too).

Nothing here names a document in an output: the receipt carries counts and reasons only."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..errors import ContractError
from ..storage.identity import input_hash
from ..storage.local_exact import canonical_scores
from ..storage.ports import EXACT_CANONICAL, EXACT_NATIVE
from ..storage.segments import catalog_entries
from .config import universal_hash

CLEAR, RELATED, UNKNOWN = "CLEAR", "RELATED", "UNKNOWN"
GUARANTEE = "authority+watermark+cutoff"

# How far below the watermark a row still counts as able to enter a search, by the index's
# precision class: canonical scores are recomputed with the index's own function (ties count),
# a backend's native scores may differ in low bits. Other classes have no exact watermark.
EPSILON = {EXACT_CANONICAL: 1e-9, EXACT_NATIVE: 1e-3}

# The writer registration (docs/rvi/RVI03_LOCAL_PROTOCOL.md §4.2 in the parent repository).
WRITERS = {
    "insert": "document id, family, supersedes targets; retrieval watermark when retrievable",
    "revoke": "document id",
    "acl": "document id; family and retrieval watermark of revisions now retrievable",
    "acl_without_outbox": "unmapped: global (e.g. tenant import writes the epoch directly)",
    "event_without_kind": "unmapped: global",
    "vector": "input hash and space (outbox `vector` rows) with `release.vectors = scoped`; "
    "otherwise none (derived)",
    "group": "outside RVG: the request's trusted principal",
    "policy": "unmapped when the policy hash differs between start and fence: global",
}


@dataclass(frozen=True)
class Start:
    """What an attempt read before its knowledge snapshot, in this order — outbox mark, ACL
    epoch — so that every change committed after them is inside the checked range."""

    outbox_mark: int | None
    acl_epoch: int
    policy: str


def begin(state, tenant):
    mark = state.outbox_mark(tenant) if hasattr(state, "outbox_mark") else None
    return Start(mark, state.acl_epoch(tenant), universal_hash())


@dataclass(frozen=True)
class Read:
    """One search of an attempt: its text and top-k, the documents it returned and the validity
    plane read for them (ids, families), its retrieval record and its query vector."""

    text: str
    k: int
    ids: frozenset
    families: frozenset
    retrieval: dict
    vector: object = None
    keys: tuple = ()  # `id@revision` of the search's documents (RVI-03 F3 revalidation)
    plane: tuple = ()  # `id@revision` of the validity plane read for them

    def watermark(self):
        """The k-th dense score; −inf when fewer than k rows were ranked (any row can enter)."""
        dense = [r["score"] for r in self.retrieval.get("ranking") or [] if r.get("via") == "dense"]
        return min(dense) if len(dense) >= self.k else -np.inf

    def consumed(self):
        """RVI-03 F3 (R-c): what this search gave the attempt, as canonical bytes — text, top-k,
        seeds in order, documents, the BM25 fallback and the pending count, the watermark and the
        validity plane."""
        ranking = self.retrieval.get("ranking") or []
        mark = self.watermark()
        body = {
            "text": self.text,
            "k": self.k,
            "seeds": list(self.retrieval.get("seed_ids") or []),
            "keys": list(self.keys),
            "lexical": [r["key"] for r in ranking if r.get("via") == "lexical_pending"],
            "pending": self.retrieval.get("pending") or 0,
            "watermark": None if not np.isfinite(mark) else float(mark).hex(),
            "plane": list(self.plane),
        }
        return json.dumps(body, sort_keys=True, ensure_ascii=False).encode()


@dataclass(frozen=True)
class VectorWrites:
    """RVI-03 F2 (`release.vectors = scoped`): the vector writes committed since an attempt
    began, read by the engine before the check — the outbox `vector` rows up to `mark`, the
    tenant's catalog, whether a search that read rows without vectors would now get another BM25
    fallback, and whether another vector row arrived while these were read. `rows` None: the
    state store keeps no outbox (nothing can be mapped)."""

    rows: tuple | None
    mark: int | None = None
    catalog: object = None
    fallback_changed: bool = False
    moved: bool = False
    # FIX-V9-CHURN: place the rows after `mark` once the check has read (`vector_moves =
    # after_check`), and the hashes the fallback recomputation asked the blob store about
    after_check: bool = False
    fallback_hashes: frozenset = frozenset()


@dataclass(frozen=True)
class Verdict:
    status: str
    items: tuple  # (change, outcome, reason), no document names
    vector_mark: int | None = None  # the outbox mark the vector step read (F2), a number only
    # RVI-03 F3: the documents the RELATED items touched (for the retry rules; never in the
    # receipt, which names no document)
    related: frozenset = frozenset()

    def receipt(self):
        reasons = Counter(f"{c}:{o}:{r}" for c, o, r in self.items)
        out = {
            "guarantee": GUARANTEE,
            "status": self.status,
            "checked": len(self.items),
            "reasons": dict(sorted(reasons.items())),
        }
        if self.vector_mark is not None:
            out["vector_mark"] = self.vector_mark
        return out


def _strongest(outcomes):
    for status in (RELATED, UNKNOWN):
        for outcome in outcomes:
            if outcome[0] == status:
                return outcome
    return outcomes[0] if outcomes else (CLEAR, "nothing_retrievable")


def check(
    storage,
    tenant,
    *,
    question,
    principal,
    allowed,
    reads,
    start,
    snapshot,
    fence_seq,
    fence_epoch,
    space,
    strategy,
    vectors=None,
    fence_hidden=None,
    excluded=None,
):
    """The scoped verdict for the changes in `(snapshot, fence_seq]` and ACL epochs
    `(start.acl_epoch, fence_epoch]` — the range the release check read — and, when `vectors`
    is given (`release.vectors = scoped`), for the vector writes it holds. `fence_hidden`
    (`release.acl_moves = after_check`, FIX-ACL-WINDOW): the principal's hidden ids read with the
    fence; the check reads visibility only from them, and the ACL rows that arrive while it
    reads are placed by `_acl_after_check` instead of withholding the verdict."""
    state, items, related = storage.state, [], set()
    read_hashes = set()  # FIX-V9-CHURN: every input whose stored vector this check read
    if universal_hash() != start.policy:
        items.append(("policy", UNKNOWN, "policy_changed"))
    ids = frozenset().union(*(r.ids for r in reads)) if reads else frozenset()
    families = frozenset().union(*(r.families for r in reads)) if reads else frozenset()
    index = getattr(storage, "index", None)
    precision = index.capabilities().get("search") if index is not None else None
    epsilon = EPSILON.get(precision)
    # The watermark of the dense searches; a search that read rows without vectors also ranked
    # them by BM25 over the pending set, which only the F2 fallback step re-checks.
    dense_exact = (
        strategy == "dense"
        and epsilon is not None
        and space is not None
        and all(r.vector is not None for r in reads)
    )
    exact = dense_exact and not any(r.retrieval.get("pending") for r in reads)
    hidden = state.hidden(tenant, principal) if fence_hidden is None else fence_hidden
    revoked = state.revoked(tenant, at=fence_seq)

    def retrievable(doc):
        return (
            doc.observed_at <= question.known_at
            and doc.id not in hidden
            and doc.id not in revoked
            and (allowed is None or doc.id in allowed)
            and (not excluded or doc.id not in excluded)  # N1F-6: the retriever's deny list
        )

    def doc_vectors(doc):
        rows = []
        try:
            entries = catalog_entries(doc)
        except ContractError:
            return rows  # an unsupported segment is never embedded nor retrieved
        for *_, text in entries:
            read_hashes.add(input_hash(text))
            value = storage.blobs.vector(tenant, space.fingerprint, text)
            if value is None:
                return None
            value = np.asarray(value, dtype=np.float64)
            if value.ndim != 1 or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
                return None
            rows.append(value)
        return rows

    def placed(doc, exact=exact):
        """Family and retrieval steps for a revision outside the read-set's ids."""
        if doc.family in families:
            return RELATED, "scope:family"
        if not retrievable(doc):
            return CLEAR, "not_retrievable"
        if not exact:
            return UNKNOWN, "no_exact_watermark"
        rows = doc_vectors(doc)
        if rows is None:
            return UNKNOWN, "pending_vector"
        if not rows:
            return CLEAR, "not_retrievable"
        block = np.vstack(rows)
        block = block / np.linalg.norm(block, axis=1, keepdims=True)
        for read in reads:
            q = np.asarray(read.vector, dtype=np.float64)
            q = q / np.linalg.norm(q)
            if float(canonical_scores(block, q).max()) >= read.watermark() - epsilon:
                return RELATED, "watermark"
        return CLEAR, "below_watermark"

    events = [e for e in state.events(tenant, snapshot) if e["seq"] <= fence_seq]
    keys = [
        (e["document_id"], e["revision"])
        for e in events
        if e["kind"] == "insert" and e["document_id"] and e["revision"]
    ]
    bodies = state.bodies(tenant, keys) if keys else {}
    for event in events:
        kind, identifier = event["kind"], event["document_id"]
        if kind == "insert" and identifier:
            doc = bodies.get((identifier, event["revision"]))
            if doc is None:
                items.append(("insert", UNKNOWN, "unregistered_event"))
            elif doc.id in ids:
                items.append(("insert", RELATED, "scope:document"))
                related.add(doc.id)
            elif any(r.document_id in ids for r in doc.supersedes):
                items.append(("insert", RELATED, "scope:supersedes"))
                related.update(r.document_id for r in doc.supersedes if r.document_id in ids)
            elif doc.observed_at > question.known_at:
                # RVI-03 v2: after the id checks, which the validity plane makes binding even
                # beyond the cutoff (run 1 of the local step, case B2).
                items.append(("insert", CLEAR, "cutoff"))
            else:
                outcome = placed(doc)
                items.append(("insert", *outcome))
                if outcome[0] == RELATED:
                    related.add(doc.id)
        elif kind == "revoke" and identifier:
            if identifier in ids:
                items.append(("revoke", RELATED, "scope:document"))
                related.add(identifier)
            elif not exact:
                items.append(("revoke", UNKNOWN, "no_exact_watermark"))
            else:
                items.append(("revoke", CLEAR, "removal_outside_read_set"))
        else:
            items.append(("event", UNKNOWN, "unregistered_event"))

    if fence_epoch != start.acl_epoch:
        rows = None
        if start.outbox_mark is not None and hasattr(state, "outbox"):
            rows = [
                o
                for o in state.outbox(tenant, start.outbox_mark)
                if o["kind"] == "acl"
                and start.acl_epoch < (o["payload"].get("epoch") or 0) <= fence_epoch
            ]
        if rows is None or len(rows) != fence_epoch - start.acl_epoch:
            items.append(("acl", UNKNOWN, "unregistered_acl_writer"))
        else:
            for identifier in sorted({o["payload"]["id"] for o in rows}):
                if identifier in ids:
                    items.append(("acl", RELATED, "scope:document"))
                    related.add(identifier)
                    continue
                now = state.documents(
                    tenant, known_at=question.known_at, allowed_ids={identifier}, at=fence_seq
                )
                outcomes = [placed(d) for d in now]
                related.update(d.id for d, o in zip(now, outcomes, strict=True) if o[0] == RELATED)
                if not exact and not any(o[0] == RELATED for o in outcomes):
                    outcomes.append((UNKNOWN, "no_exact_watermark"))
                items.append(("acl", *_strongest(outcomes)))
    if fence_hidden is not None:
        items += _acl_after_check(
            state,
            tenant,
            start=start,
            fence_seq=fence_seq,
            fence_epoch=fence_epoch,
            ids=ids,
            families=families,
        )
    elif state.acl_epoch(tenant) != fence_epoch:
        items.append(("acl", UNKNOWN, "acl_moved_during_check"))

    if vectors is not None:
        items += _vector_items(
            state,
            tenant,
            vectors,
            ids=ids,
            space=space,
            placed=lambda d: placed(d, dense_exact),
            related=related,
            read=read_hashes,
        )
        if vectors.after_check and vectors.rows is not None:
            items += _after_check(state, tenant, vectors, space, read_hashes)

    statuses = {o for _, o, _ in items}
    status = RELATED if RELATED in statuses else UNKNOWN if UNKNOWN in statuses else CLEAR
    mark = None if vectors is None else vectors.mark
    return Verdict(status, tuple(items), mark, frozenset(related))


def _acl_after_check(state, tenant, *, start, fence_seq, fence_epoch, ids, families):
    """FIX-ACL-WINDOW (parent docs/rvi/FIX_ACL_WINDOW_PROTOCOL.md §2.3): the ACL rows committed
    after the fence's epoch, read once the check has read everything. The check read visibility
    only as of the fence, so none of them changed what it read; a row on a document the attempt
    read, or on a document of a family it read, is still UNKNOWN (the visibility of what the
    answer rests on moved while it was checked); any other row is the next event, ordered after
    the release. Rows that do not account for the epoch (a writer without outbox rows) and a
    store without an outbox are UNKNOWN."""
    now = state.acl_epoch(tenant)
    if now == fence_epoch:
        return []
    if start.outbox_mark is None or not hasattr(state, "outbox"):
        return [("acl", UNKNOWN, "acl_moved_during_check")]
    rows = [
        o
        for o in state.outbox(tenant, start.outbox_mark)
        if o["kind"] == "acl" and fence_epoch < (o["payload"].get("epoch") or 0) <= now
    ]
    if len(rows) != now - fence_epoch:
        return [("acl", UNKNOWN, "unregistered_acl_writer")]
    out = []
    for identifier in sorted({o["payload"]["id"] for o in rows}):
        if identifier in ids:
            out.append(("acl", UNKNOWN, "acl_moved_during_check"))
            continue
        docs = state.documents(tenant, allowed_ids={identifier}, at=fence_seq)
        if any(d.family in families for d in docs):
            out.append(("acl", UNKNOWN, "acl_moved_during_check"))
        else:
            out.append(("acl", CLEAR, "after_check"))
    return out


def _after_check(state, tenant, vectors, space, read):
    """FIX-V9-CHURN (parent docs/rvi/FIX_V9_CHURN_PROTOCOL.md §2.3): the vector rows committed
    after the mark, read once the check has read everything. A row whose input the check read
    (a stored vector, a mapped row, the fallback's pending set) may have given it a value newer
    than the mark: UNKNOWN, as V9. Any other row changes nothing the check read: it is ordered
    after the release (the next event)."""
    read = read | vectors.fallback_hashes
    out = []
    for row in state.outbox(tenant, vectors.mark):
        if row["kind"] != "vector":
            continue
        payload = row.get("payload") or {}
        same_space = space is not None and payload.get("space") == space.fingerprint
        if same_space and payload.get("input_hash") in read:
            out.append(("vector", UNKNOWN, "vector_moved_during_check"))
        else:
            out.append(("vector", CLEAR, "after_check"))
    return out


def _vector_items(state, tenant, vectors, *, ids, space, placed, related=None, read=None):
    """RVI-03 F2 (parent docs/rvi/RVI03_F2_PROTOCOL.md §3, V1-V9): one item per vector write, the
    strongest outcome over the revisions whose catalog rows have its input."""
    if vectors.rows is None:
        return [("vector", UNKNOWN, "no_outbox")]
    items, catalog = [], vectors.catalog
    for row in vectors.rows:
        payload = row.get("payload") or {}
        if space is None:
            items.append(("vector", UNKNOWN, "no_space"))
            continue
        if payload.get("space") != space.fingerprint:
            items.append(("vector", CLEAR, "other_space"))
            continue
        if read is not None:
            read.add(payload.get("input_hash"))
        at = catalog.rows_of_hash.get(payload.get("input_hash")) if catalog is not None else None
        if at is None or not len(at):
            items.append(("vector", CLEAR, "no_row"))
            continue
        keys = sorted({(d.id, d.revision) for d in catalog.docs(at)})
        bodies = state.bodies(tenant, keys)
        outcomes = []
        for key in keys:
            doc = bodies.get(key)
            if doc is None:
                outcomes.append((UNKNOWN, "unregistered_row"))
            elif doc.id in ids:
                outcomes.append((RELATED, "scope:document"))
            else:
                outcomes.append(placed(doc))
            if related is not None and doc is not None and outcomes[-1][0] == RELATED:
                related.add(doc.id)
        items.append(("vector", *_strongest(outcomes)))
    if vectors.fallback_changed:
        items.append(("vector", RELATED, "fallback_changed"))
    if vectors.moved:
        items.append(("vector", UNKNOWN, "vector_moved_during_check"))
    return items
