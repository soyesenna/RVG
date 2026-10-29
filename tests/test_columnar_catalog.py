"""N1F-6 (parent docs/N1F6_PROTOCOL.md): the opt-in columnar catalog answers as the resident one
does — every row, key, hash, document and filter — while keeping only fixed-width columns and
reading documents from the state store when asked; the `excluded` deny list retrieves exactly
what the allow list of every other id does on a knowledge base that does not change; the default
storage configuration hashes as before."""

import asyncio
import random
from datetime import datetime, timedelta

import numpy as np
import pytest

from rvg.contracts import EmbeddingBatch, EmbeddingSpace
from rvg.errors import ContractError
from rvg.retrieval.data import canonical_hash
from rvg.runtime import CallContext
from rvg.storage import load_config, open_storage
from rvg.storage.catalog import (
    LRU_REVISIONS,
    CatalogCache,
    ColumnarCatalog,
    build_columnar,
    coverage,
    vector_mask,
)
from rvg.storage.factory import local_storage, resolved_layout
from rvg.storage.identity import embedding_input
from rvg.storage.segments import catalog_entries
from rvg.storage.sqlite import SQLiteState
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.retriever import TemporalRetriever

TOPICS = ("budget", "staff", "schedule", "risk")
SPACE = EmbeddingSpace(
    provider="fixture",
    model="fixture",
    base_url="fixture://",
    dimensions=len(TOPICS) + 1,
    normalize=False,
    similarity_metric="cosine",
    encoding_format="float",
    query_prefix="",
    document_prefix="",
    query_input_type=None,
    document_input_type=None,
    version=1,
)
T0 = datetime.fromisoformat("2026-09-01T00:00:00+00:00")


def vector_of(text):
    words = text.lower().split()
    return (*(float(words.count(t)) for t in TOPICS), 0.1)


class Topics:
    mode = "fixture"

    def __init__(self, query=None):
        self.query = query

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        vectors = [vector_of(self.query if purpose == "query" and self.query else t) for t in texts]
        return EmbeddingBatch(tuple(vectors), SPACE, ("f",), ())


def doc(identifier, *, revision=1, family=None, text=None, day=1, units=None):
    text = text or f"The {TOPICS[sum(map(ord, identifier)) % 4]} note of {identifier} r{revision}."
    when = (T0 + timedelta(days=day)).isoformat()
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family or identifier,
            "revision": revision,
            "title": f"T {identifier}",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": when,
            "observed_at": when,
            "units": units or [{"id": "u0", "start": 0, "end": len(text)}],
        }
    )


def long_doc(identifier, *, revision=1, family=None, day=1, blocked=False):
    """Over the 8 KB embedding input: several K8-1 segments (or a blocked revision whose one
    unit is too long)."""
    text, units = "", []
    for t, topic in enumerate(TOPICS):
        for k in range(4):
            sentence = f"The {topic} figure {k} of {identifier}. " + " ".join([topic] * 120) + "."
            start = len(text) + (1 if text else 0)
            text += (" " if text else "") + sentence
            units.append({"id": f"u{t}x{k}", "start": start, "end": start + len(sentence)})
    if blocked:
        units = [{"id": "all", "start": 0, "end": len(text)}]
    return doc(identifier, revision=revision, family=family, text=text, day=day, units=units)


def knowledge(path, *, long_ids=False, seed=0):
    """A state store with families, several revisions, K8-1 rows, a blocked revision, unicode
    and (optionally) long ids, revocations, ACL rows and vectors for most rows."""
    rng = random.Random(seed)
    state = SQLiteState(path)
    docs = []
    for n in range(60):
        identifier = f"d{n:03d}" if n % 7 else f"é-{n}"
        if long_ids and n % 5 == 0:
            identifier = "x" * 40 + f"-{n}"
        family = f"fam{n % 9}" if n % 3 == 0 else None
        for revision in range(1, 2 + (n % 3 == 1)):
            docs.append(doc(identifier, revision=revision, family=family, day=rng.randint(0, 40)))
    docs += [long_doc("long-a", day=3), long_doc("long-a", revision=2, day=9)]
    docs += [long_doc("long-b", family="fam1", day=5), long_doc("blocked", blocked=True)]
    rng.shuffle(docs)
    for d in docs:
        state.upsert("t", d)
    for d in docs:
        if rng.random() < 0.8:
            try:
                entries = catalog_entries(d)
            except ContractError:
                continue
            for *_, text in entries:
                state.put_vector("t", SPACE.fingerprint, text, vector_of(text))
    state.revoke("t", "d004")
    state.set_access("t", "d005", "alice", "deny")
    return state, docs


def both(state):
    resident = CatalogCache(state, layout="resident").build("t")
    columnar = CatalogCache(state, layout="columnar").build("t")
    return resident, columnar


def dumped(d):
    return (getattr(d, "row_key", d.key), d.model_dump(mode="json"))


@pytest.mark.parametrize("long_ids", [False, True])
def test_every_row_key_hash_and_document_is_the_same(tmp_path, long_ids):
    state, _ = knowledge(tmp_path / "s.sqlite3", long_ids=long_ids)
    a, b = both(state)
    assert isinstance(b, ColumnarCatalog) and b._names.fixed is not long_ids
    assert len(a) == len(b) and a.generation == b.generation and a.blocked == b.blocked
    assert list(a.ids) == list(b.ids) and list(a.families) == list(b.families)
    assert np.array_equal(a.observed, b.observed) and np.array_equal(a.inserted, b.inserted)
    assert a.input_hashes == tuple(b.input_hashes) and b.input_hashes == a.input_hashes
    assert a.row_keys == tuple(b.row_keys) and b.unit_keys == a.unit_keys
    assert all(b.row_of_key[k] == a.row_of_key[k] for k in a.row_keys)
    assert b.row_of_key.get("nope@1") is None and b.row_of_key.get("d001@9") is None
    for key, rows in a.rows_of_revision.items():
        assert np.array_equal(b.rows_of_revision[key], rows)
    for digest, rows in a.rows_of_hash.items():
        assert np.array_equal(b.rows_of_hash.get(digest), rows)
    assert b.rows_of_hash.get("0" * 64) is None
    assert np.array_equal(a.revision_starts(), b.revision_starts())
    assert [dumped(d) for d in a.documents] == [dumped(d) for d in b.documents]
    rows = [7, 3, 7, len(a) - 1, 0]
    assert [dumped(d) for d in a.docs(rows)] == [dumped(d) for d in b.docs(rows)]
    families = sorted(set(a.families))
    for k in range(20):
        pick = random.Random(k).sample(families, k % 4 + 1) + ["missing"]
        assert np.array_equal(a.family_rows(pick), b.family_rows(pick))


def test_eligible_is_the_same_for_a_thousand_random_conditions(tmp_path):
    state, docs = knowledge(tmp_path / "s.sqlite3", long_ids=True)
    a, b = both(state)
    ids = sorted({d.id for d in docs})
    rng = random.Random(20261023)
    for _ in range(1000):
        known = T0 + timedelta(days=rng.randint(-1, 45))
        at = rng.choice([None, rng.randint(0, len(docs))])
        draw = lambda: (
            set(rng.sample(ids, rng.randint(0, 6)))
            | ({"unknown", "y" * 50} if rng.random() < 0.2 else set())
        )
        kw = {
            "known_at": known,
            "at": at,
            "revoked": draw(),
            "hidden": draw(),
            "excluded": draw(),
            "allowed": None if rng.random() < 0.5 else set(rng.sample(ids, rng.randint(0, 40))),
        }
        assert np.array_equal(a.eligible(**kw), b.eligible(**kw))


def test_documents_are_read_when_asked_and_kept_in_a_bounded_cache(tmp_path):
    state, _ = knowledge(tmp_path / "s.sqlite3")
    _, b = both(state)
    assert b.fetched == 0  # building keeps no document
    first = b.docs([0, 1, 0])
    assert b.fetched == 2 and first[0] is first[2]
    b.docs([0, 1])
    assert b.fetched == 2  # cached
    assert len(b._cache) <= LRU_REVISIONS


def test_a_revision_inserted_after_the_catalog_read_is_not_a_row(tmp_path):
    state, _ = knowledge(tmp_path / "s.sqlite3")
    read = state.catalog_rows("t")
    rows = sorted(read.rows, key=lambda r: (r[0], r[1]))
    state.upsert("t", doc("a-late"))
    state.upsert("t", doc("zz-late"))
    late = build_columnar(state, "t", read.generations.document_seq, rows)
    now = CatalogCache(state, layout="resident").build("t")
    assert len(late) == len(now) - 2
    assert "a-late" not in set(late.ids) and "zz-late" not in set(late.ids)


def test_coverage_and_the_vector_mask(tmp_path):
    state, _ = knowledge(tmp_path / "s.sqlite3")
    a, b = both(state)
    storage = local_storage(state)
    mask = vector_mask(storage, "t", SPACE.fingerprint, b)
    assert np.array_equal(mask, vector_mask(storage, "t", SPACE.fingerprint, a))
    have = state.has_vectors("t", SPACE.fingerprint, set(a.input_hashes))
    assert mask.tolist() == [h in have for h in a.input_hashes]
    got = coverage(b, mask)
    assert got == coverage(a, mask)
    assert got["rows"] == len(b) and got["rows_without_vector"] == int((~mask).sum())
    assert got["blocked"] == 1 and got["revisions"] == len(b.revision_starts())
    revoked = state.revoked("t")
    fewer = coverage(b, mask, revoked=revoked)
    assert fewer == coverage(a, mask, revoked=revoked) and fewer["revisions"] < got["revisions"]


# --- the deny list ------------------------------------------------------------------------------
def search(storage, query, **kw):
    question = TemporalQuestion(
        text=query, as_of="2026-12-01T00:00:00Z", known_at="2026-12-01T00:00:00Z"
    )
    retriever = TemporalRetriever(storage, Topics(query), max_documents=500, **kw)
    context = CallContext.with_timeout(30, max_embedding_calls=4)

    async def go(strategy):
        selected, info = await retriever.search("t", question, context, top_k=5, strategy=strategy)
        return [dumped(d) for d in selected], info

    return asyncio.run(go("dense")), asyncio.run(go("lexical"))


@pytest.mark.parametrize("layout", ["resident", "columnar"])
def test_excluded_ids_retrieve_what_the_allow_list_of_the_others_retrieves(tmp_path, layout):
    state, docs = knowledge(tmp_path / "s.sqlite3")
    storage = local_storage(state, catalog=layout)
    ids = sorted({d.id for d in docs})
    rng = random.Random(7)
    changed = 0
    for k in range(40):
        query = " ".join(rng.sample(TOPICS, 2))
        plain = search(storage, query)
        seeds = plain[0][1]["seed_ids"]
        hidden = {s.split("@")[0] for s in rng.sample(seeds, min(2, len(seeds)))} | {"unknown"}
        deny = search(storage, query, excluded_ids=hidden)
        allow = search(storage, query, allowed_ids=[i for i in ids if i not in hidden])
        assert deny == allow
        changed += deny != plain
        assert not any(d[1]["id"] in hidden for part in deny for d in part[0])
    assert changed >= 30  # the hidden documents were ranked before: the comparison is not empty


def test_the_engine_passes_the_deny_list_to_the_release_check():
    import inspect

    from rvg.universal import scoped
    from rvg.universal.engine import UniversalEngine

    assert "excluded" in inspect.signature(scoped.check).parameters
    source = inspect.getsource(UniversalEngine)
    assert 'excluded=getattr(self.retriever, "excluded_ids", None)' in source
    assert 'excluded=getattr(retriever, "excluded_ids", None) or ()' in source


# --- configuration ------------------------------------------------------------------------------
def test_the_columnar_catalog_is_opt_in_and_off_hashes_as_before(tmp_path):
    knowledge(tmp_path / "s.sqlite3")[0].close()
    plain = load_config({"schema_version": 1, "state": {"backend": "sqlite", "path": "s.sqlite3"}})
    body = plain.model_dump(mode="json")
    for key in ("candidate_stage", "source", "external"):
        del body["vectors"][key]
    del body["catalog"]
    body["vectors"]["layout"] = resolved_layout(plain.vectors)
    assert plain.fingerprint() == canonical_hash(body)
    config = {
        "schema_version": 1,
        "state": {"backend": "sqlite", "path": "s.sqlite3"},
        "catalog": "columnar",
    }
    assert load_config(config).fingerprint() != plain.fingerprint()
    storage = open_storage(config, root=tmp_path)
    assert storage.catalogs.layout == "columnar"
    assert isinstance(storage.catalogs.build("t"), ColumnarCatalog)
    storage.close()
    default = open_storage(plain, root=tmp_path)
    assert default.catalogs.layout == "resident"
    default.close()
    with pytest.raises(ValueError, match="unknown_catalog_layout"):
        CatalogCache(SQLiteState(":memory:"), layout="other")


def test_the_input_of_every_row_matches_its_hash(tmp_path):
    state, docs = knowledge(tmp_path / "s.sqlite3")
    _, b = both(state)
    by_key = {}
    for d in docs:
        try:
            for _row, key, *_rest, text in catalog_entries(d):
                by_key[key] = text
        except ContractError:
            continue  # the blocked revision has no row
    assert all(b.input_hashes[i] == canonical_hash(by_key[b.row_keys[i]]) for i in range(len(b)))
    assert embedding_input(docs[0]) == docs[0].title + "\n" + docs[0].text
