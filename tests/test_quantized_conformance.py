"""N1F-4 G3 (parent docs/N1F4_PROTOCOL.md §4): the PORT-2 conformance scenarios that apply to an
index in this process, run on the TurboQuant candidate stage against the local exact index — the
same retrieval semantics (filters before top-k, snapshots, principals, pending documents, writes
after the codes) and the same N3 release outcomes. Each runs with the deployment ladder and a
short one that lets these small stores reach the candidate stage. Service-only scenarios of
`test_remote_index.py` (native precision, projector fencing, overlay limit) do not apply."""

import asyncio

import numpy as np
import pytest
import test_n3_release as n3
from matrix_worker import DIMS, SPACE, Queries, doc

from rvg.contracts import EmbeddingBatch
from rvg.errors import ContractError
from rvg.runtime import CallContext
from rvg.storage.factory import local_storage
from rvg.storage.local_exact import LocalExactIndex
from rvg.storage.quantized import LADDER, QuantizedCandidateIndex, build_codes
from rvg.storage.sqlite import SQLiteState
from rvg.temporal.access import TrustedPrincipal
from rvg.temporal.models import TemporalQuestion
from rvg.temporal.store import TemporalRetriever

FP = SPACE.fingerprint
LADDERS = [pytest.param(LADDER, id="ladder400"), pytest.param((12, 24), id="ladder12")]
PRINCIPALS = [None, TrustedPrincipal("alice", groups=("staff",)), TrustedPrincipal("bob")]


class Q(Queries):
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=float)

    async def embed(self, texts, *, purpose, context):
        return EmbeddingBatch(tuple(tuple(self.vector) for _ in texts), SPACE, ("f",), ())


def deployments(tmp_path, ladder, name="s"):
    """(state, quantized storage, local exact reference storage) over one SQLite store."""
    state = SQLiteState(tmp_path / f"{name}.sqlite3")
    exact = LocalExactIndex(state, matrix_dir=tmp_path / f"{name}.sqlite3.dense")
    quantized = local_storage(state, index=QuantizedCandidateIndex(exact, ladder=ladder))
    return state, quantized, local_storage(state)


def build(storage, tenant="t", dims=DIMS):
    return asyncio.run(build_codes(storage, tenant, FP, dims))


def populate(state, n=60, seed=0, tenant="t"):
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        d = doc(i, text=f"Fact number {i} about topic {i % 7}.")
        docs.append(d)
        state.upsert(tenant, d)
        if i % 9 != 0:  # every ninth waits for its vector: the pending path
            state.put_vector(tenant, FP, TemporalRetriever.input_text(d), rng.normal(size=DIMS))
    return docs


def search(storage, query, *, principal=None, allowed=None, at=None, tenant="t", known=None):
    q = TemporalQuestion(
        text="fact topic", as_of="2026-09-20T00:00:00Z", known_at=known or "2026-09-20T00:00:00Z"
    )

    async def go():
        retriever = TemporalRetriever(
            storage, Q(query), max_documents=200, principal=principal, allowed_ids=allowed
        )
        selected, info = await retriever.search(
            tenant, q, CallContext.with_timeout(30, max_embedding_calls=4), top_k=8, at=at
        )
        return (
            [d.key for d in selected],
            info["seed_ids"],
            [r["score"] for r in info.get("ranking", [])],
            info["pending"],
        )

    try:
        return asyncio.run(go())
    except ContractError as exc:
        return ("error", exc.code)


def queries(n=6, seed=11):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=DIMS) for _ in range(n)]


@pytest.mark.parametrize("ladder", LADDERS)
def test_the_candidate_stage_equals_the_local_exact_index(tmp_path, ladder):
    state, quantized, local = deployments(tmp_path, ladder)
    docs = populate(state)
    build(quantized)
    state.revoke("t", docs[4].id)
    state.set_access("t", docs[5].id, "staff", "allow")
    state.set_access("t", docs[7].id, "bob", "deny")
    snapshot = state.snapshot("t")
    late = doc(900, text="A late fact about topic 3.")
    state.upsert("t", late)
    state.put_vector("t", FP, TemporalRetriever.input_text(late), queries(1, 3)[0])
    ids = [d.id for d in docs]
    for query in queries():
        for principal in PRINCIPALS:
            assert search(quantized, query, principal=principal) == search(
                local, query, principal=principal
            )
        assert search(quantized, query, allowed=ids[::3]) == search(local, query, allowed=ids[::3])
        assert search(quantized, query, at=snapshot) == search(local, query, at=snapshot)
        assert search(quantized, query, known="2026-09-01T00:00:00Z") == search(
            local, query, known="2026-09-01T00:00:00Z"
        )
    stats = quantized.index.stats
    if ladder != LADDER:
        assert stats["overlay_rows"] > 0  # the late document came through the overlay
    assert stats["queries"] == 6 * 6
    build(quantized)
    before = stats["overlay_rows"]
    search(quantized, queries(1)[0])
    assert quantized.index.stats["overlay_rows"] == before  # rebuilt: nothing left to overlay


@pytest.mark.parametrize("ladder", LADDERS)
def test_tenants_never_mix_and_invalid_vectors_fail_like_the_local_index(tmp_path, ladder):
    state, quantized, local = deployments(tmp_path, ladder)
    docs = populate(state, n=40, seed=8)
    twin = doc(0, text="Another tenant's secret 999.")
    state.upsert("u", twin)
    state.put_vector(
        "u",
        FP,
        TemporalRetriever.input_text(twin),
        state.vector("t", FP, TemporalRetriever.input_text(docs[1])),
    )
    build(quantized, "t")
    build(quantized, "u")
    for query in queries(4):
        got = search(quantized, query)
        assert got == search(local, query) and all("secret" not in k for k in got[0])
    state.put_vector("t", FP, TemporalRetriever.input_text(docs[1]), [1.0, 2.0])  # wrong size
    assert (
        search(quantized, queries(1)[0])
        == search(local, queries(1)[0])
        == ("error", "temporal_cached_vector_invalid")
    )


def test_a_tenant_without_codes_is_refused_not_answered_exactly(tmp_path):
    state, quantized, local = deployments(tmp_path, LADDER)
    populate(state, n=20, seed=2)
    assert search(quantized, queries(1)[0]) == ("error", "quantized_codes_missing")
    assert search(local, queries(1)[0])[0]


N3_SCENARIOS = [
    (n3.test_baseline_every_mode_releases_the_same_answer_when_nothing_changes, {}),
    (n3.test_a_new_revision_during_generation, {}),
    *[(n3.test_a_revocation_during_generation_is_never_released, {"mode": m}) for m in n3.MODES],
    *[
        (n3.test_a_permission_withdrawn_during_generation_is_never_released, {"mode": m})
        for m in n3.MODES
    ],
    (n3.test_documents_under_acl_are_invisible_without_a_matching_principal, {}),
    (n3.test_an_unrelated_insert_blocks_only_the_current_mode, {}),
    (n3.test_one_relevant_change_is_retried_and_two_end_in_stale_after_retry, {}),
    (n3.test_duplicate_events_change_nothing, {}),
    (n3.test_a_late_older_revision_does_not_replace_the_newer_one, {}),
]


class Deployment:
    def __init__(self, storage):
        self.storage = storage

    def __getattr__(self, name):
        return getattr(self.storage.state, name)


@pytest.mark.parametrize("ladder", LADDERS)
@pytest.mark.parametrize(
    ("scenario", "kwargs"), N3_SCENARIOS, ids=[f"{s.__name__}{k}" for s, k in N3_SCENARIOS]
)
def test_n3_release_scenarios_hold_on_the_candidate_stage(
    tmp_path, monkeypatch, ladder, scenario, kwargs
):
    made = []

    def knowledge():
        _, quantized, _ = deployments(tmp_path, ladder, name=f"n3-{len(made)}")
        made.append(quantized)
        deployment = Deployment(quantized)
        rng = np.random.default_rng(3)
        put(
            deployment,
            n3.doc("p1", "The budget of project P is 500.", family="P"),
            n3.Q + 0.1 * np.eye(n3.DIMS)[1],
        )
        for k in range(25):
            v = rng.normal(size=n3.DIMS)
            v[0] = -abs(v[0]) * 0.1
            put(deployment, n3.doc(f"b{k:02d}", f"Background remark {k} on weather."), v)
        asyncio.run(build_codes(quantized, "t", n3.SPACE.fingerprint, n3.DIMS))
        return deployment

    def put(store, d, vector):
        store.upsert("t", d)
        store.put_vector("t", n3.SPACE.fingerprint, TemporalRetriever.input_text(d), vector)

    monkeypatch.setattr(n3, "knowledge", knowledge)
    monkeypatch.setattr(n3, "put", put)
    scenario(**kwargs)
    assert made and all(m.index.stats["queries"] > 0 for m in made)
