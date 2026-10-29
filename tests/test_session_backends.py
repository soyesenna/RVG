"""SEP-04 G04-OBS on real backends: a session over PostgreSQL state with pgvector or Qdrant
answers readiness with bounded read-only reads, changes nothing, and never returns a connection
string. Needs `DEEPJEV_TEST_POSTGRES_DSN` and `DEEPJEV_TEST_PGVECTOR_DSN` / `DEEPJEV_TEST_QDRANT_URL`;
a missing service skips its cases (they are not passes)."""

import asyncio
import json
import os
import uuid

import httpx
import pytest
from session_fixtures import A, B, context, question, validate

import rvg
from rvg.testing.fake_openrouter import FakeOpenRouter, providers

PG = os.environ.get("DEEPJEV_TEST_POSTGRES_DSN")
PGV = os.environ.get("DEEPJEV_TEST_PGVECTOR_DSN")
QDRANT = os.environ.get("DEEPJEV_TEST_QDRANT_URL")
CASES = [
    pytest.param("local_exact", marks=pytest.mark.skipif(not PG, reason="no PostgreSQL")),
    pytest.param("pgvector", marks=pytest.mark.skipif(not (PG and PGV), reason="no pgvector")),
    pytest.param("qdrant", marks=pytest.mark.skipif(not (PG and QDRANT), reason="no Qdrant")),
]


def config(index, tag, tmp_path):
    vectors = {"index": index, "matrix_dir": str(tmp_path / "m")}
    if index == "pgvector":
        vectors |= {"dsn_env": "RVG_TEST_PGV", "schema_name": f"ix_{tag}"}
    elif index == "qdrant":
        vectors |= {"url_env": "RVG_TEST_QDRANT", "prefix": f"t{tag}"}
    return {
        "schema_version": 1,
        "state": {"backend": "postgres", "dsn_env": "RVG_TEST_PG", "schema_name": f"st_{tag}"},
        "vectors": vectors,
    }


def cleanup(index, tag):
    import psycopg

    with psycopg.connect(PG, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS st_{tag} CASCADE")
    if index == "pgvector":
        with psycopg.connect(PGV, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS ix_{tag} CASCADE")
    if index == "qdrant":
        with httpx.Client(base_url=QDRANT) as client:
            for c in client.get("/collections").json()["result"]["collections"]:
                if c["name"].startswith(f"t{tag}_"):
                    client.delete(f"/collections/{c['name']}")


def observed(state, tenant="fixture"):
    return (state.generations(tenant), state.outbox_mark(tenant), state.digest(tenant))


@pytest.mark.parametrize("index", CASES)
def test_readiness_on_real_backends_is_read_only_and_secret_free(index, tmp_path):
    tag = uuid.uuid4().hex[:10]
    environ = {"RVG_TEST_PG": PG, "RVG_TEST_PGV": PGV or "", "RVG_TEST_QDRANT": QDRANT or ""}

    async def go():
        fake = FakeOpenRouter()
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(rvg.JsonTransport(client, timeout=5, max_attempts=3))
            session = rvg.open_session(
                config(index, tag, tmp_path),
                rvg.Settings(_env_file=None),
                root=tmp_path,
                environ=environ,
                providers=p,
            )
            try:
                ingest = await session.ingest([A, B], context())
                before = observed(session.storage.state)
                fresh = await session.health("readiness", context())
                answered = await session.answer(question(), context())
                again = await session.health("readiness", context())
                unchanged = observed(session.storage.state) == before
                if session.storage.projector is not None:  # project, then read readiness again
                    space = p["embedding"].describe()
                    session.storage.projector.sync("fixture", space.fingerprint, space.dimensions)
                projected = await session.health("readiness", context())
                session.storage.state.conn.close()  # the state server goes away
                broken = await session.health("readiness", context())
                return ingest, fresh, answered, (again, projected), unchanged, broken
            finally:
                await session.aclose()

    try:
        ingest, fresh, answered, after_answer, unchanged, broken = asyncio.run(go())
    finally:
        cleanup(index, tag)
    again, projected = after_answer
    assert [r["outcome"] for r in ingest["results"]] == ["STORED", "STORED"]
    assert unchanged  # two readiness reads and an answer changed no generation or outbox entry
    assert answered["result"]["status"] == "FULL"
    checks = {c["component"]: c for c in fresh["checks"]}
    assert checks["state_store"]["status"] == "OK"
    if index == "local_exact":
        assert checks["vector_index"]["code"] == "in_process_exact_index"
        assert fresh["status"] == "OK"
    else:
        # Nothing projected yet (queries rank through the exact overlay): degraded, not broken.
        assert checks["vector_index"]["code"] == "space_not_projected_yet"
        assert checks["projection"]["code"] == "projection_frontier_behind"
        assert fresh["status"] == again["status"] == "DEGRADED"
        after = {c["component"]: c for c in projected["checks"]}
        assert after["vector_index"]["code"] == "count_ok"
        assert after["projection"]["status"] == "OK" and projected["status"] == "OK"
    state_check = next(c for c in broken["checks"] if c["component"] == "state_store")
    assert broken["status"] == "NOT_READY" and state_check["code"].startswith("read_failed:")
    for report in (fresh, again, projected, broken):
        validate(report, "health")
        dumped = json.dumps(report)
        assert PG not in dumped and "password" not in dumped.lower()
        assert (PGV or "\0") not in dumped and (QDRANT or "\0") not in dumped
