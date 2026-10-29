"""N3 invariants under generated event sequences (Hypothesis rule-based state machine): inserts,
new revisions, revocations, ACL changes, indexing, restarts, duplicate events and questions in
every release mode — with an event fired while the answer is being written.

I1 a released answer's documents are, at release, neither revoked nor hidden from the principal;
I2 in a snapshot mode every document the answer rests on was inserted by its snapshot;
I5 the matrix search equals the row-by-row reference restricted to the principal's documents;
I6 duplicate events leave the knowledge snapshot unchanged."""

import asyncio
import re
from pathlib import Path

import httpx
import numpy as np
import row_by_row_reference as bench
from hypothesis import HealthCheck, event, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule
from test_n3_release import ALICE, DIMS, QUESTION, SPACE, Fixed, doc

from rvg.errors import ContractError
from rvg.runtime import CallContext
from rvg.temporal.access import TrustedPrincipal
from rvg.temporal.models import TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore
from rvg.testing.fake_openrouter import FakeOpenRouter, providers
from rvg.transport import JsonTransport
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

PRINCIPALS = {"none": None, "alice": ALICE, "bob": TrustedPrincipal("bob")}
WHO = ("alice", "bob", "staff")
MODES = ("current", "as_of_snapshot", "latest_at_release")


def vector_of(k):
    v = np.zeros(DIMS)
    v[0], v[1 + k % (DIMS - 1)] = 1.0, 0.2 + 0.05 * k
    return v


def context():
    return CallContext.with_timeout(
        60,
        max_decision_calls=400,
        max_generation_calls=24,
        max_embedding_calls=40,
        max_semantic_proposer_calls=16,
        max_surface_proposer_calls=8,
        max_http_attempts=800,
    )


class Knowledge(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        import tempfile

        self.dir = tempfile.mkdtemp()
        self.path = Path(self.dir) / "s.sqlite3"
        self.store = TemporalStore(self.path)
        self.revisions = {}  # id -> latest revision
        self.pending_hook = None

    # --- events -------------------------------------------------------------------------
    def _insert(self, k, value):
        identifier = f"d{k}"
        revision = self.revisions.get(identifier, 0) + 1
        self.revisions[identifier] = revision
        self.store.upsert(
            "t",
            doc(
                identifier,
                f"The budget of project P is {value}.",
                family=f"f{k % 3}",
                revision=revision,
                day=min(1 + revision, 19),
            ),
        )

    def _index(self, k):
        identifier = f"d{k}"
        if identifier in self.revisions:
            d = self.store.documents("t", allowed_ids={identifier})
            for x in d:
                self.store.base.put_vector(
                    "t", SPACE.fingerprint, TemporalRetriever.input_text(x), vector_of(k)
                )

    @initialize(n=st.integers(2, 6), indexed=st.integers(0, 6))
    def seed(self, n, indexed):
        """A knowledge base to ask about: n documents, the first `indexed` with vectors."""
        for k in range(n):
            self._insert(k, 100 + k)
            if k < indexed:
                self._index(k)

    @rule(k=st.integers(0, 6), value=st.integers(100, 999))
    def insert(self, k, value):
        self._insert(k, value)

    @rule(k=st.integers(0, 6))
    def index(self, k):
        self._index(k)

    @rule(k=st.integers(0, 6))
    def revoke(self, k):
        self.store.revoke("t", f"d{k}")

    @rule(
        k=st.integers(0, 6),
        who=st.sampled_from(WHO),
        effect=st.sampled_from(["allow", "deny", None]),
    )
    def acl(self, k, who, effect):
        self.store.set_access("t", f"d{k}", who, effect)

    @rule()
    def restart(self):
        self.store.close()
        self.store = TemporalStore(self.path)

    @rule(k=st.integers(0, 6))
    def duplicates(self, k):  # I6
        identifier = f"d{k}"
        before = self.store.snapshot("t")
        if identifier in self.revisions:
            latest = self.store.conn.execute(
                "SELECT body FROM temporal_documents WHERE tenant='t' AND id=? ORDER BY revision DESC",
                (identifier,),
            ).fetchone()
            from rvg.temporal.models import TemporalDocument

            self.store.upsert("t", TemporalDocument.model_validate_json(latest[0]))
        if identifier in self.store.revoked("t"):
            self.store.revoke("t", identifier)
        assert self.store.snapshot("t") == before

    # --- a question, with one event while the answer is written -------------------------------
    @rule(
        mode=st.sampled_from(MODES),
        who=st.sampled_from(sorted(PRINCIPALS)),
        during=st.sampled_from(["none", "insert", "revoke", "deny", "index"]),
        k=st.integers(0, 6),
    )
    def ask(self, mode, who, during, k):
        principal = PRINCIPALS[who]
        fired = []

        def hook():
            if fired:
                return
            fired.append(True)
            if during == "insert":
                self._insert(k, 777)
            elif during == "revoke":
                self.store.revoke("t", f"d{k}")
            elif during == "deny" and principal is not None:
                self.store.set_access("t", f"d{k}", principal.principal_id, "deny")
            elif during == "index":
                self._index(k)

        def write(payload):
            hook()
            for unit in payload["evidence"]:
                found = re.search(r"\d+", unit["text"])
                if found:
                    return {
                        "answerable": True,
                        "short_answer": found.group(),
                        "sentences": [{"text": unit["text"], "evidence_ids": [unit["id"]]}],
                    }
            return {"answerable": False, "short_answer": "", "sentences": []}

        arm = None if mode == "current" else Arm.of("n3-" + mode, release={"mode": mode})

        async def go():
            fake = FakeOpenRouter(writer=write)
            async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
                p = providers(JsonTransport(client, timeout=5, max_attempts=2))
                engine = UniversalEngine(
                    retriever=TemporalRetriever(self.store, Fixed(), principal=principal),
                    decision=p["decision"],
                    writer=p["writer"],
                    extractor=p["extractor"],
                    arm=arm,
                )
                q = TemporalQuestion(
                    text=QUESTION, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
                )
                return await engine.answer("t", q, context())

        result = asyncio.run(go())
        event(f"ask {mode} during={during} -> {result['status']}")
        if result["status"] not in ("FULL", "PARTIAL"):
            return
        documents = {u.split("@")[0] for u in result.get("evidence_ids") or []}
        # I1 (checked right after the answer returns: nothing changed since its release check)
        assert not documents & self.store.revoked("t")
        assert not documents & self.store.hidden("t", principal)
        # I2
        if mode != "current":
            at = result["release"]["knowledge_snapshot"]
            for unit in result["evidence_ids"]:
                identifier, revision = unit.split("#")[0].split("@")
                seq = self.store.conn.execute(
                    "SELECT event_seq FROM temporal_documents WHERE tenant='t' AND id=? AND revision=?",
                    (identifier, int(revision)),
                ).fetchone()[0]
                assert seq is None or seq <= at

    # --- I5 --------------------------------------------------------------------------------
    @invariant()
    def matrix_search_equals_the_reference(self):
        if not self.revisions:
            return
        for principal in PRINCIPALS.values():
            hidden = self.store.hidden("t", principal)
            ids = sorted({i for i in self.revisions if i not in hidden})
            q = TemporalQuestion(
                text=QUESTION, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
            )

            async def outcome(search):
                # A shared contract error (e.g. a family grown past the document budget, K8)
                # is the same outcome for both searches.
                try:
                    return bench.summary(await search)
                except ContractError as exc:
                    return ("error", exc.code)

            async def both(principal=principal, ids=ids, q=q):
                new = TemporalRetriever(self.store, Fixed(), principal=principal).search(
                    "t", q, context(), top_k=4
                )
                old = bench.RowByRowRetriever(self.store, Fixed(), allowed_ids=ids).search(
                    "t", q, context(), top_k=4
                )
                return await outcome(new), await outcome(old)

            new, old = asyncio.run(both())
            assert new == old

    def teardown(self):
        import shutil

        self.store.close()  # with its index's generations (0.1.0, parent PUB-4)
        shutil.rmtree(self.dir, ignore_errors=True)


Knowledge.TestCase.settings = settings(
    max_examples=150,
    stateful_step_count=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
TestKnowledge = Knowledge.TestCase
