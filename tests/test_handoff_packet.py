"""PROD-1 (parent docs/PROD_1_PROTOCOL.md): the opt-in handoff packet. Off, nothing changes; on,
a released result and its release record are byte-identical and a result that was not released
carries `result.handoff`, which never shows the text of a unit the requester may not see now."""

import asyncio
import json
import re
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from session_fixtures import validate

import rvg
from rvg.contracts import EmbeddingBatch, EmbeddingSpace
from rvg.control import cascade
from rvg.runtime import CallContext
from rvg.temporal.access import Release, TrustedPrincipal
from rvg.temporal.models import AnswerDraft, TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore
from rvg.testing.fake_openrouter import FakeOpenRouter, providers, spread
from rvg.transport import JsonTransport
from rvg.universal import handoff
from rvg.universal.config import dimension_warning, universal_hash
from rvg.universal.engine import UniversalEngine

DIMS = 6
QUESTION = "What is the budget of project P?"
TARGET = "The budget of project P is 500."
Q = np.eye(DIMS)[0]
ALICE = TrustedPrincipal("alice", groups=("staff",))


def space(dims=DIMS):
    return EmbeddingSpace(
        provider="fixture",
        model="fixture",
        base_url="fixture://",
        dimensions=dims,
        normalize=False,
        similarity_metric="cosine",
        encoding_format="float",
        query_prefix="",
        document_prefix="",
        query_input_type=None,
        document_input_type=None,
        version=1,
    )


SPACE = space()


class Fixed:
    mode = "fixture"

    def __init__(self, dims=DIMS):
        self.space = space(dims)

    def describe(self):
        return self.space

    async def embed(self, texts, *, purpose, context):
        return EmbeddingBatch(tuple(tuple(Q) for _ in texts), SPACE, ("f",), ())


def doc(identifier, text, *, family=None, day=1):
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family or identifier,
            "revision": 1,
            "title": "Project P" if family == "P" else f"Note {identifier}",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": f"2026-09-{day:02d}T00:00:00Z",
            "observed_at": f"2026-09-{day:02d}T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
        }
    )


def knowledge():
    store = TemporalStore(":memory:")
    rng = np.random.default_rng(3)

    def put(d, vector):
        store.upsert("t", d)
        store.base.put_vector("t", SPACE.fingerprint, TemporalRetriever.input_text(d), vector)

    put(doc("p1", TARGET, family="P"), Q + 0.1 * np.eye(DIMS)[1])
    for k in range(25):
        v = rng.normal(size=DIMS)
        v[0] = -abs(v[0]) * 0.1
        put(doc(f"b{k:02d}", f"Background remark {k} on weather."), v)
    return store


class Doubting(FakeOpenRouter):
    """Every claim judged UNKNOWN: the floor holds every draft."""

    def _decide(self, body, *, malformed=False):
        response = super()._decide(body, malformed=malformed)
        data = json.loads(response.content)
        for answer in data["answers"].values():
            names = list(answer.get("probabilities") or {})
            if answer.get("type") == "choice" and "SUPPORTED" in names:
                top = (
                    "UNKNOWN" if "UNKNOWN" in names else next(n for n in names if n != "SUPPORTED")
                )
                answer["choice"], answer["probabilities"] = top, spread(names, top)
        return httpx.Response(200, json=data)


def once(hook):
    fired = []

    def write(payload):
        if not fired:
            fired.append(True)
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

    return write


def ask(store, *, make=FakeOpenRouter, writer=None, principal=None, packet=False, **retriever):
    """One answer through the engine (the policy's release block); returns (result, calls)."""

    async def go():
        fake = make(**({"writer": writer} if writer is not None else {}))
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(JsonTransport(client, timeout=5, max_attempts=2))
            engine = UniversalEngine(
                retriever=TemporalRetriever(store, Fixed(), principal=principal, **retriever),
                decision=p["decision"],
                writer=p["writer"],
                extractor=p["extractor"],
                **({"handoff_packet": True} if packet else {}),
            )
            q = TemporalQuestion(
                text=QUESTION, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
            )
            ctx = CallContext.with_timeout(
                60,
                max_decision_calls=400,
                max_generation_calls=24,
                max_embedding_calls=40,
                max_semantic_proposer_calls=16,
                max_surface_proposer_calls=8,
                max_http_attempts=800,
            )
            return await engine.answer("t", q, ctx), fake.bodies

    return asyncio.run(go())


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def without_packet(result):
    return {k: v for k, v in result.items() if k != "handoff"}


def test_a_released_result_is_byte_identical_and_carries_no_packet():
    off, calls_off = ask(knowledge())
    on, calls_on = ask(knowledge(), packet=True)
    assert off["status"] == "FULL" and "handoff" not in on
    assert canonical(on) == canonical(off) and calls_on == calls_off


def test_a_held_result_carries_the_packet_and_nothing_else_changes():
    off, calls_off = ask(knowledge(), make=Doubting)
    on, calls_on = ask(knowledge(), make=Doubting, packet=True)
    assert off["status"] == "ABSTAIN" and "handoff" not in off
    assert canonical(without_packet(on)) == canonical(off) and calls_on == calls_off
    packet = on["handoff"]
    validate(packet, "handoff_packet")
    assert (packet["kind"], packet["released"], packet["version"]) == (
        "handoff_draft",
        False,
        "handoff-v1",
    )
    assert "claim_not_supported" in packet["reasons"] and packet["premise"] == "not_checked"
    first = packet["stages"][0]
    assert first["stage"] == "stage1" and first["draft"]["sentences"]
    claims = first["control"]["claims"]
    assert claims and all(c["label"] != "SUPPORTED" for c in claims)
    assert {c["sentence"] for c in claims} == set(range(len(first["draft"]["sentences"])))
    assert set(first["control"]["signals"]) == {"answered", "qualifiers", "qualifiers_relative"}
    assert packet["handed_draft"] in ("stage2", "stage1")
    shown = {e["id"]: e for e in packet["evidence"]}
    assert set(first["evidence_ids"]) <= set(shown)
    assert all("text" in e and e["gate"] == "ELIGIBLE" for e in shown.values())
    assert packet["flags"] and all(f["note"] == handoff.FLAG_NOTE for f in packet["flags"])
    assert {f["kind"] for f in packet["flags"]} <= {"claim_not_supported", "answered_below"}


def test_a_writer_refusal_names_its_condition_and_output():
    declined = {"answerable": False, "short_answer": "", "sentences": []}
    result, _ = ask(knowledge(), writer=lambda payload: declined, packet=True)
    assert result["status"] == "ABSTAIN"
    packet = result["handoff"]
    assert packet["reasons"][0] == "writer_refusal" and packet["handed_draft"] is None
    writer = packet["stages"][0]["writer"]
    assert writer == {"refused": True, "refusal": "not_answerable", "raw": declined}


def test_a_document_revoked_during_generation_is_never_shown():
    store = knowledge()
    result, _ = ask(store, writer=once(lambda: store.revoke("t", "p1")), packet=True)
    assert result["status"] == "ABSTAIN" and result["reason"] == "revoked_before_release"
    packet = result["handoff"]
    validate(packet, "handoff_packet")
    assert packet["reasons"][0] == "release:blocked_documents"
    unit = next(e for e in packet["evidence"] if e["id"] == "p1@1#u0")
    assert unit["withheld"] == "revoked" and "text" not in unit
    stage = packet["stages"][0]
    assert stage["draft"]["withheld"] == "window_has_withheld_unit"
    assert all(c["text"]["withheld"] for c in stage["control"]["claims"])
    assert "project P is 500" not in canonical(packet)


def test_a_permission_withdrawn_during_generation_is_never_shown():
    store = knowledge()
    store.set_access("t", "p1", "staff", "allow")
    deny = once(lambda: store.set_access("t", "p1", "alice", "deny"))
    result, _ = ask(store, writer=deny, principal=ALICE, packet=True)
    assert result["status"] == "ABSTAIN"
    packet = result["handoff"]
    unit = next(e for e in packet["evidence"] if e["id"] == "p1@1#u0")
    assert unit["withheld"] == "not_visible" and "project P is 500" not in canonical(packet)


def trail():
    units = [
        {"id": "d1@1#u0", "document_id": "d1", "revision": 1, "title": "D1", "text": "One is 1."},
        {"id": "d2@1#u0", "document_id": "d2", "revision": 1, "title": "D2", "text": "Two is 2."},
    ]
    draft = {
        "answerable": True,
        "short_answer": "1",
        "sentences": [{"text": "One is 1.", "evidence_ids": ["d1@1#u0"]}],
    }
    control = {
        "claims": [
            {"text": "One is 1.", "sentence": 0, "cited": ["d1@1#u0"], "label": "UNCERTAIN"}
        ],
        "signals": {"answered": {"score": 0.5, "threshold": 0.84, "passed": False}},
        "eligible": False,
        "accepted": False,
        "certified_kind": True,
    }
    return [
        {
            "stage": "stage1",
            "evidence_ids": [u["id"] for u in units],
            "units": units,
            "writer": {"refused": False, "refusal": None, "raw": None},
            "draft": draft,
            "control": control,
            "status": "ABSTAIN",
            "reason": "claim_verification_failed",
        }
    ]


def gates(**statuses):
    return [{"gate": {"statuses": {"d1@1#u0": "ELIGIBLE", "d2@1#u0": "ELIGIBLE", **statuses}}}]


HELD = {"status": "ABSTAIN", "reason": "claim_verification_failed"}


@pytest.mark.parametrize(
    "view, withheld",
    [
        (handoff.sight(None, audits=gates(), excluded={"d2"}), "excluded"),
        (handoff.sight(None, audits=gates(), allowed={"d1"}), "not_selected"),
        (handoff.sight(None, audits=gates(**{"d2@1#u0": "SUPERSEDED"})), "gate:SUPERSEDED"),
        (
            handoff.sight(Release(True, None, 1, 1, hidden=frozenset({"d2"})), audits=gates()),
            "not_visible",
        ),
        (
            handoff.sight(Release(False, "revoked_before_release", 1, 1, ("d2",)), audits=gates()),
            "revoked",
        ),
    ],
)
def test_a_withheld_unit_withholds_every_text_its_stage_wrote(view, withheld):
    packet = handoff.packet(HELD, trail(), view)
    validate(packet, "handoff_packet")
    by_id = {e["id"]: e for e in packet["evidence"]}
    assert by_id["d1@1#u0"]["text"] == "One is 1."
    assert by_id["d2@1#u0"] == {
        "id": "d2@1#u0",
        "gate": by_id["d2@1#u0"]["gate"],
        "withheld": withheld,
    }
    stage = packet["stages"][0]
    assert stage["draft"]["withheld"] == "window_has_withheld_unit"
    assert stage["control"]["claims"][0]["label"] == "UNCERTAIN"
    assert "Two is 2" not in canonical(packet) and "One is 1" not in canonical(stage)
    assert packet["reasons"] == ["claim_not_supported", "profile:answered"]
    assert [f["kind"] for f in packet["flags"]] == ["claim_not_supported", "answered_below"]


def test_visibility_that_cannot_be_read_withholds_every_unit():
    view = handoff.sight(None, audits=gates(), unknown="visibility_unknown")
    packet = handoff.packet(HELD, trail(), view)
    assert all(e.get("withheld") == "visibility_unknown" for e in packet["evidence"])
    assert "One is 1" not in canonical(packet)


def test_a_visible_trail_shows_its_texts():
    packet = handoff.packet(HELD, trail(), handoff.sight(None, audits=gates()))
    stage = packet["stages"][0]
    assert stage["draft"] == {
        "short_answer": "1",
        "sentences": [{"text": "One is 1.", "evidence_ids": ["d1@1#u0"]}],
    }
    assert packet["handed_draft"] == "stage1" and packet["comparison"] is None


class StubWriter:
    """The first unit that states a number, in the order given (a reversed pool can differ)."""

    async def generate_structured(self, request, context):
        for unit in request.payload["evidence"]:
            found = re.search(r"\d+", unit["text"])
            if found:
                value = {
                    "answerable": True,
                    "short_answer": found.group(),
                    "sentences": [{"text": unit["text"], "evidence_ids": [unit["id"]]}],
                }
                return SimpleNamespace(value=AnswerDraft.model_validate(value))
        empty = {"answerable": False, "short_answer": "", "sentences": []}
        return SimpleNamespace(value=AnswerDraft.model_validate(empty))


class StubController:
    """FULL on the wider evidence only."""

    async def control(self, question, evidence, draft, context, *, slots=None, trail=None):
        full = len(evidence) > 3
        if trail is not None:
            trail["draft"] = draft.model_dump(mode="json")
            trail["control"] = {"claims": [], "signals": {}, "eligible": full, "accepted": full}
        body = {"text": draft.short_answer, "sentences": [], "missing": [], "action": None}
        if full:
            return {**body, "status": "FULL"}
        return {**body, "status": "ABSTAIN", "reason": "claim_verification_failed"}


def test_the_consistency_comparison_is_in_the_packet():
    units = [
        {
            "id": f"n{i}@1#u0",
            "document_id": f"n{i}",
            "revision": 1,
            "start": 0,
            "title": f"N{i}",
            "text": text,
        }
        for i, text in enumerate(
            [
                "No figure here.",
                "Nothing numeric.",
                "Plain words.",
                "It is 500.",
                "Words.",
                "It is 650.",
                "More words.",
                "Still words.",
                "Last words.",
                "End.",
            ]
        )
    ]

    async def retrieve(text, k):
        return units

    def run(stages):
        q = TemporalQuestion(
            text="What is it?", as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
        )
        return asyncio.run(
            cascade.run(
                q,
                retrieve=retrieve,
                writer=StubWriter(),
                controller=StubController(),
                asker=None,
                context=None,
                resolve_scope=False,
                **({"handoff": stages} if stages is not None else {}),
            )
        )

    stages = []
    assert run(None) == run(stages)
    assert [s["stage"] for s in stages] == ["stage1", "stage2", "partner_draft"]
    audits = [{"gate": {"statuses": {u["id"]: "ELIGIBLE" for u in units}}}]
    packet = handoff.packet(
        {"status": "ABSTAIN", "reason": "no_draft"}, stages, handoff.sight(None, audits=audits)
    )
    validate(packet, "handoff_packet")
    assert packet["reasons"] == ["writer_refusal", "consistency_disagreement"]
    assert packet["handed_draft"] == "stage2"
    comparison = packet["comparison"]
    assert (comparison["first"], comparison["second"], comparison["partner"]) == (
        None,
        "500",
        "650",
    )
    assert comparison["agreed"] is False and comparison["judge"] == "same_text"
    assert comparison["sentences_added"] == [{"text": "It is 500.", "evidence_ids": ["n3@1#u0"]}]


def test_the_session_option_and_its_envelope():
    async def go(packet):
        store, fake = rvg.TemporalStore(":memory:"), Doubting()
        store.upsert("fixture", doc("p1", TARGET, family="P"))
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(rvg.JsonTransport(client, timeout=5, max_attempts=3))
            session = rvg.open_session(
                store, rvg.Settings(_env_file=None), providers=p, handoff_packet=packet
            )
            try:
                await TemporalRetriever(store, p["embedding"]).index("fixture", budget())
                context = rvg.RequestContext(tenant="fixture", principal=None, call=budget())
                q = {
                    "text": QUESTION,
                    "as_of": "2026-09-20T00:00:00Z",
                    "known_at": "2026-09-20T00:00:00Z",
                }
                return await session.answer(q, context), session.capabilities()["session"]
            finally:
                await session.aclose()

    def budget():
        return CallContext.with_timeout(
            60,
            max_decision_calls=400,
            max_generation_calls=24,
            max_embedding_calls=40,
            max_semantic_proposer_calls=16,
            max_surface_proposer_calls=8,
            max_http_attempts=800,
        )

    on, table_on = asyncio.run(go(True))
    off, table_off = asyncio.run(go(False))
    validate(on, "answer_envelope")
    assert on["result"]["status"] == "ABSTAIN" and on["result"]["handoff"]["released"] is False
    assert "handoff" not in off["result"]
    assert (table_on["handoff_packet"], table_off["handoff_packet"]) == (True, False)


def test_a_released_result_with_a_packet_fails_the_envelope_schema():
    import jsonschema

    from rvg.session import API_VERSION

    envelope = {
        "api_version": API_VERSION,
        "request_id": "r1",
        "result": {
            "status": "FULL",
            "as_of": "x",
            "known_at": "x",
            "snapshot": 0,
            "config_hash": "0" * 64,
            "storage": {},
            "audit": [],
        },
        "usage": {
            "http_attempts": 0,
            "decision_calls": 0,
            "embedding_calls": 0,
            "generation_calls": 0,
            "schema_repairs": 0,
            "reported_cost_usd": 0,
            "unknown_cost_calls": 0,
        },
    }
    validate(envelope, "answer_envelope")
    held = handoff.packet(HELD, trail(), handoff.sight(None, audits=gates()))
    envelope["result"]["handoff"] = held
    with pytest.raises(jsonschema.ValidationError):
        validate(envelope, "answer_envelope")
    envelope["result"]["status"] = "ABSTAIN"
    validate(envelope, "answer_envelope")


def test_d_dim_decl_is_a_warning_only():
    assert universal_hash().startswith("24777fc8")
    warned = dimension_warning(Fixed(DIMS))
    assert warned and "3072" in warned and f"{DIMS} dimensions" in warned
    assert dimension_warning(Fixed(3072)) is None and dimension_warning(None) is None


def test_a_record_keeps_the_packet_only_when_there_is_one():
    from rvg.testing.records import RESULT_KEYS, answer_one

    async def go(packet):
        store, fake = knowledge(), Doubting()
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(JsonTransport(client, timeout=5, max_attempts=2))
            engine = UniversalEngine(
                retriever=TemporalRetriever(store, Fixed()),
                decision=p["decision"],
                writer=p["writer"],
                extractor=p["extractor"],
                handoff_packet=packet,
            )
            q = TemporalQuestion(
                text=QUESTION, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
            )
            ctx = CallContext.with_timeout(60, max_decision_calls=400, max_http_attempts=800)
            return await answer_one(engine, "t", q, request_id="r1", context=ctx)

    on, off = asyncio.run(go(True)), asyncio.run(go(False))
    assert tuple(off["result"]) == RESULT_KEYS
    assert on["result"]["handoff"]["released"] is False
    assert {k: v for k, v in on["result"].items() if k != "handoff"} == off["result"]
