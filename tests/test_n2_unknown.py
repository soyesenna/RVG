"""N2 (docs/UNIVERSAL_N2_PROTOCOL.md): forced relation judgements, programmatic cases, the three
arms. The fake writer omits an item whose evidence of one family states different numbers (the
writer instruction's behaviour, simulated); a passing test here says what the engine shows and
releases under a given wrong judgement — never how often Jev is wrong."""

import asyncio
import re

import httpx
import pytest

from rvg.runtime import CallContext
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore
from rvg.testing.fake_openrouter import FakeOpenRouter, providers
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

ARMS = {
    "A": None,
    "B": Arm.of("n2-keep-both", validity={"unknown_revision": "keep_both"}),
    "C": Arm.of("n2-tentative", validity={"unknown_revision": "tentative"}),
}
AS_OF = "2026-09-20T00:00:00Z"


def doc(identifier, text, *, family="fam", revision=1, day=1, effective=None, supersedes=()):
    parts = text.split(" | ")
    units, offset = [], 0
    for i, part in enumerate(parts):
        units.append({"id": f"u{i}", "start": offset, "end": offset + len(part)})
        offset += len(part) + 1
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family,
            "revision": revision,
            "title": "Project P",
            "text": " ".join(parts),
            "source": f"fixture://{identifier}",
            "published_at": f"2026-09-{day:02d}T00:00:00Z",
            "observed_at": f"2026-09-{day:02d}T00:00:00Z",
            "effective_from": effective,
            "units": units,
            "supersedes": list(supersedes),
        }
    )


def careful_writer(payload):
    """Answers with the first numeric unit, but omits the item (answerable=false) when two
    evidence units of one family state different numbers — the writer's conflict instruction."""
    by_family = {}
    for unit in payload["evidence"]:
        found = re.findall(r"\d+", unit["text"])
        if found:
            by_family.setdefault(unit.get("family"), set()).add(found[0])
    if any(len(values) > 1 for values in by_family.values()):
        return {"answerable": False, "short_answer": "", "sentences": []}
    for unit in payload["evidence"]:
        found = re.search(r"\d+", unit["text"])
        if found:
            return {
                "answerable": True,
                "short_answer": found.group(),
                "sentences": [{"text": unit["text"], "evidence_ids": [unit["id"]]}],
            }
    return {"answerable": False, "short_answer": "", "sentences": []}


def run(docs, arm, relation, *, revocations=(), store=None):
    async def go():
        nonlocal store
        store = store or TemporalStore(":memory:")
        for d in docs:
            store.upsert("t", d)
        for r in revocations:
            store.revoke("t", r)
        fake = FakeOpenRouter(writer=careful_writer, relation=relation)
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            from rvg.transport import JsonTransport

            p = providers(JsonTransport(client, timeout=5, max_attempts=2))
            retriever = TemporalRetriever(store, p["embedding"])
            await retriever.index("t", CallContext.with_timeout(10, max_embedding_calls=20))
            engine = UniversalEngine(
                retriever=retriever,
                decision=p["decision"],
                writer=p["writer"],
                extractor=p["extractor"],
                arm=ARMS[arm],
            )
            q = TemporalQuestion(
                text="What is the budget of project P?", as_of=AS_OF, known_at=AS_OF
            )
            ctx = CallContext.with_timeout(
                60,
                max_decision_calls=200,
                max_generation_calls=12,
                max_embedding_calls=40,
                max_semantic_proposer_calls=8,
                max_surface_proposer_calls=4,
                max_http_attempts=400,
            )
            return await engine.answer("t", q, ctx), store

    return asyncio.run(go())


def gate(result):
    return result["audit"][0]["gate"]


def uncertain(choice="UNKNOWN", p=0.94):
    return lambda state, qid: (choice, p)


CONFLICT = [
    doc("x", "The budget of project P is 500."),
    doc("y", "The budget of project P is 700.", day=5),
]


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_s1_an_uncertain_judgement_never_yields_a_confirmed_supersession(arm):
    result, _ = run(CONFLICT, arm, uncertain())
    g = gate(result)
    assert set(g["provenance"].values()) <= {"inferred"}
    if arm == "A":  # the adopted prior decides the true conflict by publication order (U17)
        assert g["statuses"]["x@1#u0"] == "SUPERSEDED" and result["status"] == "FULL"
        assert result["short_answer"] == "700"
    else:  # both stay in force and visible; the conflict reaches the writer, nothing is decided
        assert g["statuses"]["x@1#u0"] == g["statuses"]["y@1#u0"] == "ELIGIBLE"
        assert set(result["trace"]["first"]["units"]) >= {"x@1#u0", "y@1#u0"}
        assert result["status"] == "ABSTAIN"
        assert any(u.get("prior_conditions_met") for u in g["unresolved"])
    if arm == "C":
        assert g["tentative"] and g["tentative"][0]["provenance"] == "inferred"


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_a_true_revision_under_uncertainty_is_the_price_of_keeping_both(arm):
    # The same shapes as the conflict: a later same-family text with one value changed. The
    # engine cannot tell them apart without a confident judgement — that is N2's trade-off.
    revision = [
        doc("x", "The budget of project P is 500."),
        doc("y", "The budget of project P is 700.", day=5),
    ]
    result, _ = run(revision, arm, uncertain())
    assert result["status"] == ("FULL" if arm == "A" else "ABSTAIN")


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_s2_a_future_effective_revision_is_never_current(arm):
    docs = [
        doc("x", "The budget of project P is 500."),
        doc("y", "The budget of project P is 700.", day=5, effective="2026-12-01T00:00:00Z"),
    ]
    result, _ = run(docs, arm, uncertain())
    g = gate(result)
    assert g["statuses"]["y@1#u0"] == "FUTURE_EFFECTIVE" and result["short_answer"] == "500"


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_s4_a_partial_amendment_keeps_the_unchanged_units(arm):
    docs = [
        doc("x", "The budget of project P is 500. | The deadline of project P is October."),
        doc(
            "y",
            "The budget of project P is 650.",
            day=5,
            supersedes=[{"document_id": "x", "unit_ids": ["u0"]}],
        ),
    ]
    result, _ = run(docs, arm, uncertain())
    g = gate(result)
    assert g["statuses"]["x@1#u0"] == "SUPERSEDED" and g["provenance"]["x@1#u0"] == "confirmed_code"
    assert g["statuses"]["x@1#u1"] == "ELIGIBLE"


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_s5_revocation_and_other_families_are_untouched_by_the_prior(arm):
    docs = CONFLICT + [doc("z", "The budget of project P is 500.", family="other", day=6)]
    result, _ = run(docs, arm, uncertain(), revocations=["y"])
    g = gate(result)
    assert "y@1#u0" not in g["statuses"]  # revoked: never retrieved
    assert g["statuses"]["x@1#u0"] == "ELIGIBLE" and g["statuses"]["z@1#u0"] == "ELIGIBLE"


@pytest.mark.parametrize(("p", "confident"), [(0.84, False), (0.86, True)])
def test_the_conflicts_unknown_boundary_follows_the_fixed_confidence_rule(p, confident):
    result, _ = run(CONFLICT, "A", uncertain("CONFLICTS", p))
    g = gate(result)
    if confident:  # a confident CONFLICTS vetoes the prior: both stay in force
        assert g["conflicts"] and g["statuses"]["x@1#u0"] == "ELIGIBLE"
    else:  # below the fixed .85 the judgement is uncertain and the prior applies
        assert g["statuses"]["x@1#u0"] == "SUPERSEDED"


def test_an_injected_judgement_is_not_hidden_by_an_earlier_answer_s_cache():
    first, store = run(CONFLICT, "A", uncertain("CONFLICTS", 0.95))
    assert gate(first)["conflicts"]
    second, _ = run([], "A", uncertain(), store=store)  # same store and pair, a new answer
    assert gate(second)["statuses"]["x@1#u0"] == "SUPERSEDED"


def test_arms_are_stamped_and_limited_to_the_validity_overrides():
    result, _ = run(CONFLICT, "C", uncertain())
    assert result["arm"]["name"] == "n2-tentative" and len(result["arm"]["hash"]) == 64
    with pytest.raises(ValueError, match="arm_override_not_allowed"):
        Arm.of("bad", validity={"threshold": 0.1})
    base, _ = run(CONFLICT, "A", uncertain())
    assert base["arm"] is None
