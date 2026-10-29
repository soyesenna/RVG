import asyncio
import re

from rvg.contracts import BooleanAnswer, ChoiceAnswer, DecisionBatch, ScoreAnswer
from rvg.control.compose import Slots
from rvg.providers.generation import GenerationResult
from rvg.runtime import CallContext
from rvg.temporal.access import Release as Grant
from rvg.temporal.models import AnswerDraft, TemporalDocument, TemporalQuestion
from rvg.universal import universal_hash
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm


def document(identifier, title, text, *, revision=1, day="01"):
    parts = text.split(" | ")
    units, offset = [], 0
    for i, part in enumerate(parts):
        units.append({"id": f"u{i}", "start": offset, "end": offset + len(part)})
        offset += len(part) + 1
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": identifier,
            "revision": revision,
            "title": title,
            "text": " ".join(parts),
            "source": f"fixture://{identifier}",
            "published_at": f"2026-09-{day}T00:00:00Z",
            "observed_at": f"2026-09-{day}T00:00:00Z",
            "units": units,
        }
    )


class Store:
    def snapshot(self, tenant):
        return 7

    def acl_epoch(self, tenant):  # a port method the engine reads before every attempt
        return 0


class Release:
    def authorize(self, tenant, document_ids, principal):
        return Grant(True, None, 7, 0)


class Storage:
    state, release = Store(), Release()

    def describe(self):
        return {"state": "fake", "index": "fake", "precision": None, "config_hash": None}


class Retriever:
    principal = None

    def __init__(self, docs):
        self.docs, self.storage, self.queries = docs, Storage(), []
        self.store = self.storage.state

    async def search(self, tenant, question, context, *, strategy, top_k, expand):
        self.queries.append(question.text)
        return self.docs[:top_k], {"seed_ids": [d.key for d in self.docs[:top_k]]}


class Decision:
    provider, model, mode = "fake", "fake-jev", "mock"

    async def evaluate(self, request, context):
        answers = {}
        for qid, q in request.questions.items():
            if q.type == "choice":
                names = list(q.criteria)
                top = "SUPPORTED" if "SUPPORTED" in names else names[0]
                answers[qid] = ChoiceAnswer(
                    choice=top, probabilities={k: 0.94 if k == top else 0.02 for k in names}
                )
            elif q.type == "score":  # the answerable gate: "states the asked item explicitly"
                answers[qid] = ScoreAnswer(
                    score=4, probabilities={str(i): float(i == 4) for i in range(5)}
                )
            elif re.fullmatch(r"[abd]\d+_\d+", qid):
                i, j = map(int, qid[1:].split("_"))
                answers[qid] = BooleanAnswer(
                    probability_true=0.95 if qid[0] in "ab" and i == j else 0.02
                )
            else:
                answers[qid] = BooleanAnswer(probability_true=0.95)
        return DecisionBatch(answers, self.provider, self.model)


class Writer:
    provider, model, mode = "fake", "fake-llm", "mock"

    async def generate_structured(self, request, context):
        evidence = request.payload["evidence"]
        hit = next((e for e in evidence if "500" in e["text"]), None)
        if hit is None:
            value = {"answerable": False, "short_answer": "", "sentences": []}
        else:
            value = {
                "answerable": True,
                "short_answer": "500",
                "sentences": [{"text": "The budget is 500.", "evidence_ids": [hit["id"]]}],
            }
        return GenerationResult(AnswerDraft.model_validate(value), {})


class Extractor:
    provider, model, mode = "fake", "fake-llm", "mock"

    async def generate_structured(self, request, context):
        assert request.response_type is Slots
        return GenerationResult(
            Slots.model_validate({"slots": [{"id": "s0", "ask": "budget"}]}), {}
        )


def question():
    return TemporalQuestion(
        text="What is the budget of project A?",
        as_of="2026-09-20T00:00:00Z",
        known_at="2026-09-20T00:00:00Z",
        locale="en-US",
    )


def test_engine_releases_a_supported_answer_with_the_policy_hash_and_an_audit():
    docs = [
        document("a", "Project A", "The budget is 500. | The deadline is October 1."),
        document("b", "Project A", "Applications open in March."),
    ]
    retriever = Retriever(docs)
    # U24 (0.1.0.dev27): this store fake has no release-check interface (outbox, ACL plane), so
    # the test pins `current`; the policy's scoped default runs on real stores (test_u24_*)
    engine = UniversalEngine(
        retriever=retriever,
        decision=Decision(),
        writer=Writer(),
        extractor=Extractor(),
        arm=Arm.of("u24-current", release={"mode": "current"}),
    )
    ctx = CallContext.with_timeout(5, max_decision_calls=40, max_generation_calls=9)
    found = asyncio.run(engine.answer("demo", question(), ctx))
    assert found["status"] == "FULL" and found["stage"] == 1
    assert found["text"] == "The budget is 500." and found["evidence_ids"][0] == "a@1#u0"
    assert found["scope"]["status"] == "RESOLVED" and found["scope"]["anchor"] == "Project A"
    assert found["config_hash"] == universal_hash() and found["snapshot"] == 7
    assert found["audit"][0]["gate"]["complete_within_local_snapshot"] is True
    assert retriever.queries == ["What is the budget of project A?"]


def test_engine_forwards_the_policy_validity_rules_to_the_gate(monkeypatch):
    """U14: the adopted revision prior (U10) must reach `select_evidence` on the product path."""
    from rvg.universal import engine as module
    from rvg.universal import universal

    captured = {}

    class Incomplete:
        complete = False

    async def fake_select(docs, question, **kwargs):
        captured.update(kwargs)
        return Incomplete()

    monkeypatch.setattr(module, "select_evidence", fake_select)
    engine = UniversalEngine(
        retriever=Retriever([]), decision=Decision(), writer=Writer(), extractor=Extractor()
    )
    ctx = CallContext.with_timeout(5, max_decision_calls=4, max_generation_calls=1)
    ranked, evidence, _ = asyncio.run(engine.units("demo", question(), "x", 3, ctx, None))
    rules = universal()["validity"]
    assert ranked == [] and evidence.complete is False
    assert captured["revision_prior"] == rules["implicit_revision_prior"]
    assert captured["revision_prior_scope"] == rules["implicit_revision_prior_scope"]
    assert captured["revision_overlap"] == rules["implicit_revision_overlap"]
    assert captured["pairing"] == rules.get("pairing", "unit")


def test_engine_abstains_when_nothing_in_force_answers_and_widening_finds_nothing_new():
    docs = [document("c", "Project C", "Applications open in March.")]
    engine = UniversalEngine(
        retriever=Retriever(docs), decision=Decision(), writer=Writer(), extractor=Extractor()
    )
    ctx = CallContext.with_timeout(5, max_decision_calls=40, max_generation_calls=9)
    found = asyncio.run(engine.answer("demo", question(), ctx))
    assert found["status"] == "ABSTAIN" and found["reason"] == "no_draft"
    assert found["trace"]["second"] == {"skipped": "no_new_evidence"}


def test_prefer_latest_revision_moves_only_the_older_of_coexisting_revisions():
    """U10-b: when two revisions of one family are both in force, the newest comes first; other
    families and single-revision families keep their retrieval order."""
    from rvg.universal.engine import prefer_latest_revision

    def unit(identifier, family, revision, day):
        return {
            "id": identifier,
            "family": family,
            "revision": revision,
            "published_at": f"2026-09-{day}T00:00:00Z",
        }

    ranked = [
        unit("old#u0", "fam", 1, "01"),
        unit("other#u0", "other", 1, "05"),
        unit("old#u1", "fam", 1, "01"),
        unit("new#u0", "fam", 2, "10"),
        unit("solo#u0", "solo", 3, "02"),
    ]
    ordered = [u["id"] for u in prefer_latest_revision(ranked)]
    assert ordered == ["other#u0", "new#u0", "solo#u0", "old#u0", "old#u1"]
    single = [unit("a#u0", "fam", 1, "01"), unit("b#u0", "fam2", 1, "02")]
    assert [u["id"] for u in prefer_latest_revision(single)] == ["a#u0", "b#u0"]


def test_interleave_revisions_alternates_coexisting_revisions_at_the_family_position():
    """U10-c: units of two in-force revisions of one family alternate (newest first) where the
    family first appears; other families keep their retrieval order."""
    from rvg.universal.engine import interleave_revisions

    def unit(identifier, family, revision, day):
        return {
            "id": identifier,
            "family": family,
            "revision": revision,
            "published_at": f"2026-09-{day}T00:00:00Z",
        }

    ranked = [
        unit("old#u0", "fam", 1, "01"),
        unit("other#u0", "other", 1, "05"),
        unit("old#u1", "fam", 1, "01"),
        unit("new#u0", "fam", 2, "10"),
        unit("new#u1", "fam", 2, "10"),
        unit("solo#u0", "solo", 3, "02"),
    ]
    ordered = [u["id"] for u in interleave_revisions(ranked)]
    assert ordered == ["new#u0", "old#u0", "new#u1", "old#u1", "other#u0", "solo#u0"]
    single = [unit("a#u0", "fam", 1, "01"), unit("b#u0", "fam2", 1, "02")]
    assert [u["id"] for u in interleave_revisions(single)] == ["a#u0", "b#u0"]
