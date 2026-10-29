"""N4: the U12-d conflict rewrite as an engine arm (`cascade.conflict_partial = rewrite`). Off by
default: the policy's path and every recorded field stay as they were. On: a draft the floor
blocked for conflict alone is written once in the conflict form and judged on its cited units;
disagreeing values come out as a `source_disagreement_report` (never counted as the answer), and
the release fence still applies to every document the rewrite rests on."""

import asyncio
import json

import httpx
import numpy as np
import pytest

from rvg.contracts import EmbeddingBatch, EmbeddingSpace
from rvg.runtime import CallContext
from rvg.temporal.models import TemporalDocument, TemporalQuestion
from rvg.temporal.store import TemporalRetriever, TemporalStore
from rvg.testing.fake_openrouter import FakeOpenRouter, providers, spread
from rvg.transport import JsonTransport
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

DIMS = 4
SPACE = EmbeddingSpace(
    provider="fixture",
    model="fixture",
    base_url="fixture://",
    dimensions=DIMS,
    normalize=False,
    similarity_metric="cosine",
    encoding_format="float",
    query_prefix="",
    document_prefix="",
    query_input_type=None,
    document_input_type=None,
    version=1,
)
REWRITE = Arm.of("n4-rewrite", cascade={"conflict_partial": "rewrite"})
QUESTION = "What is the budget of project P?"


class Fixed:
    mode = "fixture"

    def describe(self):
        return SPACE

    async def embed(self, texts, *, purpose, context):
        return EmbeddingBatch(tuple((1.0, 0.0, 0.0, 0.0) for _ in texts), SPACE, ("f",), ())


def doc(identifier, text, family):
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family,
            "revision": 1,
            "title": f"Report {identifier}",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": "2026-09-01T00:00:00Z",
            "observed_at": "2026-09-01T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
        }
    )


class ConflictAware(FakeOpenRouter):
    """Claims are CONFLICTING-heavy when the judged evidence holds both disagreeing values and
    SUPPORTED when it holds one; the writer answers with one value, or with both (each citing
    its own source) when asked for the conflict form."""

    def __init__(self, values, *, on_conflict_write=None):
        super().__init__()
        self.values, self.on_conflict_write = values, on_conflict_write
        self.conflict_writes = 0

    def _chat(self, body):
        system = body["messages"][0]["content"]
        payload = json.loads(body["messages"][-1]["content"])
        if body["response_format"]["json_schema"]["name"] == "temporal_answer_v1":
            evidence = payload["evidence"]
            if "CONFLICT FORM" in system:
                self.conflict_writes += 1
                if self.on_conflict_write:
                    self.on_conflict_write()
                sentences = [
                    {"text": f"{e['title']} states a budget of {v}.", "evidence_ids": [e["id"]]}
                    for e in evidence
                    for v in self.values
                    if v in e["text"]
                ]
                value = {
                    "answerable": bool(sentences),
                    "short_answer": " | ".join(v for v in self.values),
                    "sentences": sentences,
                }
            else:
                first = next(e for e in evidence if self.values[0] in e["text"])
                value = {
                    "answerable": True,
                    "short_answer": self.values[0],
                    "sentences": [
                        {"text": f"The budget is {self.values[0]}.", "evidence_ids": [first["id"]]}
                    ],
                }
            return httpx.Response(
                200,
                json={
                    "id": "gen",
                    "model": body["model"],
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(value)},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001},
                },
            )
        return super()._chat(body)

    def _decide(self, body, *, malformed=False):
        evidence = json.dumps(body.get("state", {}).get("evidence", []))
        both = all(v in evidence for v in self.values)
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "choice" and "CONFLICTING" in q["criteria"]:
                names = list(q["criteria"])
                if both:
                    p = {n: 0.02 for n in names}
                    p["CONFLICTING"], p["SUPPORTED"] = 0.60, 1 - 0.02 * (len(names) - 2) - 0.60
                else:
                    p = spread(names, "SUPPORTED")
                top = max(p, key=p.get)
                answers[qid] = {"type": "choice", "choice": top, "probabilities": p}
            elif q["type"] == "choice":
                names = list(q["criteria"])
                top = "SUPPORTED" if "SUPPORTED" in names else names[0]
                answers[qid] = {
                    "type": "choice",
                    "choice": top,
                    "probabilities": spread(names, top),
                }
            elif q["type"] == "score":
                top = len(q["criteria"]) - 1
                answers[qid] = {
                    "type": "score",
                    "score": top,
                    "probabilities": {str(i): float(i == top) for i in range(len(q["criteria"]))},
                }
            elif qid.startswith("r"):
                # A sweep: the uncited units refute a claim when they state the other value.
                claims = json.dumps(body.get("state", {}).get("claims", []))
                other = any(v in evidence and v not in claims for v in self.values)
                answers[qid] = {"type": "noul", "noul": 0.9 if other else 0.02}
            else:
                answers[qid] = {"type": "noul", "noul": 0.95}
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": 20, "output_tokens": 1, "cost": 0.0002},
                "rounding": {"probabilityDecimals": 6},
            },
        )


def ask(docs, arm, *, values=("500", "700"), on_conflict_write=None, store=None):
    store = store or TemporalStore(":memory:")

    async def go():
        for d in docs:
            store.upsert("t", d)
            store.put_vector(
                "t", SPACE.fingerprint, TemporalRetriever.input_text(d), np.array([1.0, 0.1, 0, 0])
            )
        fake = ConflictAware(list(values), on_conflict_write=on_conflict_write)
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(JsonTransport(client, timeout=5, max_attempts=2))
            engine = UniversalEngine(
                retriever=TemporalRetriever(store, Fixed()),
                decision=p["decision"],
                writer=p["writer"],
                extractor=p["extractor"],
                arm=arm,
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
            return await engine.answer("t", q, ctx), fake

    return asyncio.run(go())


CONFLICT = [
    doc("a", "The budget of project P is 500.", "P-a"),
    doc("b", "The budget of project P is 700.", "P-b"),
]


def test_off_keeps_the_policy_path_and_only_observes_the_conflict_signal():
    result, fake = ask(CONFLICT, None)
    assert result["status"] != "FULL" and fake.conflict_writes == 0
    assert result.get("conflict_signal") is True
    assert "answer_kind" not in result and result["arm"] is None


def test_rewrite_reports_the_disagreement_with_each_source_and_is_not_an_answer():
    result, fake = ask(CONFLICT, REWRITE)
    assert fake.conflict_writes == 1  # at most one rewrite per stage that the floor blocked
    assert result["status"] == "PARTIAL" and result["reason"] == "conflicting_evidence"
    assert result["answer_kind"] == "source_disagreement_report"
    cited = {i for s in result["sentences"] for i in s["evidence_ids"]}
    assert cited == {"a@1#u0", "b@1#u0"}
    # U24 (0.1.0.dev27): an arm without release keys runs the policy's release block
    assert result["release"]["mode"] == "scoped_dependency"
    assert result["arm"]["name"] == "n4-rewrite"


def test_a_clean_answer_is_untouched_by_the_arm():
    clean = [doc("a", "The budget of project P is 500.", "P-a")]
    off, _ = ask(clean, None, values=("500", "999"))
    on, fake = ask(clean, REWRITE, values=("500", "999"))
    assert off["status"] == on["status"] == "FULL" and fake.conflict_writes == 0


def test_a_revocation_during_the_rewrite_is_never_released():
    store = TemporalStore(":memory:")
    result, fake = ask(
        CONFLICT, REWRITE, store=store, on_conflict_write=lambda: store.revoke("t", "b")
    )
    assert fake.conflict_writes >= 1
    assert result["status"] not in ("FULL", "PARTIAL")


def test_arms_refuse_unknown_values_and_older_arm_hashes_do_not_move():
    with pytest.raises(ValueError, match="unknown_conflict_partial"):
        Arm.of("bad", cascade={"conflict_partial": "always"})
    n2 = Arm.of("n2-keep-both", validity={"unknown_revision": "keep_both"})
    assert "cascade" not in n2.stamp()
    assert n2.digest == Arm("n2-keep-both", (("unknown_revision", "keep_both"),)).digest
