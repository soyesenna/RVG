"""PROD-2 (parent docs/PROD_2_PROTOCOL.md): the opt-in decision cache. Recording does not change
an answer; a question whose evidence window shares the top document or overlaps, and that the
judge calls the same question, reuses the recorded decision without a writer or extractor call
(judged again on another window); a cited unit no longer retrieved, a low equivalence or a
failed recheck falls back to the usual path; a hold stays a hold on the same window."""

import asyncio
import json
from typing import ClassVar

import httpx
import numpy as np
from test_handoff_packet import QUESTION, SPACE, Doubting, Fixed, doc, knowledge

from rvg.contracts import EmbeddingBatch
from rvg.runtime import CallContext
from rvg.temporal.models import TemporalQuestion
from rvg.temporal.store import TemporalRetriever
from rvg.testing.fake_openrouter import FakeOpenRouter, providers
from rvg.transport import JsonTransport
from rvg.universal import cache as cache_store
from rvg.universal.cache import CacheMode, DecisionCache
from rvg.universal.engine import UniversalEngine

PARAPHRASE = "How large is the budget of project P?"
SPACE_FP = Fixed().describe().fingerprint


class Override(FakeOpenRouter):
    """The fake with chosen boolean answers by question id (e.g. a low equivalence)."""

    overrides: ClassVar[dict] = {}

    def _decide(self, body, *, malformed=False):
        response = super()._decide(body, malformed=malformed)
        data = json.loads(response.content)
        for qid, value in self.overrides.items():
            if qid in data["answers"]:
                data["answers"][qid] = {"type": "noul", "noul": value}
        return httpx.Response(200, json=data)


class LowSame(Override):
    overrides: ClassVar[dict] = {"same": 0.2}


class LowAnswered(Override):
    overrides: ClassVar[dict] = {"answered": 0.2}


class Shifted(Fixed):
    """The paraphrase's vector leans away from the question's: the same top document (p1), other
    background units after it — another first-stage window."""

    async def embed(self, texts, *, purpose, context):
        q = np.eye(6)[0]
        vectors = [q + (0.6 * np.eye(6)[3] if t == PARAPHRASE else 0) for t in texts]
        return EmbeddingBatch(tuple(tuple(v) for v in vectors), SPACE, ("f",), ())


def ask(
    kb,
    store,
    *,
    make=FakeOpenRouter,
    text=QUESTION,
    record=False,
    lookup=True,
    embedding=None,
    defaults=False,
    **mode,
):
    """One answer with a decision cache; returns (result, chat completion calls, decision states).
    `defaults`: the mode as `CacheMode(store)` builds it, with no record or lookup argument."""

    async def go():
        fake = make()
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as client:
            p = providers(JsonTransport(client, timeout=5, max_attempts=2))
            engine = UniversalEngine(
                retriever=TemporalRetriever(kb, embedding or Fixed()),
                decision=p["decision"],
                writer=p["writer"],
                extractor=p["extractor"],
                **(
                    {
                        "decision_cache": CacheMode(store, **mode)
                        if defaults
                        else CacheMode(store, record=record, lookup=lookup, **mode)
                    }
                    if store is not None
                    else {}
                ),
            )
            q = TemporalQuestion(
                text=text, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
            )
            ctx = CallContext.with_timeout(
                60, max_decision_calls=400, max_generation_calls=24, max_http_attempts=800
            )
            result = await engine.answer("t", q, ctx)
            chats = sum("chat/completions" in url for url, _ in fake.bodies)
            stages = [
                "cache_equivalence"
                if "same" in (b.get("questions") or {})
                else "claims"
                if "c0" in (b.get("questions") or {})
                else "other"
                for url, b in fake.bodies
                if "decisions" in url
            ]
            return result, chats, stages

    return asyncio.run(go())


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def recorded(make=FakeOpenRouter, text=QUESTION):
    kb, store = knowledge(), DecisionCache()
    result, _, _ = ask(kb, store, make=make, text=text, record=True, lookup=False)
    return kb, store, result


def test_recording_does_not_change_the_answer_or_the_calls():
    plain, chats, stages = ask(knowledge(), None)
    kb, store = knowledge(), DecisionCache()
    again, chats2, stages2 = ask(kb, store, record=True, lookup=False)
    assert canonical(plain) == canonical(again) and (chats, stages) == (chats2, stages2)
    assert store.count() == 1
    (entry,) = store.scope("t", entry_policy(store), None)
    assert entry["decision"]["status"] == "FULL" and entry["draft"]["sentences"]
    assert len(entry["judgements"]) == len(entry["draft"]["sentences"])


def test_the_default_mode_records_and_reuses():
    """0.1.0: `CacheMode(DecisionCache(...))` with no other argument (the README's example)
    records every finished decision and looks it up: asked again, the question reuses it."""
    mode = CacheMode(DecisionCache())
    assert mode.record and mode.lookup
    kb, store = knowledge(), DecisionCache()
    first, chats, _ = ask(kb, store, defaults=True)
    assert first["status"] == "FULL" and chats > 0 and store.count() == 1
    again, chats2, _ = ask(kb, store, defaults=True)
    assert chats2 == 0 and again["short_answer"] == first["short_answer"]
    assert again["release"]["source"]["cache"]["entry"]
    assert store.count() == 1  # a cached answer is not recorded again


def entry_policy(store):
    return store.conn.execute("SELECT policy FROM entries").fetchone()[0]


def test_the_same_question_reuses_the_decision_without_a_writer_call():
    kb, store, original = recorded()
    result, chats, stages = ask(kb, store)
    assert chats == 0 and "cache_equivalence" not in stages
    assert result["status"] == "FULL" and result["short_answer"] == original["short_answer"]
    assert result["trace"]["cache"]["outcome"] == "released"
    source = result["release"]["source"]["cache"]
    assert source["p"] == 1.0 and source["theta"] == 0.9 and source["match"] == "top1_or_overlap"
    assert store.get(source["entry"]) is not None
    assert result["evidence_ids"] == original["evidence_ids"]
    # the release window is recorded by a trace stage (EVID-4 C5: window within the trace units)
    assert result["trace"]["cache"]["units"] == result["evidence_ids"]


def test_a_paraphrase_the_judge_calls_the_same_reuses_it():
    kb, store, original = recorded()
    result, chats, stages = ask(kb, store, text=PARAPHRASE)
    assert chats == 0 and "cache_equivalence" in stages
    assert result["status"] == "FULL" and result["short_answer"] == original["short_answer"]
    assert result["release"]["source"]["cache"]["p"] >= 0.9


def test_a_low_equivalence_is_a_miss_and_the_writer_answers():
    kb, store, _ = recorded()
    result, chats, _ = ask(kb, store, make=LowSame, text=PARAPHRASE)
    log = result["trace"]["cache"]
    assert (
        log["outcome"] == "no_equivalent" and log["candidates"] == 1 and log["judged"][0][1] < 0.9
    )
    assert chats > 0 and "source" not in result["release"]


def test_a_revoked_cited_document_is_never_reused():
    kb, store, _ = recorded()
    kb.revoke("t", "p1")
    result, chats, _ = ask(kb, store, text=PARAPHRASE)
    assert result["trace"]["cache"]["outcome"] == "no_candidate" and chats > 0
    assert "p1@1#u0" not in json.dumps(result.get("release") or {})


def test_a_cited_unit_outside_the_window_is_skipped_by_code():
    kb, store, _ = recorded()
    (entry,) = store.scope("t", entry_policy(store), None)
    forged = dict(entry, id="f" * 64)
    forged["draft"] = dict(entry["draft"])
    forged["draft"]["sentences"] = [{"text": "x", "evidence_ids": ["zz@1#u0"]}]
    store.put(forged)
    result, _, _ = ask(kb, store, text=PARAPHRASE)
    log = result["trace"]["cache"]
    assert log["skipped"]["cited_outside_window"] == 1 and log["entry"] == entry["id"]


def test_a_failed_recheck_falls_back_to_the_usual_path():
    kb, store, _ = recorded()
    result, chats, _ = ask(kb, store, make=LowAnswered, text=PARAPHRASE)
    log = result["trace"]["cache"]
    assert log["outcome"] == "recheck_failed" and log["recheck"]["status"] != "FULL"
    assert chats > 0


def test_a_held_decision_gives_the_same_hold():
    kb, store, original = recorded(make=Doubting)
    assert original["status"] == "ABSTAIN"
    result, chats, _ = ask(kb, store, text=PARAPHRASE)  # the fresh path would release here
    assert result["status"] == "ABSTAIN" and result["trace"]["cache"]["outcome"] == "held"
    assert chats == 0 and result.get("reason") == original.get("reason")


def test_holds_of_the_release_check_errors_and_cached_answers_are_not_entries():
    q = TemporalQuestion(
        text=QUESTION, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
    )
    base = {"trace": {"first": {"units": ["p1@1#u0"]}}, "evidence_ids": [], "stage": 1}
    for result in (
        {**base, "status": "ABSTAIN", "reason": "revoked_before_release"},
        {**base, "status": "ERROR", "reason": "stage_failed"},
        {**base, "status": "UNRESOLVED_SCOPE", "reason": "scope_unresolved"},
        {
            **base,
            "status": "ABSTAIN",
            "reason": "no_draft",
            "trace": {**base["trace"], "cache": {"outcome": "held"}},
        },
    ):
        assert cache_store.entry_of(result, [], q, "t") is None
    assert cache_store.entry_of({**base, "status": "ABSTAIN", "reason": "no_draft"}, [], q, "t")


def test_an_excluded_entry_is_not_returned():
    kb, store, _ = recorded()
    (entry,) = store.scope("t", entry_policy(store), None)
    result, chats, _ = ask(kb, store, exclude=frozenset({entry["id"]}))
    assert result["trace"]["cache"]["outcome"] == "no_candidate" and chats > 0


def test_same_snapshot_generation_drops_an_entry_of_another_knowledge_number():
    kb, store, _ = recorded()
    kb.upsert("t", doc("zz", "Background remark on trains."))
    kb.base.put_vector(
        "t",
        SPACE_FP,
        TemporalRetriever.input_text(doc("zz", "Background remark on trains.")),
        -np.eye(6)[0],
    )
    same_window, chats, _ = ask(kb, store)
    assert same_window["trace"]["cache"]["outcome"] == "released" and chats == 0
    stricter, _, _ = ask(kb, store, generation="same_snapshot")
    assert stricter["trace"]["cache"]["skipped"] == {"snapshot": 1}


def test_the_store_is_a_file_that_reopens_read_only(tmp_path):
    path = tmp_path / "cache.sqlite3"
    kb, store = knowledge(), DecisionCache(path)
    ask(kb, store, record=True, lookup=False)
    store.close()
    again = DecisionCache(path, readonly=True)
    assert again.count() == 1
    result, chats, _ = ask(kb, again)
    assert result["trace"]["cache"]["outcome"] == "released" and chats == 0


def test_a_source_is_kept_outside_the_entry_id():
    kb, a, b = knowledge(), DecisionCache(), DecisionCache()
    ask(kb, a, record=True, lookup=False)
    ask(kb, b, record=True, lookup=False, source={"request_id": "r1"})
    (plain,) = a.scope("t", entry_policy(a), None)
    (tagged,) = b.scope("t", entry_policy(b), None)
    assert plain["id"] == tagged["id"] and tagged["source"] == {"request_id": "r1"}
    assert "source" not in plain


def test_another_window_with_the_same_top_document_is_judged_again():
    kb, store, original = recorded()
    exact, chats, _ = ask(kb, store, text=PARAPHRASE, embedding=Shifted(), match="exact")
    assert exact["trace"]["cache"]["outcome"] == "no_candidate" and chats > 0
    result, chats, stages = ask(kb, store, text=PARAPHRASE, embedding=Shifted())
    log = result["trace"]["cache"]
    (_, p, keys) = log["judged"][0]
    assert keys["top1"] and not keys["same_first"] and not keys["same_window"]
    assert keys["jaccard"] < 1 and p >= 0.9
    assert log["outcome"] == "released" and chats == 0 and "claims" in stages
    assert result["short_answer"] == original["short_answer"]
    assert result["evidence_ids"] != original["evidence_ids"]


def test_the_same_question_among_other_candidates_is_chosen_without_a_judge_call():
    kb, store, original = recorded()
    ask(kb, store, text=PARAPHRASE, record=True, lookup=False)
    assert store.count() == 2
    result, chats, stages = ask(kb, store)
    log = result["trace"]["cache"]
    assert log["usable"] == 2 and log["candidates"] == 1 and "cache_equivalence" not in stages
    (_, p, keys) = log["judged"][0]
    assert p == 1.0 and keys["identical_text"] and chats == 0
    assert result["short_answer"] == original["short_answer"]


def test_an_entry_keeps_the_released_short_answer_beside_the_draft_judged_last():
    q = TemporalQuestion(
        text=QUESTION, as_of="2026-09-20T00:00:00Z", known_at="2026-09-20T00:00:00Z"
    )
    window = ["p1@1#u0"]
    rewritten = {
        "short_answer": "joined rewritten sentences",
        "sentences": [{"text": "joined rewritten sentences", "evidence_ids": window}],
    }
    result = {
        "status": "FULL",
        "stage": 1,
        "short_answer": "the writer's answer",
        "evidence_ids": window,
        "trace": {"first": {"units": window}},
    }
    stages = [{"stage": "stage1", "evidence_ids": window, "draft": rewritten, "judgements": [{}]}]
    entry = cache_store.entry_of(result, stages, q, "t")
    assert entry["short_answer"] == "the writer's answer" and entry["made_at"] == "stage1"
    assert entry["draft"]["short_answer"] == "joined rewritten sentences"
