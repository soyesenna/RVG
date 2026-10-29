"""The decision cache (PROD-2, opt-in, parent docs/PROD_2_PROTOCOL.md §2 and §12-1): a decision
the controller already made — released or held — is reused for a later question whose evidence
window shares its top-ranked document or overlaps it (or is the same window, as chosen) and
that the judge (Jev) calls the same question.

A reuse never calls the writer or the extractor. A released entry's draft is judged again for
the new question by the same controller (the recorded claim judgements are reused when the
evidence window is the same; `answered`, citations, qualifiers, relations and the question kind
are asked again; no recovery), and then goes through the same release check as any draft. A
held entry gives the same hold. The deterministic gate is code: the policy hash, the arm and the
tenant match, the evidence window is the one the current retrieval and validity gate produced, and
every unit the draft cites is in it — so a revoked, hidden, excluded or no longer valid unit
cannot be reused. The equivalence and recheck calls are ordinary judge calls: they are traced,
taped and replayed like the rest. Nothing here runs when the cache is off."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field

from ..errors import ProviderError
from ..prompts import prompt
from ..storage.decision_cache import DecisionCache, canonical
from .config import universal_hash
from .trace import node

VERSION = "decision-cache-v1"
EQUIVALENCE = prompt("universal.cache.EQUIVALENCE")
RELEASED = ("FULL", "PARTIAL")
MAX_CANDIDATES = 5
# The candidate key (PROD-2 §12-1): the same first-stage window (`exact`), the same top-ranked
# document (`top1`), a window overlap (`overlap`, Jaccard >= tau) or either of the last two.
MATCHES = ("exact", "top1", "overlap", "top1_or_overlap")
GENERATIONS = ("same_window", "same_snapshot")
# A hold the release check made about the knowledge base changing under the answer, not about the
# question: never a cache entry.
RELEASE_TIME = {
    "revoked_before_release",
    "permission_changed_before_release",
    "source_changed",
    "scope_changed",
    "access_changed",
    "stale_after_retry",
    "retry_budget_exhausted",
}


def normalized(text):
    return " ".join((text or "").casefold().split())


@dataclass(frozen=True)
class CacheMode:
    """How an engine uses a decision cache. By default it does both: every finished decision is
    recorded (`record`) and later questions look it up (`lookup`). An evaluation passes
    `record=False` for a lookup-only cache (or `lookup=False` to fill one). `exclude` (evaluation
    only) names entry ids a lookup must not return, e.g. a question's own entry when it is asked
    as a negative query; `source` (evaluation only) is stored with a recorded entry, outside its
    id."""

    store: DecisionCache
    theta: float = 0.9
    match: str = "top1_or_overlap"
    tau: float | None = 0.5
    record: bool = True
    lookup: bool = True
    generation: str = "same_window"
    exclude: frozenset = field(default_factory=frozenset)
    source: dict | None = None
    max_candidates: int = MAX_CANDIDATES

    def __post_init__(self):
        if self.match not in MATCHES:
            raise ValueError("unknown_cache_match")
        if self.match in ("overlap", "top1_or_overlap") and not self.tau:
            raise ValueError("cache_overlap_needs_tau")
        if self.generation not in GENERATIONS:
            raise ValueError("unknown_cache_generation")

    def describe(self):
        return {
            "version": VERSION,
            "theta": self.theta,
            "match": self.match,
            "tau": self.tau,
            "generation": self.generation,
            "max_candidates": self.max_candidates,
        }


def entry_of(result, stages, question, tenant, *, arm=None, snapshot=None, scope=None, source=None):
    """The cache entry of one finished answer, or None when it is not a decision about the
    question (an error, an unresolved scope, a hold the release check made, a cached answer)."""
    status, reason = result.get("status"), result.get("reason")
    trace = result.get("trace") or {}
    if trace.get("cache") and trace["cache"].get("outcome") in ("released", "held"):
        return None
    if status not in (*RELEASED, "ABSTAIN") or (status == "ABSTAIN" and reason in RELEASE_TIME):
        return None
    first_ids = list((trace.get("first") or {}).get("units") or [])
    if not first_ids:
        return None
    stage = result.get("stage") or 1
    body = {
        "version": VERSION,
        "tenant": tenant,
        "question": {
            "text": question.text,
            "as_of": question.as_of.isoformat(),
            "known_at": question.known_at.isoformat(),
            "locale": question.locale,
        },
        "first_ids": first_ids,
        "stage": stage,
        "decision": {
            "status": status,
            "reason": reason,
            "evidence_ids": list(result.get("evidence_ids") or []),
        },
        "policy": universal_hash(),
        "arm": arm,
        "snapshot": snapshot,
        "scope": scope or {},
    }
    if status in RELEASED:
        window = list(result.get("evidence_ids") or [])
        made = next(
            (
                s
                for s in stages or []
                if s.get("stage") in ("stage1", "stage1_top1", "stage2")
                and list(s.get("evidence_ids") or []) == window
                and s.get("draft")
                and s.get("judgements") is not None
            ),
            None,
        )
        if made is None:
            return None  # e.g. the conflict rewrite of an arm: no controller record to reuse
        draft = made["draft"]
        body.update(
            window_ids=window,
            made_at=made["stage"],
            # The released short answer is the writer's even when recovery rewrote the sentences
            # (the rewritten draft, judged last, carries its own): a reuse returns the released one.
            short_answer=result.get("short_answer"),
            draft={
                "answerable": True,
                "short_answer": draft.get("short_answer"),
                "sentences": [
                    {"text": s.get("text"), "evidence_ids": list(s.get("evidence_ids") or [])}
                    for s in draft.get("sentences") or []
                ],
            },
            judgements=list(made["judgements"]),
            slots=made.get("slots"),
            partner=trace.get("consistency") if stage == 2 else None,
        )
    else:
        body.update(window_ids=sorted(first_ids), draft=None, judgements=None, slots=None)
    body["id"] = hashlib.sha256(canonical(body).encode()).hexdigest()
    if source is not None:
        body["source"] = dict(source)
    return body


@dataclass
class Lookup:
    """What one lookup saw: the candidates, the judge's probabilities, the chosen entry."""

    pool: int = 0
    usable: int = 0
    candidates: int = 0
    judged: list = field(default_factory=list)
    entry: dict | None = None
    p: float | None = None
    evidence: list | None = None
    reuse_labels: bool = False
    skipped: dict = field(default_factory=dict)
    error: str | None = None

    def trace(self, mode):
        out = {
            **mode.describe(),
            "pool": self.pool,
            "usable": self.usable,
            "candidates": self.candidates,
            "judged": [[entry_id, p, keys] for entry_id, p, keys in self.judged],
            "outcome": "no_candidate" if not self.candidates else "no_equivalent",
        }
        if self.skipped:
            out["skipped"] = dict(self.skipped)
        if self.error:
            out["error"] = self.error
        if self.entry is not None:
            out.update(entry=self.entry["id"], p=self.p, stage=self.entry["stage"])
        return out


async def equivalent(asker, cached, asked):
    """P(the two questions ask the same thing), the judge's probability; None when it failed."""
    answers, _ = await asker.ask(
        "cache_equivalence",
        {"first_question": cached, "second_question": asked},
        {"same": {"type": "boolean", "instructions": EQUIVALENCE}},
    )
    value = (answers.get("same") or {}).get("probability_true")
    return float(value) if value is not None else None


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def _document(unit_id):
    return str(unit_id).split("#")[0]


def keys_of(entry, first_ids):
    """How an entry's first-stage window relates to a question's: the same window, the same
    top-ranked document (the first unit of a window belongs to it), the Jaccard overlap."""
    theirs = entry["first_ids"]
    return {
        "same_first": sorted(theirs) == sorted(first_ids),
        "top1": bool(theirs and first_ids) and _document(theirs[0]) == _document(first_ids[0]),
        "jaccard": round(_jaccard(theirs, first_ids), 4),
    }


def qualifies(mode, keys):
    if mode.match == "exact":
        return keys["same_first"]
    if mode.match == "top1":
        return keys["top1"]
    if mode.match == "overlap":
        return keys["jaccard"] >= mode.tau
    return keys["top1"] or keys["jaccard"] >= mode.tau


async def find(mode, *, asker, question, first, second_of, tenant, arm=None, snapshot=None):
    """The candidates of `question` (its first-stage window `first`; `second_of()` gives the
    widened window a stage-2 entry needs), the judge's equivalence for each, and the entry to
    reuse when one reaches `mode.theta`. A released entry is judged again on the evidence the
    question retrieved now (its recorded claim judgements only when that window is the same
    window); a held entry is reused only on the same window (a hold depends on its evidence)."""
    found = Lookup()
    policy = universal_hash()
    first_ids = [u["id"] for u in first]
    if mode.match == "exact":
        entries = mode.store.by_first(tenant, policy, arm, first_ids)
    else:
        entries = mode.store.scope(tenant, policy, arm)
    pool = []
    for entry in entries:
        if entry["id"] in mode.exclude:
            continue
        keys = keys_of(entry, first_ids)
        if qualifies(mode, keys):
            pool.append((entry, keys))
    found.pool = len(pool)
    text = question.text
    usable, second = [], None
    for entry, keys in pool:
        if mode.generation == "same_snapshot" and entry.get("snapshot") != snapshot:
            found.skipped["snapshot"] = found.skipped.get("snapshot", 0) + 1
            continue
        released = entry["decision"]["status"] in RELEASED
        if not released and not keys["same_first"]:
            found.skipped["held_needs_same_window"] = (
                found.skipped.get("held_needs_same_window", 0) + 1
            )
            continue
        if entry["stage"] == 2 and released:
            if second is None:
                second = list(second_of())
            evidence = second
        else:
            evidence = list(first)
        ids = [u["id"] for u in evidence]
        cited = {
            i for s in (entry.get("draft") or {}).get("sentences") or [] for i in s["evidence_ids"]
        }
        if not cited <= set(ids):  # a cited unit is no longer retrieved, valid or visible
            found.skipped["cited_outside_window"] = found.skipped.get("cited_outside_window", 0) + 1
            continue
        same_window = sorted(ids) == sorted(entry["window_ids"])
        if mode.match == "exact" and released and not same_window:
            found.skipped["window"] = found.skipped.get("window", 0) + 1
            continue
        identical = normalized(entry["question"]["text"]) == normalized(text)
        keys = {**keys, "same_window": same_window, "identical_text": identical}
        usable.append((entry, evidence, same_window, keys))
    usable.sort(
        key=lambda u: (not u[3]["same_first"], -u[3]["jaccard"], not u[3]["top1"], u[0]["id"])
    )
    found.usable = len(usable)
    same = next((u for u in usable if u[3]["identical_text"]), None)
    if same is not None:  # the same question (§2.3 3): p = 1, no judge call, no other candidate
        entry, evidence, same_window, keys = same
        found.candidates = 1
        found.judged = [(entry["id"], 1.0, keys)]
        found.entry, found.p, found.evidence, found.reuse_labels = entry, 1.0, evidence, same_window
        return found
    usable = usable[: mode.max_candidates]
    found.candidates = len(usable)
    if not usable:
        return found

    async def judge(entry):
        return await equivalent(asker, entry["question"]["text"], text)

    try:
        with node("cache_lookup", [u[0]["id"] for u in usable]):
            scores = await asyncio.gather(*(judge(u[0]) for u in usable))
    except ProviderError as exc:  # a failed lookup is a miss, never an error of the answer
        found.error = exc.code
        return found
    found.judged = [(u[0]["id"], p, u[3]) for u, p in zip(usable, scores, strict=True)]
    best = None
    for u, p in zip(usable, scores, strict=True):
        if p is not None and p >= mode.theta and (best is None or p > best[1]):
            best = (u, p)
    if best is not None:
        (entry, evidence, same_window, _), p = best
        found.entry, found.p, found.evidence, found.reuse_labels = entry, p, evidence, same_window
    return found
