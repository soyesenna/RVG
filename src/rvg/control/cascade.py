"""The answer cascade (Universal RVG P3): retrieve → scope → write → control → if not FULL, one
bounded step → the same control again.

Every transition rule and every limit comes from `rvg/resources/policies/universal.json`; nothing depends on the
knowledge base, the language or the request. The bounded step is one of
- `widen`: the same query, more units (the staged widening of experiments M/T/W);
- `gap`: up to N short queries the extractor proposes for what is still missing, their units
  added to the first evidence (S2G-style, with Qwen proposing and the floor judging);
- `rewrite`: the same evidence in reverse order, written once more (the control arm of H).
A second draft is judged exactly like a first one and replaces the first result only when its
status is strictly better (FULL over PARTIAL over ABSTAIN); the same draft is never re-judged."""

from __future__ import annotations

import json

from pydantic import Field

from ..contracts import Contract
from ..prompts import prompt
from ..providers.generation import GenerationOptions, GenerationRequest
from ..universal import universal
from ..universal.trace import node
from .grade import flat, normalized_equal, same_text
from .scope import anchor_of, resolve, restrict
from .verify import ANSWERABLE, graded, lean, self_contained
from .writer import write_answer

_U = universal()
RANK = {"ABSTAIN": 0, "PARTIAL": 1, "FULL": 2}
PROPOSE = prompt("control.cascade.PROPOSE")


class Queries(Contract):
    queries: tuple[str, ...] = Field(max_length=4)


def budgeted(units, *, byte_budget=None, min_units=None):
    """The same width rule as the evidence files: drop units from the tail until the JSON fits."""
    byte_budget = byte_budget or _U["retrieval"]["byte_budget"]
    min_units = min_units or _U["retrieval"]["min_units"]
    kept = list(units)
    size = lambda items: len(json.dumps(items, ensure_ascii=False).encode())
    while len(kept) > min_units and size(kept) > byte_budget:
        kept.pop()
    return kept


# U22: the containment threshold of the `same_text` judge, fixed before any result (θ of U12-b)
SAME_TEXT_THETA = 0.7


def same_short_answer(a, b, judge="identical_text", theta=SAME_TEXT_THETA):
    """The second-stage agreement of two short answers (U8; U22 adds the other judges)."""
    if judge == "identical_text":
        return bool(flat(a or "")) and flat(a or "") == flat(b or "")
    if judge == "normalized":
        return normalized_equal(a, b)
    if judge == "same_text":
        return same_text(a, b, theta=theta)
    raise ValueError("unknown_consistency_judge")


SCOPE_TOP1 = ("off", "entity-tie-v1", "dual-draft-v1")


def _flat_title(text):
    """U30: a title or a text for the anchor match — underscores as spaces, casefolded, one space."""
    return " ".join((text or "").replace("_", " ").casefold().split())


def _document_of(unit):
    return unit.get("document_id") or str(unit.get("id", "")).split("#")[0].split("@")[0]


def windowed(pool, k, window="none"):
    """The writing pool of the first k units. "none" is `budgeted(pool[:k])` (the policy's before
    0.1.0.dev17). "neighbors-v1" (N6-2, the policy's since 0.1.0.dev17) adds, to each of the k
    units, the unit just before and just after it in the same document revision among the units
    that passed the gate (`pool`), and gives the units grouped by document (in the order of their
    first writing unit) and in document order; neighbours are dropped first, last-added first,
    until the byte budget holds."""
    if window == "none":
        return budgeted(pool[:k])
    if window != "neighbors-v1":
        raise ValueError("unknown_evidence_window")
    seeds = list(pool[:k])
    by_doc = {}
    for unit in pool:
        by_doc.setdefault((unit.get("document_id"), unit.get("revision")), []).append(unit)
    for units in by_doc.values():
        units.sort(key=lambda u: (u.get("start") or 0, u["id"]))
    chosen = {u["id"] for u in seeds}
    extra = []
    for seed in seeds:
        doc = by_doc[(seed.get("document_id"), seed.get("revision"))]
        at = next(i for i, u in enumerate(doc) if u["id"] == seed["id"])
        for j in (at - 1, at + 1):
            if 0 <= j < len(doc) and doc[j]["id"] not in chosen:
                chosen.add(doc[j]["id"])
                extra.append(doc[j])
    order = []
    for seed in seeds:
        key = (seed.get("document_id"), seed.get("revision"))
        if key not in order:
            order.append(key)

    def arrange(units):
        ids = {u["id"] for u in units}
        return [u for key in order for u in by_doc[key] if u["id"] in ids]

    byte_budget = _U["retrieval"]["byte_budget"]
    size = lambda items: len(json.dumps(items, ensure_ascii=False).encode())
    while extra and size(arrange(seeds + extra)) > byte_budget:
        extra.pop()
    return arrange(seeds + extra) if extra else budgeted(seeds)


async def propose_queries(extractor, question_text, draft, missing, context, *, n=None):
    n = n or _U["cascade"]["gap_queries"]
    result = await extractor.generate_structured(
        GenerationRequest(
            role="semantic",
            system=PROPOSE.format(n=n),
            payload={
                "question": question_text,
                "missing": list(missing or []),
                "draft": draft.model_dump(mode="json") if draft is not None else None,
            },
            response_type=Queries,
            schema_name="gap_queries_v1",
            prompt_version="gap-queries-1",
            options=GenerationOptions(max_tokens=300),
        ),
        context,
    )
    seen, out = {question_text.strip().casefold()}, []
    for query in result.value.queries:
        key = query.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            out.append(query.strip())
    return out[:n]


def abstain(reason):
    return {
        "status": "ABSTAIN",
        "text": "",
        "sentences": [],
        "missing": [],
        "action": None,
        "reason": reason,
    }


async def answerable(asker, text, evidence):
    """The pre-writing grade of experiments J/K (U4-a): how well the evidence holds what the
    question asks for, 0..1. One Jev call; None when the judgement failed."""
    return await graded(
        asker,
        "control_instance_answerable",
        {"question": text, "evidence": lean(evidence)},
        ANSWERABLE,
    )


def handoff_stage(handoff, label, evidence):
    """PROD-1: one stage of the handoff trail (only when the packet is on) — the writing units as
    the writer saw them; the writer and the controller fill in the rest."""
    if handoff is None:
        return None
    entry = {"stage": label, "evidence_ids": [u["id"] for u in evidence], "units": list(evidence)}
    handoff.append(entry)
    return entry


def settle(entry, result):
    if entry is not None:
        entry["status"], entry["reason"] = result["status"], result.get("reason")
    return result


async def write_and_control(
    question,
    evidence,
    writer,
    controller,
    context,
    *,
    gate=None,
    label="stage",
    conflict_partial="off",
    writer_schema="W0",
    writer_contract="W0",
    handoff=None,
):
    """`handoff` (PROD-1) is a list only when the handoff packet is on: this stage's trail is
    appended to it. The calls, the trace and the result are those without it."""
    entry = handoff_stage(handoff, label, evidence)
    with node(label, [u["id"] for u in evidence]) as n:
        if not evidence:
            return None, settle(entry, abstain("no_active_evidence"))
        if gate is not None:
            asker, text, threshold = gate
            score = await answerable(asker, text, evidence)
            if score is not None and score < threshold:
                return None, settle(
                    entry, abstain("evidence_not_answerable") | {"answerable": score}
                )
        draft = await write_answer(
            writer,
            question,
            evidence,
            context,
            schema=writer_schema,
            contract=writer_contract,
            **({"trail": entry} if entry is not None else {}),
        )
        if draft is None:
            return None, settle(entry, abstain("no_draft"))
        result = await controller.control(
            question, evidence, draft, context, **({"trail": entry} if entry is not None else {})
        )
        if (
            conflict_partial == "rewrite"
            and result["status"] != "FULL"
            and result.get("conflict_signal")
        ):
            # N4 (arm only): the floor blocked the draft for conflict alone. Write it once in
            # the conflict form and judge it on its cited units; keep it only if it is better.
            settle(entry, result)
            rewrite = handoff_stage(handoff, "conflict_rewrite", evidence)
            with node("conflict_rewrite", [u["id"] for u in evidence], attempt_kind="rewrite"):
                again_draft = await write_answer(
                    writer,
                    question,
                    evidence,
                    context,
                    conflict=True,
                    schema=writer_schema,
                    contract=writer_contract,
                    **({"trail": rewrite} if rewrite is not None else {}),
                )
            if again_draft is None:
                settle(rewrite, abstain("no_draft"))
            if again_draft is not None:
                again = await controller.conflict_check(question, evidence, again_draft, context)
                if rewrite is not None:
                    rewrite["draft"] = again_draft.model_dump(mode="json")
                    rewrite["control"] = dict(again.get("assessment") or {})
                    settle(rewrite, again)
                if RANK.get(again["status"], -1) > RANK.get(result["status"], -1):
                    n.output({"status": again["status"], "reason": again.get("reason")})
                    return again_draft, {**again, "conflict_rewrite_of": result.get("reason")}
            result = {
                **result,
                "conflict_rewrite": "declined" if again_draft is None else "not_better",
            }
        n.output({"status": result["status"], "reason": result.get("reason")})
        if entry is not None and "status" not in entry:
            settle(entry, result)
        return draft, result


async def run(
    question,
    *,
    retrieve,
    writer,
    controller,
    asker,
    context,
    extractor=None,
    step=None,
    resolve_scope=True,
    unresolved_status=None,
    answerable_gate=None,
    consistency=None,
    conflict_partial="off",
    writer_schema="W0",
    writer_contract="W0",
    evidence_window=None,
    consistency_partner=None,
    consistency_judge=None,
    write_top=None,
    scope_top1="off",
    handoff=None,
    cache=None,
):
    """`retrieve(text, k)` returns ranked units ({id, title, text}); everything else is fixed.
    `scope_top1` (U30 arm, parent docs/UNIVERSAL_U30_PROTOCOL.md §12) acts only when the scope is
    RESOLVED and the top-ranked unit's title is not the anchor (the top document left the pool):
    "entity-tie-v1" writes from the top unit's title when the anchor title occurs in that document's
    retrieved units; "dual-draft-v1" also writes and judges a first draft from the top unit's title
    and keeps it when only it is released, withholds when both are released with different short
    answers, and otherwise goes on as the policy. "off" is the policy.

    `handoff` (PROD-1) is a list only when the handoff packet is on: each written stage appends its
    trail (`write_and_control`), and the partner draft its own with the comparison; nothing else
    changes.

    `cache` (PROD-2, only when the decision cache looks up) is `cache(first, second_of)` returning
    the lookup and its mode (`rvg.universal.cache.find`). A released entry's draft is judged again
    for this question (recorded claim judgements reused on the same window, no recovery) and
    returned when it is released again, else the question goes on as without the cache; a held
    entry gives the same hold. No writer or extractor call is made on that path.

    Returns the controller's result plus `stage`, `scope`, `evidence_ids` and a `trace`."""
    r, c = _U["retrieval"], _U["cascade"]
    if write_top is not None:  # U23 (arm only): the first stage's writing units
        r = {**r, "write_top": write_top}
    step = step or c["policy"]
    text = question.text if hasattr(question, "text") else question["text"]
    ranked = list(await retrieve(text, r["candidates"]))
    trace = {"candidates": len(ranked)}
    scope = {"status": "OPEN", "anchor": None, "candidates": []}
    if unresolved_status is None:
        unresolved_status = _U["scope"]["unresolved_status"]
    if evidence_window is None:  # N6-2: the policy's window unless an arm chose one
        evidence_window = c.get("evidence_window", "none")
    if resolve_scope:
        contained = await self_contained(asker, text)
        scope = resolve(ranked, contained) | {"self_contained": contained}
        # U3: diverting spread, unnamed-target questions lost more correct answers than it
        # avoided wrong ones, so by policy the resolution is recorded and the answer proceeds.
        if scope["status"] == "UNRESOLVED" and unresolved_status:
            return {
                **abstain("scope_unresolved"),
                "status": "UNRESOLVED_SCOPE",
                "stage": 0,
                "scope": scope,
                "evidence_ids": [],
                "trace": trace,
            }
    pool = restrict(ranked, scope["anchor"]) if scope["status"] == "RESOLVED" else ranked
    top_pool = None
    if scope_top1 != "off":
        if scope_top1 not in SCOPE_TOP1:
            raise ValueError("unknown_scope_top1")
        top = anchor_of(ranked[0]) if ranked else None
        if scope["status"] == "RESOLVED" and top and top != scope["anchor"]:
            top_pool = restrict(ranked, top)
            log = trace["scope_top1"] = {"mode": scope_top1, "anchor": scope["anchor"], "top1": top}
            if scope_top1 == "entity-tie-v1":
                body = " ".join(
                    u.get("text", "") for u in ranked if _document_of(u) == _document_of(ranked[0])
                )
                log["anchor_in_top1"] = _flat_title(scope["anchor"]) in _flat_title(body)
                log["chosen"] = "top1" if log["anchor_in_top1"] else "anchor"
                if log["anchor_in_top1"]:
                    pool = top_pool
    first = windowed(pool, r["write_top"], evidence_window)
    if cache is not None:
        lookup, mode = await cache(first, lambda: windowed(pool, r["widen_top"], evidence_window))
        log = trace["cache"] = lookup.trace(mode)
        entry = lookup.entry
        if entry is not None and entry["decision"]["status"] in ("FULL", "PARTIAL"):
            with node("cache_recheck", [u["id"] for u in lookup.evidence]):
                again = await controller.control(
                    question,
                    lookup.evidence,
                    entry["draft"],
                    context,
                    slots=entry.get("slots") or [],
                    judgements=entry["judgements"] if lookup.reuse_labels else None,
                    recover=False,
                )
            if again["status"] in ("FULL", "PARTIAL"):
                # the units the reuse was judged on: the stage record of a cached release (the
                # usual stage records `first`/`second` are never written on this path)
                units = [u["id"] for u in lookup.evidence]
                log.update(outcome="released", status=again["status"], units=units)
                return {
                    **again,
                    "short_answer": (
                        entry["short_answer"]
                        if entry.get("short_answer") is not None
                        else entry["draft"]["short_answer"]
                    ),
                    "stage": entry["stage"],
                    "scope": scope,
                    "evidence_ids": [u["id"] for u in lookup.evidence],
                    "trace": trace,
                }
            log.update(
                outcome="recheck_failed",
                recheck={"status": again["status"], "reason": again.get("reason")},
            )
        elif entry is not None:
            log.update(outcome="held")
            return {
                **abstain(entry["decision"]["reason"]),
                "stage": entry["stage"],
                "scope": scope,
                "evidence_ids": list(entry["decision"]["evidence_ids"]),
                "trace": trace,
            }
    # U4-a: the pre-writing answerable gate cut wrong and right releases alike on the holdout
    # (nogold +73 vs +93, correct -20), so the policy leaves it off; it stays an option.
    gate_cfg = answerable_gate if answerable_gate is not None else (c.get("answerable_gate") or {})
    gate_at = lambda stage: (
        (asker, text, gate_cfg["threshold"]) if stage in gate_cfg.get("stages", []) else None
    )
    draft, result = await write_and_control(
        question,
        first,
        writer,
        controller,
        context,
        gate=gate_at("first"),
        label="stage1",
        conflict_partial=conflict_partial,
        writer_schema=writer_schema,
        writer_contract=writer_contract,
        **({"handoff": handoff} if handoff is not None else {}),
    )
    trace["first"] = {
        "units": [u["id"] for u in first],
        "status": result["status"],
        "reason": result.get("reason"),
    }
    best = {
        **result,
        "short_answer": draft.short_answer if draft else None,
        "stage": 1,
        "scope": scope,
        "evidence_ids": [u["id"] for u in first],
    }
    if scope_top1 == "dual-draft-v1" and top_pool is not None:
        # U30 S3: the same first stage on the top unit's title
        top_first = windowed(top_pool, r["write_top"], evidence_window)
        with node("scope_top1_draft", [u["id"] for u in top_first], attempt_kind="scope_top1"):
            top_draft, top_result = await write_and_control(
                question,
                top_first,
                writer,
                controller,
                context,
                gate=gate_at("first"),
                label="stage1_top1",
                conflict_partial=conflict_partial,
                writer_schema=writer_schema,
                writer_contract=writer_contract,
                **({"handoff": handoff} if handoff is not None else {}),
            )
        log = trace["scope_top1"]
        log.update(
            units=[u["id"] for u in top_first],
            top1_status=top_result["status"],
            top1_reason=top_result.get("reason"),
            anchor_status=result["status"],
        )
        released = ("FULL", "PARTIAL")
        if top_result["status"] in released and result["status"] not in released:
            log["chosen"] = "top1"
            return {
                **top_result,
                "short_answer": top_draft.short_answer if top_draft else None,
                "stage": 1,
                "scope": scope,
                "evidence_ids": [u["id"] for u in top_first],
                "trace": trace,
            }
        if top_result["status"] in released and result["status"] in released:
            rule = consistency if consistency is not None else (c.get("consistency") or {})
            judge = consistency_judge or rule.get("judge", "identical_text")
            same = same_short_answer(
                best["short_answer"],
                top_draft.short_answer if top_draft else None,
                "identical_text" if judge == "off" else judge,
                rule.get("theta", SAME_TEXT_THETA),
            )
            log["same_answer"] = same
            if not same:
                log["chosen"] = "disagree"
                return {
                    **abstain("scope_top1_disagree"),
                    "stage": 1,
                    "scope": scope,
                    "evidence_ids": [],
                    "trace": trace,
                }
        log["chosen"] = "anchor"
    if result["status"] == "FULL" or c["widen"] < 1 or step == "none":
        return best | {"trace": trace}
    if step == "widen":
        second = windowed(pool, r["widen_top"], evidence_window)
    elif step == "rewrite":
        second = first[::-1]
    elif step == "gap":
        if extractor is None:
            return best | {"trace": trace}
        queries = await propose_queries(extractor, text, draft, result.get("missing"), context)
        trace["queries"] = queries
        seen, second = {u["id"] for u in first}, list(first)
        for query in queries:
            for unit in await retrieve(query, r["write_top"]):
                if unit["id"] not in seen:
                    seen.add(unit["id"])
                    second.append(unit)
        second = budgeted(second)
    else:
        raise ValueError("unknown_cascade_step")
    if step != "rewrite" and [u["id"] for u in second] == [u["id"] for u in first]:
        trace["second"] = {"skipped": "no_new_evidence"}
        return best | {"trace": trace}
    second_draft, again = await write_and_control(
        question,
        second,
        writer,
        controller,
        context,
        gate=gate_at("second"),
        label="stage2",
        conflict_partial=conflict_partial,
        writer_schema=writer_schema,
        writer_contract=writer_contract,
        **({"handoff": handoff} if handoff is not None else {}),
    )
    trace["second"] = {
        "units": [u["id"] for u in second],
        "status": again["status"],
        "reason": again.get("reason"),
    }
    # U8: a second-stage release must agree with an independent second draft written on the
    # first evidence in reverse order (identical short answers after normalisation). Right
    # answers agree far more often than answers made where the knowledge base has none.
    # U22 (arm only): the partner may be written on the second-stage evidence in reverse, the two
    # short answers may be compared after `normalized` or by `same_text`, or the rule turned off.
    rule = consistency if consistency is not None else (c.get("consistency") or {})
    # the rule's own values unless an arm chose (U22: the policy's are the widened partner and the
    # same-answer judge since 0.1.0.dev18; a rule without them is U8's first evidence, identical text)
    if consistency_partner is None:
        consistency_partner = "widened" if rule.get("partner") == "widened" else "first"
    if consistency_judge is None:
        consistency_judge = rule.get("judge", "identical_text")
    theta = rule.get("theta", SAME_TEXT_THETA)
    if consistency_judge == "off":
        rule = {}
    if rule and step == "widen" and RANK.get(again["status"], -1) > RANK.get(result["status"], -1):
        pool_of_partner = (second if consistency_partner == "widened" else first)[::-1]
        mate = handoff_stage(handoff, "partner_draft", pool_of_partner)
        with node(
            "partner_draft", [u["id"] for u in pool_of_partner], attempt_kind="partner_draft"
        ):
            partner = await write_answer(
                writer,
                question,
                pool_of_partner,
                context,
                schema=writer_schema,
                contract=writer_contract,
                **({"trail": mate} if mate is not None else {}),
            )
        agreed = bool(
            second_draft is not None
            and partner is not None
            and same_short_answer(
                second_draft.short_answer, partner.short_answer, consistency_judge, theta
            )
        )
        if mate is not None:
            mate["draft"] = partner.model_dump(mode="json") if partner else None
            mate["status"], mate["reason"] = None, None if partner else "no_draft"
            mate["comparison"] = {
                "judge": consistency_judge,
                "theta": theta,
                "partner": consistency_partner,
                "agreed": agreed,
            }
        trace["consistency"] = {
            "partner_short_answer": partner.short_answer if partner else None,
            "agreed": agreed,
            **(
                {"partner": consistency_partner, "judge": consistency_judge}
                if (consistency_partner, consistency_judge) != ("first", "identical_text")
                else {}
            ),
        }
        if not agreed:
            return best | {"trace": trace}
    if RANK.get(again["status"], -1) > RANK.get(result["status"], -1):
        best = {
            **again,
            "short_answer": second_draft.short_answer if second_draft else None,
            "stage": 2,
            "scope": scope,
            "evidence_ids": [u["id"] for u in second],
        }
    return best | {"trace": trace}
