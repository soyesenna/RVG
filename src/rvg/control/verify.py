from __future__ import annotations

import asyncio
import re

from ..errors import ContractError, ProviderError
from ..gateway import DecisionGateway
from ..prompts import prompt
from ..temporal.engine import CLAIM_CRITERIA, DRAFT_CHECKS, claim_choice

FORMS = ("choice", "noul")
CLAIM_SCOPES = ("full", "no_query", "lean", "isolated")
LEAN_FIELDS = ("id", "title", "origin", "text")
# Experiment R: a unit may carry `origin` instead of `title` (`source` is taken: temporal units use
# it for something else); then everyone who reads it is told what that field is. Units without the field leave every instruction exactly as it was.
SOURCE_NOTE = prompt("control.verify.SOURCE_NOTE")


def sourced(evidence):
    return SOURCE_NOTE if any("origin" in e for e in evidence) else ""


SUPPORT = prompt("control.verify.SUPPORT")
REFUTE = prompt("control.verify.REFUTE")
TIMED = prompt("control.verify.TIMED")
# Lean states carry no validity metadata: upstream code already kept only evidence in force.
FILTERED = prompt("control.verify.FILTERED")
CHOICE_FILTERED = prompt("control.verify.CHOICE_FILTERED")


def lean(evidence):
    return [{k: e[k] for k in LEAN_FIELDS if k in e} for e in evidence]


def as_query(question):
    return question.model_dump(mode="json") if hasattr(question, "model_dump") else dict(question)


def claim_state(scope, question, evidence, claims, draft=None):
    query = as_query(question)
    if scope == "full":
        return {"query": query, "evidence": evidence, "claims": claims, "draft": draft}
    if scope == "no_query":
        timing = {k: query[k] for k in ("as_of", "known_at") if k in query}
        return {"query": timing, "evidence": evidence, "claims": claims}
    if scope == "lean":
        return {"evidence": lean(evidence), "claims": claims}
    raise ValueError("unknown_claim_scope")


def claim_questions(form, indexes, *, refute_only=False, filtered=False, note=""):
    questions = _claim_questions(form, indexes, refute_only=refute_only, filtered=filtered)
    return {k: {**q, "instructions": q["instructions"] + note} for k, q in questions.items()}


def _claim_questions(form, indexes, *, refute_only=False, filtered=False):
    if form not in FORMS:
        raise ValueError("unknown_judgement_form")
    scope = FILTERED if filtered else TIMED
    questions = {}
    for i in indexes:
        if form == "choice" and not refute_only:
            questions[f"c{i}"] = (
                {
                    "type": "choice",
                    "instructions": CHOICE_FILTERED.format(i=i),
                    "criteria": dict(CLAIM_CRITERIA),
                }
                if filtered
                else claim_choice(i)
            )
            continue
        if not refute_only:
            questions[f"s{i}"] = {
                "type": "boolean",
                "instructions": SUPPORT.format(i=i, scope=scope),
            }
        questions[f"r{i}"] = {"type": "boolean", "instructions": REFUTE.format(i=i, scope=scope)}
    return questions


class Asker:
    """Gateway access with the pre-registered contract-error policy and an audit trail."""

    def __init__(self, decision, context, *, split_on_contract_error=True):
        self.decision, self.context = decision, context
        self.split = split_on_contract_error
        self.gateway = DecisionGateway(decision, context)
        self.extra_gateways = []
        self.events = []

    @property
    def trace(self):
        return self.gateway.trace + [e for g in self.extra_gateways for e in g.trace]

    async def ask(self, stage, state, questions, *, fresh=False):
        """Returns (answers, errors); a failed question is an error, never a fabricated value."""
        gateway = self.gateway
        if fresh:  # bypass the run cache for a genuinely repeated observation
            gateway = DecisionGateway(self.decision, self.context)
            self.extra_gateways.append(gateway)
        try:
            return await gateway.ask(stage, state, questions), {}
        except ContractError as exc:
            if exc.code != "probability_sum_mismatch":
                raise
            event = {"stage": stage, "ids": sorted(questions), "detail": exc.detail}
            self.events.append(event)
            if not self.split:
                return {}, {q: exc.code for q in questions}
        answers, errors = {}, {}
        for qid, spec in questions.items():
            # One isolated re-request per question, applied to every affected request alike.
            retry = DecisionGateway(self.decision, self.context)
            self.extra_gateways.append(retry)
            try:
                value = await retry.ask(stage + ":isolated", state, {qid: spec})
                # Marked so an evaluation can also report the outcome without this policy.
                answers[qid] = {**value[qid], "split_recovered": True}
            except ProviderError as exc:
                errors[qid] = exc.code
        event.update(recovered=sorted(answers), failed=sorted(errors))
        return answers, errors


class PolicyGateway:
    """Gateway-shaped view of an Asker, so upstream gates get the same contract-error policy."""

    def __init__(self, asker):
        self.asker = asker

    @property
    def trace(self):
        return self.asker.trace

    async def ask(self, stage, state, questions):
        answers, errors = await self.asker.ask(stage, state, questions)
        if errors:
            raise ContractError(next(iter(errors.values())))
        return answers


def judgement(form, answers, errors, i, sweep=None):
    keys = [f"c{i}"] if form == "choice" else [f"s{i}", f"r{i}"]
    failed = [errors[k] for k in keys if k in errors]
    if failed or any(k not in answers for k in keys):
        return {"form": form, "error": failed[0] if failed else "missing_answer"}
    if form == "choice":
        value = {"form": form, "probabilities": answers[f"c{i}"]["probabilities"]}
    else:
        value = {
            "form": form,
            "s": answers[f"s{i}"]["probability_true"],
            "r": answers[f"r{i}"]["probability_true"],
        }
    if sweep is not None:
        value["sweep_r"] = sweep
    if any(answers[k].get("split_recovered") for k in keys):
        value["split_recovered"] = True
    return value


def support_score(value):
    if "error" in value:
        return 0.0
    return value["probabilities"]["SUPPORTED"] if value["form"] == "choice" else value["s"]


def refute_score(value):
    if "error" in value:
        return 0.0
    own = (
        value["probabilities"].get("CONTRADICTED", 0.0) if value["form"] == "choice" else value["r"]
    )
    return max(own, value.get("sweep_r") or 0.0)


def label(value, threshold=0.85, margin=0.15, conflict_floor=0.5):
    if value.get("no_evidence"):
        return "UNKNOWN"  # decided by code: nothing valid to judge against
    if "error" in value:
        return "ERROR"
    if value["form"] == "choice":
        probabilities = value["probabilities"]
        top = max(probabilities, key=probabilities.get)
        second = max(v for k, v in probabilities.items() if k != top)
        result = (
            top
            if probabilities[top] >= threshold and probabilities[top] - second >= margin
            else "UNCERTAIN"
        )
    else:
        s, r = value["s"], value["r"]
        if s >= threshold and r <= margin:
            result = "SUPPORTED"
        elif r >= threshold and s <= margin:
            result = "CONTRADICTED"
        elif s >= conflict_floor and r >= conflict_floor:
            result = "CONFLICTING"
        elif s <= margin and r <= margin:
            result = "UNKNOWN"
        else:
            result = "UNCERTAIN"
    sweep = value.get("sweep_r")
    if result == "SUPPORTED" and sweep is not None and sweep > margin:
        return "CONFLICTING" if sweep >= conflict_floor else "UNCERTAIN"
    return result


def average(first, second):
    if "error" in first or "error" in second or first["form"] != second["form"]:
        return first
    merged = dict(first)
    if first["form"] == "choice":
        merged["probabilities"] = {
            k: (v + second["probabilities"][k]) / 2 for k, v in first["probabilities"].items()
        }
    else:
        merged["s"], merged["r"] = (first["s"] + second["s"]) / 2, (first["r"] + second["r"]) / 2
    if first.get("sweep_r") is not None and second.get("sweep_r") is not None:
        merged["sweep_r"] = (first["sweep_r"] + second["sweep_r"]) / 2
    merged["averaged"] = True
    return merged


def borderline(value, band=(0.7, 0.85), **policy):
    return label(value, **policy) != "SUPPORTED" and band[0] <= support_score(value) < band[1]


async def judge_claims(
    asker,
    question,
    evidence,
    claims,
    *,
    form,
    scope,
    draft=None,
    only=None,
    fresh=False,
    stage="control_claims",
):
    """Judgement against all active evidence; `isolated` gives every claim its own lean state."""
    indexes = list(range(len(claims))) if only is None else list(only)
    if not evidence:
        return {
            i: {"form": form, "no_evidence": True, "error": "no_active_evidence"} for i in indexes
        }
    if scope == "isolated":
        # Independent single-claim states: nothing orders them, so they go out together.
        async def one(i):
            answers, errors = await asker.ask(
                f"{stage}:{form}:isolated",
                claim_state("lean", question, evidence, [claims[i]]),
                claim_questions(form, [0], filtered=True, note=sourced(evidence)),
                fresh=fresh,
            )
            return i, judgement(form, answers, errors, 0)

        return dict(await asyncio.gather(*(one(i) for i in indexes)))
    answers, errors = await asker.ask(
        f"{stage}:{form}:{scope}",
        claim_state(scope, question, evidence, claims, draft),
        claim_questions(form, indexes, filtered=scope == "lean", note=sourced(evidence)),
        fresh=fresh,
    )
    return {i: judgement(form, answers, errors, i) for i in indexes}


async def judge_minimal(
    asker,
    question,
    evidence,
    claims,
    cited,
    *,
    form,
    only=None,
    fresh=False,
    stage="control_claims",
):
    """Claims sharing a citation set see only those units; uncited active units are swept."""
    indexes = set(range(len(claims))) if only is None else set(only)
    by_id = {e["id"]: e for e in evidence}
    groups = {}
    for i in range(len(claims)):
        groups.setdefault(tuple(sorted(set(cited[i]) & set(by_id))), []).append(i)

    async def group(ids, members):
        wanted = [i for i in members if i in indexes]
        if not wanted:
            return {}
        if not ids:
            return {i: {"form": form, "error": "no_cited_evidence"} for i in wanted}
        # The state keeps the whole group so a re-query sees exactly the first-pass state.
        local = {i: j for j, i in enumerate(members)}
        claim_group = [claims[i] for i in members]
        asked = [local[i] for i in wanted]
        rest = [e for k, e in by_id.items() if k not in ids]

        async def sweep():
            if not rest:
                return {}, {}
            return await asker.ask(
                f"{stage}:sweep",
                {"evidence": lean(rest), "claims": claim_group},
                claim_questions("noul", asked, refute_only=True, filtered=True),
                fresh=fresh,
            )

        (answers, errors), (sweeps, sweep_errors) = await asyncio.gather(
            asker.ask(
                f"{stage}:{form}:minimal",
                {"evidence": lean([by_id[k] for k in ids]), "claims": claim_group},
                claim_questions(form, asked, filtered=True),
                fresh=fresh,
            ),
            sweep(),
        )
        found = {}
        for i in wanted:
            j = local[i]
            if f"r{j}" in sweep_errors:
                found[i] = {"form": form, "error": sweep_errors[f"r{j}"]}
                continue
            value = sweeps[f"r{j}"]["probability_true"] if rest else None
            found[i] = judgement(form, answers, errors, j, value)
            if rest and sweeps[f"r{j}"].get("split_recovered"):
                found[i]["split_recovered"] = True
        return found

    result = {}
    for found in await asyncio.gather(*(group(ids, members) for ids, members in groups.items())):
        result.update(found)
    return result


async def draft_checks(asker, question, evidence, claims, draft, *, strict=False):
    # Extraction faithfulness needs only the draft and its claims, not evidence or the query.
    (answers, errors), (minimal, minimal_errors) = await asyncio.gather(
        asker.ask(
            "control_draft_checks",
            claim_state("full", question, evidence, claims, draft),
            dict(DRAFT_CHECKS)
            | (
                {"answered_strict": {"type": "boolean", "instructions": ANSWERED_STRICT}}
                if strict
                else {}
            ),
        ),
        asker.ask(
            "control_draft_checks:coverage_minimal",
            {"draft": draft, "claims": claims},
            {"coverage": DRAFT_CHECKS["coverage"]},
        ),
    )
    value = {
        name: answers[name]["probability_true"] if name in answers else None
        for name in (*DRAFT_CHECKS, *(("answered_strict",) if strict else ()))
    }
    value["coverage_minimal"] = (
        minimal["coverage"]["probability_true"] if "coverage" in minimal else None
    )
    if errors or minimal_errors:
        value["errors"] = {**errors, **{"minimal:" + k: v for k, v in minimal_errors.items()}}
    return value


SENTENCE_COVERAGE = prompt("control.verify.SENTENCE_COVERAGE")


async def sentence_coverage(asker, sentences, claims, owner):
    """Extraction faithfulness at the unit that partial output ships: one sentence at a time."""

    async def one(i, text):
        own = [c for c, o in zip(claims, owner, strict=True) if o == i]
        if not own:
            return None
        answers, _ = await asker.ask(
            "control_sentence_coverage",
            {"sentence": text, "claims": own},
            {"coverage": {"type": "boolean", "instructions": SENTENCE_COVERAGE}},
        )
        return answers["coverage"]["probability_true"] if "coverage" in answers else None

    return list(await asyncio.gather(*(one(i, text) for i, text in enumerate(sentences))))


SUBSTANCE = prompt("control.verify.SUBSTANCE")
ANSWERED_STRICT = DRAFT_CHECKS["answered"]["instructions"] + prompt(
    "control.verify.ANSWERED_STRICT"
)


async def sentence_substance(asker, question_text, sentences):
    """Entailed is not the same as informative: one isolated question per sentence."""

    async def one(text):
        answers, _ = await asker.ask(
            "control_sentence_substance",
            {"question": question_text, "sentence": text},
            {"substance": {"type": "boolean", "instructions": SUBSTANCE}},
        )
        return answers["substance"]["probability_true"] if "substance" in answers else None

    return list(await asyncio.gather(*(one(text) for text in sentences)))


QUALIFIERS = prompt("control.verify.QUALIFIERS")


# The same check, asked one-sidedly: only a dropped restriction counts, never added precision.
QUALIFIERS_DROPPED = prompt("control.verify.QUALIFIERS_DROPPED")


async def sentence_qualifiers(asker, sentences, cited, *, dropped_only=False):
    """Supported is not the same as faithful in scope: one sentence against its own citations."""
    stage = "control_sentence_qualifiers" + (":dropped" if dropped_only else "")
    instructions = QUALIFIERS_DROPPED if dropped_only else QUALIFIERS

    async def one(text, evidence):
        if not evidence or not text:
            return None
        answers, _ = await asker.ask(
            stage,
            {"sentence": text, "evidence": lean(evidence)},
            {"qualifiers": {"type": "boolean", "instructions": instructions}},
        )
        return answers["qualifiers"]["probability_true"] if "qualifiers" in answers else None

    return list(await asyncio.gather(*(one(t, e) for t, e in zip(sentences, cited, strict=True))))


def trigrams(text):
    flat = re.sub(r"\s+", " ", text.casefold()).strip()
    return {flat[i : i + 3] for i in range(max(len(flat) - 2, 1))}


# A sentence ends at closing punctuation followed by white space. Text in scripts that are often
# set without that space ("…전망이다.17일 기상청…") also ends one where the character before the
# punctuation is a letter outside ASCII; decimals, initials and host names are left alone.
SENTENCE_BREAK = re.compile(
    r"(?<=[.!?。！？])\s+|(?<=[^\x00-\x7F\W\d_][.!?。！？])(?=[^\s.!?。！？\"'’”」』)\]])"
)


def reference_sentence(text, evidence, *, shortest=25):
    """The verbatim evidence sentence closest to `text`: a known-faithful control for that evidence."""
    wanted, best = trigrams(text), (0.0, None)
    for unit in evidence:
        parts, held = [], ""
        for piece in SENTENCE_BREAK.split(unit["text"].strip()):
            held = f"{held} {piece}".strip()
            # An initial or a short abbreviation ("Anthony J.", "St.") is not a sentence end.
            if len(held) >= shortest and not re.search(r"(?:^|\s)\S{1,2}\.$", held):
                parts.append(held)
                held = ""
        if held:
            parts.append(held)
        for part in parts:
            own = trigrams(part)
            overlap = len(wanted & own) / len(wanted | own)
            if overlap > best[0]:
                best = (overlap, part)
    return best[1]


def relative(score, control, *, floor=0.05):
    """A sentence is held to what the check says about the evidence's own wording, not to 1."""
    if score is None or control is None:
        return score
    return min(1.0, score / max(control, floor))


FITS = prompt("control.verify.FITS")
FOLLOWS = prompt("control.verify.FOLLOWS")


async def answer_checks(asker, question_text, answer, sentences, *, only=("fits", "follows")):
    """Two evidence-free looks at the short answer: is it the asked kind, and does it follow?"""
    states = {
        "fits": ("control_answer_fits", {"question": question_text, "answer": answer}, FITS),
        "follows": (
            "control_answer_follows",
            {"question": question_text, "answer": answer, "sentences": sentences},
            FOLLOWS,
        ),
    }

    async def one(name):
        stage, state, instructions = states[name]
        answers, _ = await asker.ask(
            stage, state, {"check": {"type": "boolean", "instructions": instructions}}
        )
        return answers["check"]["probability_true"] if "check" in answers else None

    names = [n for n in states if n in only]
    return dict(zip(names, await asyncio.gather(*(one(n) for n in names)), strict=True))


# Experiment P: the two shapes of wrong answer that still got through a retrieval-fed controller.
# Not part of the controller's signals unless docs/CONTROL_SUITE_P_REPORT.md says they earned it.
CONDITIONS = prompt("control.verify.CONDITIONS")
NAMES_ENTITY = prompt("control.verify.NAMES_ENTITY")


async def targeted_checks(asker, question_text, answer, sentences, cited):
    """`conditions` sees no evidence; `names_entity` sees only what the draft cites (the minimum)."""

    async def noul(stage, state, instructions):
        answers, _ = await asker().ask(
            stage, state, {"check": {"type": "boolean", "instructions": instructions}}
        )
        return answers["check"]["probability_true"] if "check" in answers else None

    # One look at everything the draft cites: a second hop may rightly be about another entity.
    units, seen = [], set()
    for unit in (u for group in cited for u in lean(group)):
        if unit["id"] not in seen:
            seen.add(unit["id"])
            units.append(unit)
    conditions, named = await asyncio.gather(
        noul(
            "control_answer_conditions",
            {"question": question_text, "answer": answer, "sentences": sentences},
            CONDITIONS,
        ),
        noul(
            "control_names_entity",
            {"question": question_text, "evidence": units[:8]},
            NAMES_ENTITY,
        ),
    )
    return {"conditions": conditions, "names_entity": named}


# Experiment R: the wrong answers that still get through are faithful to what they cite, but the
# question takes a relation or an entity for granted that the citation does not give. Named
# outcomes, because a named near-miss is what made the gold grader work.
PREMISE_RULES = prompt("control.verify.PREMISE_RULES")
PREMISE = {
    "AGREES": prompt("control.verify.PREMISE.AGREES"),
    "REVERSED_OR_OTHER": prompt("control.verify.PREMISE.REVERSED_OR_OTHER"),
    "NOT_STATED": prompt("control.verify.PREMISE.NOT_STATED"),
}
BOUND_RULES = prompt("control.verify.BOUND_RULES")
BOUND = {
    "NAMED": prompt("control.verify.BOUND.NAMED"),
    "UNNAMED": prompt("control.verify.BOUND.UNNAMED"),
    "OTHER": prompt("control.verify.BOUND.OTHER"),
    "NO_ENTITY": prompt("control.verify.BOUND.NO_ENTITY"),
}


async def relation_checks(asker, question_text, answer, sentences, cited):
    """`premise` once over everything the draft cites; `bound` per sentence, the weakest counts."""

    async def choice(stage, state, rules, outcomes, wanted):
        answers, _ = await asker().ask(
            stage,
            state,
            {"check": {"type": "choice", "instructions": rules, "criteria": dict(outcomes)}},
        )
        if "check" not in answers:
            return None
        return sum(answers["check"]["probabilities"][name] for name in wanted)

    units, seen = [], set()
    for unit in (u for group in cited for u in lean(group)):
        if unit["id"] not in seen:
            seen.add(unit["id"])
            units.append(unit)
    pairs = [(s, lean(group)) for s, group in zip(sentences, cited, strict=True) if group]
    found = await asyncio.gather(
        choice(
            "control_premise",
            {"question": question_text, "answer": answer, "evidence": units[:8]},
            PREMISE_RULES,
            PREMISE,
            ("AGREES",),
        ),
        *(
            choice(
                "control_bound",
                {"sentence": s, "evidence": group},
                BOUND_RULES,
                BOUND,
                ("NAMED", "NO_ENTITY"),
            )
            for s, group in pairs
        ),
    )
    bound = [v for v in found[1:] if v is not None]
    return {"premise": found[0], "bound": min(bound) if bound else None}


SELF_CONTAINED = prompt("control.verify.SELF_CONTAINED")


async def self_contained(asker, question_text):
    answers, _ = await asker.ask(
        "control_self_contained",
        {"question": question_text},
        {"check": {"type": "boolean", "instructions": SELF_CONTAINED}},
    )
    return answers["check"]["probability_true"] if "check" in answers else None


# Experiment W-3: what the question takes for granted, judged by the floor's own claim gate. No
# new judgement is defined; a twisted premise should come back CONTRADICTED. Experimental.
PRESUPPOSITIONS = prompt("control.verify.PRESUPPOSITIONS")


async def presuppositions(extractor, question_text, context):
    from pydantic import Field

    from ..contracts import Contract
    from ..providers.generation import GenerationOptions, GenerationRequest

    class Presuppositions(Contract):
        claims: tuple[str, ...] = Field(max_length=4)

    result = await extractor.generate_structured(
        GenerationRequest(
            role="semantic",
            system=PRESUPPOSITIONS,
            payload={"question": question_text},
            response_type=Presuppositions,
            schema_name="presuppositions_v1",
            prompt_version="presuppositions-1",
            options=GenerationOptions(max_tokens=400),
        ),
        context,
    )
    return [c.strip() for c in result.value.claims if c.strip()]


async def presupposition_gate(asker, question, evidence, claims):
    """Each presupposition against the evidence, isolated, with the floor's 4-way claim question."""
    if not claims or not evidence:
        return {"presupposition_contradicted": 0.0, "claims": claims, "verdicts": []}
    judged = await judge_claims(asker, question, evidence, claims, form="choice", scope="isolated")
    values = [judged[i] for i in range(len(claims))]
    return {
        "presupposition_contradicted": max(refute_score(v) for v in values),
        "presupposition_supported_min": min(support_score(v) for v in values),
        "claims": claims,
        "verdicts": [label(v) if "error" not in v else "ERROR" for v in values],
        "probabilities": [v.get("probabilities") for v in values],
    }


# The same two looks as a choice whose near-miss has a name. That wording fixed the gold grader
# (docs/CONTROL_GRADER_REPORT.md) but bought no coverage here (docs/CONTROL_SUITE_G_REPORT.md):
# kept for reproduction, not part of the controller's signals.
NEAR_MISS = prompt("control.verify.NEAR_MISS")
ASKED_ITEM = {
    "ASKED_ITEM": prompt("control.verify.ASKED_ITEM.ASKED_ITEM"),
    "RELATED_NOT_ASKED": NEAR_MISS,
    "VAGUE": prompt("control.verify.ASKED_ITEM.VAGUE"),
    "OFF": prompt("control.verify.ASKED_ITEM.OFF"),
}
ADDRESSES = {
    "ANSWERS": prompt("control.verify.ADDRESSES.ANSWERS"),
    "RELATED_NOT_ASKED": NEAR_MISS,
    "UNSETTLED": prompt("control.verify.ADDRESSES.UNSETTLED"),
    "OFF": prompt("control.verify.ADDRESSES.OFF"),
}
ADDRESS_RULES = prompt("control.verify.ADDRESS_RULES")


async def answer_addresses(
    asker, question_text, answer, sentences, *, only=("asked_item", "addresses")
):
    """Evidence-free, like `answer_checks`; the score is the probability of the first outcome."""
    states = {
        "asked_item": ({"question": question_text, "answer": answer}, ASKED_ITEM),
        "addresses": (
            {"question": question_text, "answer": answer, "sentences": sentences},
            ADDRESSES,
        ),
    }

    async def one(name):
        state, criteria = states[name]
        answers, _ = await asker.ask(
            "control_answer_" + name,
            state,
            {
                "check": {
                    "type": "choice",
                    "instructions": ADDRESS_RULES,
                    "criteria": dict(criteria),
                }
            },
        )
        return (
            answers["check"]["probabilities"][next(iter(criteria))] if "check" in answers else None
        )

    names = [n for n in states if n in only]
    return dict(zip(names, await asyncio.gather(*(one(n) for n in names)), strict=True))


# Graded (Score) judgements: the meaning of a cut lives in the level descriptions, not in a number.
RESTRICTED = prompt("control.verify.RESTRICTED")
ANSWERABLE = (
    prompt("control.verify.ANSWERABLE.instructions"),
    [
        prompt("control.verify.ANSWERABLE.levels.0"),
        prompt("control.verify.ANSWERABLE.levels.1"),
        prompt("control.verify.ANSWERABLE.levels.2"),
        prompt("control.verify.ANSWERABLE.levels.3"),
        prompt("control.verify.ANSWERABLE.levels.4"),
    ],
)
STRICTNESS = (
    prompt("control.verify.STRICTNESS.instructions"),
    [
        prompt("control.verify.STRICTNESS.levels.0"),
        prompt("control.verify.STRICTNESS.levels.1"),
        prompt("control.verify.STRICTNESS.levels.2"),
        prompt("control.verify.STRICTNESS.levels.3"),
        prompt("control.verify.STRICTNESS.levels.4"),
    ],
)
ANSWERED_LEVELS = (
    prompt("control.verify.ANSWERED_LEVELS.instructions"),
    [
        prompt("control.verify.ANSWERED_LEVELS.levels.0"),
        prompt("control.verify.ANSWERED_LEVELS.levels.1"),
        prompt("control.verify.ANSWERED_LEVELS.levels.2"),
        prompt("control.verify.ANSWERED_LEVELS.levels.3"),
        prompt("control.verify.ANSWERED_LEVELS.levels.4"),
    ],
)
HOLISTIC_LEVELS = (
    prompt("control.verify.HOLISTIC_LEVELS.instructions"),
    [
        prompt("control.verify.HOLISTIC_LEVELS.levels.0"),
        prompt("control.verify.HOLISTIC_LEVELS.levels.1"),
        prompt("control.verify.HOLISTIC_LEVELS.levels.2"),
        prompt("control.verify.HOLISTIC_LEVELS.levels.3"),
        prompt("control.verify.HOLISTIC_LEVELS.levels.4"),
    ],
)


async def graded(asker, stage, state, levels):
    """Score question; the expected level is scaled to 0..1 so cuts read like the other checks."""
    instructions, criteria = levels
    answers, _ = await asker.ask(
        stage,
        state,
        {"level": {"type": "score", "instructions": instructions, "criteria": list(criteria)}},
    )
    return answers["level"]["score"] / (len(criteria) - 1) if "level" in answers else None


async def instance_checks(ask, question_text, evidence):
    """What question and evidence alone say about the case, before any draft exists."""
    state = {"question": question_text, "evidence": lean(evidence)}

    async def restricted():
        answers, _ = await ask().ask(
            "control_instance_restricted",
            state,
            {"check": {"type": "boolean", "instructions": RESTRICTED}},
        )
        return answers["check"]["probability_true"] if "check" in answers else None

    found = await asyncio.gather(
        restricted(),
        graded(ask(), "control_instance_answerable", state, ANSWERABLE),
        graded(ask(), "control_instance_strictness", state, STRICTNESS),
        return_exceptions=True,
    )
    names = ("restricted", "answerable", "strictness")
    return {n: None if isinstance(v, Exception) else v for n, v in zip(names, found, strict=True)}


async def anchored_checks(ask, question, evidence, sentences, draft):
    """Graded versions of the draft-level looks, on the same state as `direct_checks`."""
    state = claim_state("full", question, evidence, sentences, draft)
    found = await asyncio.gather(
        graded(ask(), "control_answered_score", state, ANSWERED_LEVELS),
        graded(ask(), "control_holistic_score", state, HOLISTIC_LEVELS),
        return_exceptions=True,
    )
    names = ("answered_score", "holistic_score")
    return {n: None if isinstance(v, Exception) else v for n, v in zip(names, found, strict=True)}


KINDS = {
    "comparison": prompt("control.verify.KINDS.comparison"),
    "other": prompt("control.verify.KINDS.other"),
}


async def question_kind(asker, question_text):
    """Deployable grouping: decided from the wording of the question alone."""
    answers, _ = await asker.ask(
        "control_question_kind",
        {"question": question_text},
        {
            "kind": {
                "type": "choice",
                "instructions": prompt("control.verify.question_kind.kind.instructions"),
                "criteria": dict(KINDS),
            }
        },
    )
    if "kind" not in answers:
        return None
    probabilities = answers["kind"]["probabilities"]
    return max(probabilities, key=probabilities.get)


async def direct_checks(asker, question, evidence, sentences, draft):
    """Draft-level checks of the extractor-free path: the sentences are the claims."""
    names = ("answered", "citations")
    questions = {n: DRAFT_CHECKS[n] for n in names} | {
        "answered_strict": {"type": "boolean", "instructions": ANSWERED_STRICT}
    }
    answers, errors = await asker.ask(
        "control_direct_checks",
        claim_state("full", question, evidence, sentences, draft),
        questions,
    )
    value = {n: answers[n]["probability_true"] if n in answers else None for n in questions}
    return value | ({"errors": errors} if errors else {})


PAIR_RULES = prompt("control.verify.PAIR_RULES")
PAIR_SUPPORT = prompt("control.verify.PAIR_SUPPORT") + PAIR_RULES
PAIR_REFUTE = prompt("control.verify.PAIR_REFUTE") + PAIR_RULES


async def judge_pair(asker, item, *, form, fresh=False):
    """One claim against its given evidence, as in the public claim/evidence datasets."""
    from ..approval.core import PROTOCOL, choice_question

    questions = (
        {"c0": choice_question()}
        if form == "choice"
        else {
            "s0": {"type": "boolean", "instructions": PAIR_SUPPORT},
            "r0": {"type": "boolean", "instructions": PAIR_REFUTE},
        }
    )
    answers, errors = await asker.ask(
        f"{PROTOCOL}:{form}", item.model_dump(), questions, fresh=fresh
    )
    return judgement(form, answers, errors, 0)
