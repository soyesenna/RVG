from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..contracts import Contract
from ..prompts import prompt
from ..providers.generation import GenerationOptions, GenerationRequest, strict_schema
from ..temporal.models import AnswerDraft, AnswerSentence
from .verify import as_query, sourced

PRESERVE = prompt("control.writer.PRESERVE")
# U12-d: the conflict form. The default instruction drops an item whose sources disagree; this
# form states each value with its source instead of choosing, and never resolves by recency.
CONFLICT = prompt("control.writer.CONFLICT")
# N6-2 (arm only): the answer-form contract. The writer decides from the query and the evidence
# whether the answer is a value, a list or a procedure; a procedure is one sentence per step.
FORM = prompt("control.writer.FORM")


# N5 (arm only): the structured writing contract. The core answer, the sentences that state it and
# the evidence spans it rests on are separate fields, so the short answer can be copied exactly as
# the evidence writes it; the comparison rules downstream do not change.
STRUCTURED = prompt("control.writer.STRUCTURED")
QUOTED = prompt("control.writer.QUOTED")


class SupportingSpan(Contract):
    unit_id: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=600)


class StructuredDraft(Contract):
    answerable: bool
    short_answer: str = Field(max_length=1000)
    sentences: tuple[AnswerSentence, ...] = Field(max_length=8)
    answer_basis: Literal["extractive", "derived"]
    supporting_spans: tuple[SupportingSpan, ...] = Field(max_length=12)
    conditions: tuple[str, ...] = Field(max_length=6)


def bound(structured, evidence):
    """W2: every quote is a verbatim part of the unit it names, and an extractive short answer
    lies inside one of them. Decided by string matching, never by a model."""
    text = {e["id"]: e["text"] for e in evidence}
    quotes = [(s.unit_id, s.quote) for s in structured.supporting_spans]
    if not quotes or any(u not in text or q not in text[u] for u, q in quotes):
        return False
    return structured.answer_basis == "derived" or any(
        structured.short_answer.strip() in q for _, q in quotes
    )


def refusal(draft, allowed, *, quoted_unbound=False):
    """Why a written draft is not a draft (PROD-1 handoff packet): the first condition that
    fails, in the order the writer checks them, or None when it is one."""
    if not draft.answerable:
        return "not_answerable"
    if not draft.sentences:
        return "no_sentences"
    if not draft.short_answer.strip():
        return "empty_short_answer"
    if any(not set(s.evidence_ids) <= allowed for s in draft.sentences):
        return "citation_outside_evidence"
    if quoted_unbound:
        return "quotes_not_bound"
    return None


def note(trail, written, allowed, *, quoted_unbound=False):
    """PROD-1: the writer's part of a handoff trail — whether it declined, why, and what it
    returned (observation only; nothing here changes the draft or the decision)."""
    if trail is None:
        return
    why = refusal(written, allowed, quoted_unbound=quoted_unbound)
    trail["writer"] = {
        "refused": why is not None,
        "refusal": why,
        "raw": written.model_dump(mode="json") if why is not None else None,
    }


async def write_answer(
    writer,
    question,
    evidence,
    context,
    *,
    issues=None,
    preserve=False,
    conflict=False,
    schema="W0",
    contract="W0",
    trail=None,
):
    """`schema` W1/W2 (N5 arms) write the structured form; W0 is the policy's form, unchanged.
    `contract` "form-v1" (N6-2 arm) replaces the plain form's base instruction (FORM). `trail`
    (PROD-1, a dict only when the handoff packet is on) receives the writer's refusal and output."""
    if schema != "W0":
        return await write_structured(
            writer,
            question,
            evidence,
            context,
            issues=issues,
            preserve=preserve,
            conflict=conflict,
            quoted=schema == "W2",
            trail=trail,
        )
    return await write_plain(
        writer,
        question,
        evidence,
        context,
        issues=issues,
        preserve=preserve,
        conflict=conflict,
        form=contract == "form-v1",
        trail=trail,
    )


async def write_structured(
    writer,
    question,
    evidence,
    context,
    *,
    issues=None,
    preserve=False,
    conflict=False,
    quoted=False,
    trail=None,
):
    if not evidence:
        return None
    schema = strict_schema(StructuredDraft)
    ids = [e["id"] for e in evidence]
    schema["$defs"]["AnswerSentence"]["properties"]["evidence_ids"]["items"]["enum"] = ids
    schema["$defs"]["SupportingSpan"]["properties"]["unit_id"]["enum"] = ids
    result = await writer.generate_structured(
        GenerationRequest(
            role="surface",
            system=BASE
            + (PRESERVE if preserve else "")
            + (CONFLICT if conflict else "")
            + STRUCTURED
            + (QUOTED if quoted else "")
            + sourced(evidence),
            payload={"query": as_query(question), "evidence": evidence, "issues": issues or {}},
            response_type=StructuredDraft,
            schema_name="temporal_answer_v2",
            prompt_version=("control-writer-structured-" + ("2" if quoted else "1"))
            + ("-preserve" if preserve else "")
            + ("-conflict" if conflict else ""),
            options=GenerationOptions(max_tokens=2048),
            schema_override=schema,
        ),
        context,
    )
    structured = result.value
    allowed = set(ids)
    note(
        trail,
        structured,
        allowed,
        quoted_unbound=bool(quoted and structured.answerable and not bound(structured, evidence)),
    )
    if (
        not structured.answerable
        or not structured.sentences
        or not structured.short_answer.strip()
        or any(not set(s.evidence_ids) <= allowed for s in structured.sentences)
        or (quoted and not bound(structured, evidence))
    ):
        return None
    return AnswerDraft(
        answerable=True, short_answer=structured.short_answer, sentences=structured.sentences
    )


BASE = prompt("control.writer.BASE")


async def write_plain(
    writer,
    question,
    evidence,
    context,
    *,
    issues=None,
    preserve=False,
    conflict=False,
    form=False,
    trail=None,
):
    """One sentence per requested item; unsupported items are left out rather than guessed.
    `conflict=True` (U12-d) states disagreeing values side by side instead of dropping them."""
    if not evidence:
        return None
    schema = strict_schema(AnswerDraft)
    schema["$defs"]["AnswerSentence"]["properties"]["evidence_ids"]["items"]["enum"] = [
        e["id"] for e in evidence
    ]
    result = await writer.generate_structured(
        GenerationRequest(
            role="surface",
            system=(FORM if form else BASE)
            + (PRESERVE if preserve else "")
            + (CONFLICT if conflict else "")
            + sourced(evidence),
            payload={"query": as_query(question), "evidence": evidence, "issues": issues or {}},
            response_type=AnswerDraft,
            schema_name="temporal_answer_v1",
            prompt_version=("control-writer-2" if preserve else "control-writer-1")
            + ("-form" if form else "")
            + ("-conflict" if conflict else ""),
            options=GenerationOptions(max_tokens=1536),
            schema_override=schema,
        ),
        context,
    )
    draft = result.value
    allowed = {e["id"] for e in evidence}
    note(trail, draft, allowed)
    if (
        not draft.answerable
        or not draft.sentences
        or not draft.short_answer.strip()
        or any(not set(s.evidence_ids) <= allowed for s in draft.sentences)
    ):
        return None
    return draft


async def write_with_second_try(writer, question, evidence, context, **options):
    """One more try, with the evidence in reverse order, when the writer declines.

    The writer's refusal depends on the order of the evidence about as much as on its content:
    this recovered as many declined questions as letting Jev narrow the evidence first
    (docs/CONTROL_SUITE_H_REPORT.md). The controller judges a second-try draft exactly like a
    first one. Returns (draft or None, whether the second try produced it).
    """
    draft = await write_answer(writer, question, evidence, context, **options)
    if draft is not None or len(evidence) < 2:
        return draft, False
    draft = await write_answer(writer, question, evidence[::-1], context, **options)
    return draft, draft is not None
