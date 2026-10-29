"""N5: the structured writing contract (W1) and its quote binding (W2), as arms. W0 — the policy's
writer — is unchanged byte for byte (its prompt and schema name are those of the recorded tapes).
W2 decides by string matching whether each quote is a verbatim part of the unit it names and an
extractive short answer lies inside one; the comparison rule downstream (`grade.flat`, strict
identical text) does not change, and its minimal pairs are pinned here as they are."""

import asyncio

import pytest

from rvg.control.grade import flat
from rvg.control.writer import (
    BASE,
    StructuredDraft,
    bound,
    write_answer,
)
from rvg.providers.generation import GenerationResult
from rvg.runtime import CallContext
from rvg.temporal.models import TemporalQuestion
from rvg.universal.experiment import Arm

EVIDENCE = [
    {"id": "a@1#u0", "title": "Fleet", "text": "The airline operates 44 aircraft as of 2025."},
    {"id": "b@1#u0", "title": "Share", "text": "Its market share is 44% in the region."},
]
QUESTION = TemporalQuestion(
    text="How many aircraft does the airline operate?",
    as_of="2026-09-20T00:00:00Z",
    known_at="2026-09-20T00:00:00Z",
)


class Recorder:
    """A writer that records the request and answers with a fixed structured draft."""

    def __init__(self, value):
        self.value, self.requests = value, []

    async def generate_structured(self, request, context):
        self.requests.append(request)
        return GenerationResult(request.response_type.model_validate(self.value), {})


def structured(short, quote, unit="a@1#u0", basis="extractive"):
    return {
        "answerable": True,
        "short_answer": short,
        "sentences": [{"text": f"The airline operates {short}.", "evidence_ids": [unit]}],
        "answer_basis": basis,
        "supporting_spans": [{"unit_id": unit, "quote": quote}],
        "conditions": [],
    }


def write(value, schema):
    writer = Recorder(value)
    draft = asyncio.run(
        write_answer(writer, QUESTION, EVIDENCE, CallContext.with_timeout(10), schema=schema)
    )
    return draft, writer.requests[0]


def test_w0_is_the_policy_writer_unchanged():
    writer = Recorder(
        {
            "answerable": True,
            "short_answer": "44",
            "sentences": [
                {"text": "The airline operates 44 aircraft.", "evidence_ids": ["a@1#u0"]}
            ],
        }
    )
    asyncio.run(write_answer(writer, QUESTION, EVIDENCE, CallContext.with_timeout(10)))
    request = writer.requests[0]
    assert request.schema_name == "temporal_answer_v1"
    assert request.prompt_version == "control-writer-1"
    assert request.system.startswith(BASE) and "STRUCTURED" not in request.system


def test_w1_writes_the_structured_form_and_hands_the_controller_a_plain_draft():
    draft, request = write(structured("44 aircraft", "44 aircraft"), "W1")
    assert request.schema_name == "temporal_answer_v2" and "STRUCTURED FORM" in request.system
    assert "character for character" not in request.system
    assert draft.short_answer == "44 aircraft" and draft.sentences[0].evidence_ids == ("a@1#u0",)


@pytest.mark.parametrize(
    ("short", "quote", "unit", "basis", "kept"),
    [
        ("44 aircraft", "operates 44 aircraft", "a@1#u0", "extractive", True),
        ("44", "44 aircraft", "a@1#u0", "extractive", True),
        ("44%", "44 aircraft", "a@1#u0", "extractive", False),  # not inside the quote
        ("44 aircraft", "44 planes", "a@1#u0", "extractive", False),  # not verbatim
        ("44 aircraft", "44 aircraft", "b@1#u0", "extractive", False),  # quote from another unit
        ("88", "44 aircraft", "a@1#u0", "derived", True),  # computed: only the quote is checked
    ],
)
def test_w2_binds_quotes_to_the_unit_text_by_string_matching(short, quote, unit, basis, kept):
    draft, request = write(structured(short, quote, unit, basis), "W2")
    assert "character for character" in request.system
    assert (draft is not None) == kept
    assert (
        bound(StructuredDraft.model_validate(structured(short, quote, unit, basis)), EVIDENCE)
        == kept
    )


def test_the_strict_comparison_keeps_these_minimal_pairs_apart():
    for a, b in [
        ("up to 44", "exactly 44"),
        ("A acquired B", "B acquired A"),
        ("본사", "자회사"),
        ("2024", "2025"),
        ("44 aircraft", "44"),
    ]:
        assert flat(a) != flat(b), (a, b)


def test_the_strict_comparison_does_not_see_symbols_a_known_limit():
    # `flat` drops every non-word character, so a percent sign is not compared: the second-stage
    # agreement rule treats "44" and "44%" as the same text (the released sentence is still
    # judged claim by claim). Recorded as a limitation, not changed by N5.
    assert flat("44") == flat("44%")


def test_arms_accept_only_the_known_writer_schemas():
    assert Arm.of("n5-w2", cascade={"writer_schema": "W2"}).stamp()["cascade"] == {
        "writer_schema": "W2"
    }
    with pytest.raises(ValueError, match="unknown_writer_schema"):
        Arm.of("bad", cascade={"writer_schema": "W9"})
