from __future__ import annotations

from pydantic import Field

from ..contracts import Contract
from ..prompts import prompt
from ..providers.generation import GenerationOptions, GenerationRequest

ANSWERS = prompt("control.compose.ANSWERS")
DEPENDS = prompt("control.compose.DEPENDS")

CONTENTLESS = prompt("control.compose.CONTENTLESS")
ANSWERS_STRICT = ANSWERS.replace(
    prompt("control.compose.ANSWERS_STRICT.replace0"),
    CONTENTLESS + prompt("control.compose.ANSWERS_STRICT.replace1"),
)


class Slot(Contract):
    id: str = Field(pattern=r"^s\d$")
    ask: str = Field(min_length=1, max_length=400)


class Slots(Contract):
    slots: tuple[Slot, ...] = Field(min_length=1, max_length=6)


async def decompose(generator, question_text, context):
    """Requested items come from the question itself, never from a domain schema in code."""
    result = await generator.generate_structured(
        GenerationRequest(
            role="semantic",
            system=prompt("control.compose.decompose.system"),
            payload={"question": question_text},
            response_type=Slots,
            schema_name="control_slots_v1",
            prompt_version="control-slots-1",
            options=GenerationOptions(max_tokens=600),
        ),
        context,
    )
    slots = [s.model_dump() for s in result.value.slots]
    if [s["id"] for s in slots] != [f"s{i}" for i in range(len(slots))]:
        return None
    return slots


async def relate(asker, question_text, slots, sentences, *, strict=False):
    """(sentence, slot) answers/depends probabilities from one shared-state request."""
    questions = {}
    for i in range(len(sentences)):
        for j in range(len(slots)):
            questions[f"a{i}_{j}"] = {"type": "boolean", "instructions": ANSWERS.format(i=i, j=j)}
            questions[f"d{i}_{j}"] = {"type": "boolean", "instructions": DEPENDS.format(i=i, j=j)}
            if strict:  # same state, so the stricter wording costs no extra request
                questions[f"b{i}_{j}"] = {
                    "type": "boolean",
                    "instructions": ANSWERS_STRICT.format(i=i, j=j),
                }
    answers, errors = await asker.ask(
        "control_slot_relations_v2",
        {"question": question_text, "slots": slots, "sentences": sentences},
        questions,
    )
    if errors:
        return None
    grid = lambda name: [
        [answers[f"{name}{i}_{j}"]["probability_true"] for j in range(len(slots))]
        for i in range(len(sentences))
    ]
    return {"answers": grid("a"), "depends": grid("d")} | (
        {"answers_strict": grid("b")} if strict else {}
    )


def compose(
    sentences,
    accepted,
    relations,
    slots,
    *,
    partial,
    answer_threshold=0.85,
    dependency_threshold=0.5,
):
    """Dependency closure over verified sentences; PARTIAL is never reported as FULL."""
    answers = [[p >= answer_threshold for p in row] for row in relations["answers"]]
    depends = [
        [p >= dependency_threshold and not answers[i][j] for j, p in enumerate(row)]
        for i, row in enumerate(relations["depends"])
    ]
    shipped = set(accepted)
    while True:
        covered = {j for i in shipped for j in range(len(slots)) if answers[i][j]}
        dropped = {
            i for i in shipped if any(d and j not in covered for j, d in enumerate(depends[i]))
        }
        if not dropped:
            break
        shipped -= dropped
    complete = len(covered) == len(slots)
    if not complete and not partial:
        shipped, covered = set(), set()
    status = "FULL" if complete and shipped else "PARTIAL" if covered else "ABSTAIN"
    if status == "ABSTAIN":
        shipped = set()
    order = sorted(shipped)
    return {
        "status": status,
        "sentences": order,
        "text": " ".join(sentences[i] for i in order),
        "answered_slots": sorted(covered) if order else [],
        "missing_slots": [j for j in range(len(slots)) if j not in covered or not order],
        "dropped_for_dependency": sorted(set(accepted) - shipped) if status != "ABSTAIN" else [],
    }
