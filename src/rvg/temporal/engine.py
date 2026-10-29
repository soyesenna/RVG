"""Claim extraction and the claim/draft judgement questions of the controller. The legacy
temporal answer engine and its claim verifier stayed in the research repository."""

from __future__ import annotations

from ..prompts import prompt
from ..providers.generation import GenerationOptions, GenerationRequest, strict_schema
from .models import ExtractedClaims

CLAIM_INSTRUCTIONS = prompt("temporal.engine.CLAIM_INSTRUCTIONS")


CLAIM_CRITERIA = {
    "SUPPORTED": prompt("temporal.engine.CLAIM_CRITERIA.SUPPORTED"),
    "CONTRADICTED": prompt("temporal.engine.CLAIM_CRITERIA.CONTRADICTED"),
    "UNKNOWN": prompt("temporal.engine.CLAIM_CRITERIA.UNKNOWN"),
    "CONFLICTING": prompt("temporal.engine.CLAIM_CRITERIA.CONFLICTING"),
}


DRAFT_CHECKS = {
    "coverage": {
        "type": "boolean",
        "instructions": prompt("temporal.engine.DRAFT_CHECKS.coverage.instructions"),
    },
    "answered": {
        "type": "boolean",
        "instructions": prompt("temporal.engine.DRAFT_CHECKS.answered.instructions"),
    },
    "citations": {
        "type": "boolean",
        "instructions": prompt("temporal.engine.DRAFT_CHECKS.citations.instructions"),
    },
}


def claim_choice(i):
    return {
        "type": "choice",
        "instructions": CLAIM_INSTRUCTIONS.format(i=i),
        "criteria": dict(CLAIM_CRITERIA),
    }


EXTRACTION_SYSTEM = prompt("temporal.engine.EXTRACTION_SYSTEM")


async def extract_claims(extractor, question, draft, context, *, previous=None):
    """Independent claim extraction; `previous` marks the single bounded re-extraction."""
    schema = strict_schema(ExtractedClaims)
    schema["$defs"]["SentenceClaims"]["properties"]["sentence_index"]["enum"] = list(
        range(len(draft.sentences))
    )
    query = question.model_dump(mode="json") if hasattr(question, "model_dump") else dict(question)
    payload = {"query": query, "draft": draft.model_dump(mode="json")}
    system = EXTRACTION_SYSTEM
    if previous is not None:
        payload["rejected_extraction"] = previous
        system += prompt("temporal.engine.extract_claims.reextraction")
    result = await extractor.generate_structured(
        GenerationRequest(
            role="semantic",
            system=system,
            payload=payload,
            response_type=ExtractedClaims,
            schema_name="temporal_claims_v1",
            prompt_version="temporal-extraction-2"
            if previous is None
            else "temporal-reextraction-1",
            options=GenerationOptions(max_tokens=2048),
            schema_override=schema,
        ),
        context,
    )
    expected = set(range(len(draft.sentences)))
    groups = result.value.sentences
    if {c.sentence_index for c in groups} != expected or len(groups) != len(expected):
        return None
    return result.value
