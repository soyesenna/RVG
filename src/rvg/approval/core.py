"""The three-way claim/evidence judgement question used by claim verification. The learned
approval research (features, model artifacts) stayed in the research repository."""

from __future__ import annotations

from ..prompts import prompt

PROTOCOL = "claim-evidence-3way-v1"


def choice_question():
    return {
        "type": "choice",
        "instructions": prompt("approval.core.choice_question.instructions"),
        "criteria": {
            "SUPPORTED": prompt("approval.core.choice_question.criteria.SUPPORTED"),
            "CONTRADICTED": prompt("approval.core.choice_question.criteria.CONTRADICTED"),
            "UNKNOWN": prompt("approval.core.choice_question.criteria.UNKNOWN"),
        },
    }
