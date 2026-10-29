from __future__ import annotations

import json
import math

from ..contracts import (
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)
from ..errors import ContractError


def probability(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ContractError("invalid_probability")
    return float(value)


def validate_request(request: DecisionRequest) -> None:
    if not isinstance(request.state, (str, dict, list)):
        raise ContractError("invalid_state_type")
    try:
        state_size = len(json.dumps(request.state, ensure_ascii=False, allow_nan=False).encode())
        question_sizes = [len(q.model_dump_json().encode()) for q in request.questions.values()]
        if state_size + max(question_sizes) > 24000 or state_size + sum(question_sizes) > 48000:
            raise ContractError("request_exceeds_local_byte_limit")
        for qid, question in request.questions.items():
            if not qid.strip() or not question.instructions.strip():
                raise ContractError("empty_question_id_or_instructions")
            if isinstance(question, ChoiceQuestion) and any(
                not k.strip() for k in question.criteria
            ):
                raise ContractError("empty_option_id")
    except (ValueError, TypeError):
        raise ContractError("invalid_request_json") from None


def parse_answers(payload: dict, request: DecisionRequest, *, native: bool = False) -> dict:
    answers = payload.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(request.questions):
        raise ContractError("response_question_ids_mismatch")
    rounding = payload.get("rounding", {})
    if not isinstance(rounding, dict):
        raise ContractError("invalid_rounding_metadata")
    decimals = rounding.get("probabilityDecimals")
    score_decimals = rounding.get("scoreDecimals")
    if any(
        value is not None and (type(value) is not int or not 0 <= value <= 15)
        for value in (decimals, score_decimals)
    ):
        raise ContractError("invalid_rounding_metadata")
    # Declared precision controls accumulation error, not the number of options.
    # Keep a local minimum precision for confidence-sensitive selection.
    if decimals is not None and decimals < 2:
        raise ContractError("probability_rounding_too_coarse")
    probability_error = 0 if decimals is None else 0.5 * 10 ** (-decimals)
    score_error = 0 if score_decimals is None else 0.5 * 10 ** (-score_decimals)
    normalized = {}
    for qid, q in request.questions.items():
        answer = answers[qid]
        expected_type = "noul" if native and isinstance(q, BooleanQuestion) else q.type
        if not isinstance(answer, dict) or answer.get("type") != expected_type:
            raise ContractError("response_question_type_mismatch")
        if isinstance(q, BooleanQuestion):
            normalized[qid] = BooleanAnswer(
                probability_true=probability(answer.get("noul" if native else "probability"))
            )
            continue
        expected = (
            set(q.criteria)
            if isinstance(q, ChoiceQuestion)
            else {str(i) for i in range(len(q.criteria))}
        )
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != expected:
            raise ContractError("missing_or_incomplete_choice_distribution")
        probs = {key: probability(value) for key, value in probs.items()}
        error = probability_error
        if decimals is None:
            # Some providers round to a few decimals without declaring it; bound the sum by
            # the precision actually observed instead of rejecting 0.99 as a broken distribution.
            places = max(len(repr(v).partition(".")[2]) for v in probs.values())
            if places <= 4 and all("e" not in repr(v) for v in probs.values()):
                error = 0.5 * 10 ** (-max(places, 2))
        tolerance = 0.000001 + len(probs) * error
        if abs(sum(probs.values()) - 1) > tolerance + 1e-9:
            raise ContractError(
                "probability_sum_mismatch",
                detail={
                    "sum": round(sum(probs.values()), 9),
                    "options": len(probs),
                    "declared_decimals": decimals,
                    "tolerance": tolerance,
                },
            )
        confidence = answer.get("confidence")
        if confidence is not None:
            confidence = probability(confidence)
        if isinstance(q, ChoiceQuestion):
            chosen = answer.get("choice")
            if not isinstance(chosen, str) or chosen not in expected:
                raise ContractError("unknown_choice_option")
            if probs[chosen] + 1e-9 < max(probs.values()):
                raise ContractError("choice_not_argmax")
            normalized[qid] = ChoiceAnswer(
                choice=chosen, probabilities=probs, provider_confidence=confidence
            )
        elif isinstance(q, ScoreQuestion):
            score = answer.get("score")
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not 0 <= score <= len(expected) - 1
            ):
                raise ContractError("invalid_score")
            mean = sum(int(key) * value for key, value in probs.items())
            rounding_of_score = score_error
            if score_decimals is None:
                # Same undeclared rounding as above: the mean of rounded probabilities can sit a
                # few hundredths away from a score the provider computed before rounding.
                places = len(repr(float(score)).partition(".")[2])
                if places <= 4 and "e" not in repr(float(score)):
                    rounding_of_score = 0.5 * 10 ** (-max(places, 2))
            mean_tolerance = 0.000001 + sum(range(len(probs))) * error + rounding_of_score
            if abs(mean - score) > mean_tolerance:
                raise ContractError("score_distribution_mismatch")
            normalized[qid] = ScoreAnswer(
                score=score, probabilities=probs, provider_confidence=confidence
            )
    return normalized


def parse_usage(payload: dict, *, camel_case: bool = False, embeddings: bool = False) -> Usage:
    raw = payload.get("usage") or {}
    if not isinstance(raw, dict):
        raise ContractError("invalid_usage")

    def token(key):
        value = raw.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ContractError("invalid_usage")
        return value

    input_key = "prompt_tokens" if embeddings else "inputTokens" if camel_case else "input_tokens"
    output_key = "outputTokens" if camel_case else "output_tokens"
    cost = raw.get("cost")
    if camel_case:
        meta = payload.get("providerMetadata") or {}
        gateway = meta.get("gateway", {}) if isinstance(meta, dict) else {}
        if isinstance(gateway, dict):
            cost = gateway.get("cost", cost)
    if cost is not None:
        try:
            if isinstance(cost, bool):
                raise TypeError
            cost = float(cost)
            if not math.isfinite(cost) or cost < 0:
                raise ValueError
        except (ValueError, TypeError):
            raise ContractError("invalid_usage_cost") from None
    return Usage(token(input_key), None if embeddings else token(output_key), cost)
