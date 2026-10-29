from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from ..errors import ContractError
from ..prompts import prompt
from ..universal import universal


def confident(answer, threshold=0.85, margin=0.15):
    probabilities = answer["probabilities"]
    score = probabilities[answer["choice"]]
    second = max(v for k, v in probabilities.items() if k != answer["choice"])
    return score >= threshold and score - second >= margin


@dataclass
class EvidenceSet:
    units: dict
    statuses: dict
    reasons: dict
    conflicts: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    complete: bool = True
    tentative: list = field(default_factory=list)

    @property
    def active(self):
        return [value for key, value in self.units.items() if self.statuses[key] == "ELIGIBLE"]

    def audit(self):
        return {
            "statuses": self.statuses,
            "reasons": self.reasons,
            "provenance": {
                unit: provenance(reason)
                for unit, reason in self.reasons.items()
                if self.statuses.get(unit) == "SUPERSEDED"
            },
            "conflicts": self.conflicts,
            "unresolved": self.unresolved,
            "tentative": self.tentative,
            "complete_within_local_snapshot": self.complete,
        }


# How a supersession was established, derived from the existing reason strings (N2): by a code
# rule on source metadata, by a confident Jev judgement, or inferred (a prior under an uncertain
# judgement, or the text-overlap rule). A new state is not needed; the reason already says it.
PROVENANCE = {
    "new_effective_revision": "confirmed_code",
    "explicit_replacement": "confirmed_code",
    "semantic_replacement": "confirmed_judgement",
    "implicit_revision_prior": "inferred",
    "implicit_revision": "inferred",
}


def provenance(reason):
    return PROVENANCE.get((reason or "").split(":")[0], "other")


# What an uncertain (UNKNOWN) relation between a later same-family document and an older one it
# overlaps does when the U10 prior's conditions hold. "prior" is the adopted policy; the others
# are N2 experiment arms, chosen only by an evaluation runner (never by a request).
UNKNOWN_POLICIES = ("prior", "keep_both", "tentative")


def overlap(older, newer):
    """Text similarity of two documents (difflib ratio); the U10 revision rule's condition."""
    matcher = difflib.SequenceMatcher(None, older.text, newer.text)
    return matcher.ratio() if matcher.quick_ratio() >= 0.3 else 0.0


def valid_declaration(old, newer):
    """Whether `newer`'s explicit `supersedes` declaration may take `old`'s units out of force:
    it was published after `old`, and a primary source is replaced only by a primary one. The
    families need not match (RVI-02b, KB-FF R-REV-2: `supersedes` is what replaces another
    document); the implicit rules below still compare documents of one family only."""
    return old.published_at < newer.published_at and not (
        old.authority == "primary" and newer.authority != "primary"
    )


async def select_evidence(
    documents,
    question,
    *,
    gateway=None,
    threshold=0.85,
    margin=0.15,
    pairing="unit",
    revision_overlap=None,
    revision_prior=None,
    revision_prior_scope=None,
    unknown_policy="prior",
    validity_docs=(),
):
    """`validity_docs` (RVI-02) are revisions the principal may not see, was not shown or that
    retrieval did not return, which can still decide validity: a higher revision of a retrieved
    document (R-REV-1) or a `supersedes` declaration aimed at one (R-REV-2), applied by the same
    code rules from their metadata only. Their units never become evidence, and the reasons they
    produce do not name them (`…:withheld`)."""
    if unknown_policy not in UNKNOWN_POLICIES:
        raise ValueError("unknown_revision_policy")
    units = {e["id"]: e for doc in documents for e in doc.evidence()}
    result = EvidenceSet(units, {}, {})
    by_key = {d.key: d for d in documents}
    by_id = {}
    eligible_docs = []
    for doc in documents:
        by_id.setdefault(doc.id, []).append(doc)
        status = "ELIGIBLE"
        if doc.observed_at > question.known_at:
            status = "UNKNOWN"
        elif doc.starts > question.as_of:
            status = "FUTURE_EFFECTIVE"
        elif doc.effective_until and question.as_of >= doc.effective_until:
            status = "EXPIRED"
        else:
            eligible_docs.append(doc)
        for e in doc.evidence():
            result.statuses[e["id"]] = status
            result.reasons[e["id"]] = "effective_interval"
    # Expiration of a successor does not resurrect the previous revision.
    started_docs = [
        d for d in documents if d.observed_at <= question.known_at and d.starts <= question.as_of
    ]
    # A revision is a complete replacement; separately published amendments use supersedes.
    for doc in eligible_docs:
        if any(other.id == doc.id and other.revision > doc.revision for other in started_docs):
            for e in doc.evidence():
                result.statuses[e["id"]] = "SUPERSEDED"
                result.reasons[e["id"]] = "new_effective_revision"
    # RVI-02: the same two code rules for revisions outside the retrieved set. A declaration
    # that has not started (effective later, or observed after known_at) does not apply yet; one
    # whose declaring document was revoked or replaced still does (nothing is resurrected).
    plane = [
        d
        for d in validity_docs
        if d.key not in by_key and d.observed_at <= question.known_at and d.starts <= question.as_of
    ]
    for doc in eligible_docs:
        if any(other.id == doc.id and other.revision > doc.revision for other in plane):
            for e in doc.evidence():
                if result.statuses[e["id"]] == "ELIGIBLE":
                    result.statuses[e["id"]] = "SUPERSEDED"
                    result.reasons[e["id"]] = "new_effective_revision"
    for newer in plane:
        for replacement in newer.supersedes:
            for old in by_id.get(replacement.document_id, []):
                if not valid_declaration(old, newer):
                    result.complete = False
                    result.unresolved.append(
                        {"document": "withheld", "reason": "invalid_replacement_authority_or_order"}
                    )
                    continue
                ids = {u.id for u in old.units}
                if set(replacement.unit_ids) - ids:
                    result.complete = False
                    result.unresolved.append(
                        {"document": "withheld", "reason": "replacement_scope_unavailable"}
                    )
                    continue
                for unit, evidence in zip(old.units, old.evidence(), strict=True):
                    if (not replacement.unit_ids or unit.id in replacement.unit_ids) and (
                        result.statuses[evidence["id"]] == "ELIGIBLE"
                    ):
                        result.statuses[evidence["id"]] = "SUPERSEDED"
                        result.reasons[evidence["id"]] = "explicit_replacement:withheld"
    current_docs = [
        d
        for d in eligible_docs
        if any(result.statuses[e["id"]] == "ELIGIBLE" for e in d.evidence())
    ]
    for newer in started_docs:
        for replacement in newer.supersedes:
            targets = by_id.get(replacement.document_id, [])
            if not targets:
                result.complete = False
                result.unresolved.append(
                    {"document": newer.key, "reason": "replacement_target_unavailable"}
                )
            for old in targets:
                if not valid_declaration(old, newer):
                    result.complete = False
                    result.unresolved.append(
                        {"document": newer.key, "reason": "invalid_replacement_authority_or_order"}
                    )
                    continue
                ids = {u.id for u in old.units}
                if set(replacement.unit_ids) - ids:
                    result.complete = False
                    result.unresolved.append(
                        {"document": newer.key, "reason": "replacement_scope_unavailable"}
                    )
                    continue
                for unit, evidence in zip(old.units, old.evidence(), strict=True):
                    if (
                        not replacement.unit_ids or unit.id in replacement.unit_ids
                    ) and result.statuses[evidence["id"]] == "ELIGIBLE":
                        result.statuses[evidence["id"]] = "SUPERSEDED"
                        result.reasons[evidence["id"]] = "explicit_replacement:" + newer.key
    # U10 (Universal RVG): a later document of the same family whose text overlaps the older
    # one at least `revision_overlap` is a quiet revision of it — the old units are superseded by
    # code, without a semantic judgement (which experiment U5 found never confident on real
    # single-value edits). None (the policy default until adopted) switches the rule off.
    if revision_overlap is None:
        revision_overlap = universal()["validity"].get("implicit_revision_overlap")
    if revision_prior is None:
        revision_prior = universal()["validity"].get("implicit_revision_prior")
    if revision_prior_scope is None:
        revision_prior_scope = universal()["validity"].get("implicit_revision_prior_scope", "unit")

    def supersede_revision(old_id, old_key, new_key):
        """U10-1 marks the judged unit; U10-2 (`scope = document`) marks every still-eligible
        unit of the older document, since a revision replaces the document as a whole."""
        targets = [old_id]
        if revision_prior_scope == "document" and old_key in by_key:
            targets = [e["id"] for e in by_key[old_key].evidence()]
        for unit_id in targets:
            if result.statuses.get(unit_id) == "ELIGIBLE" or unit_id == old_id:
                result.statuses[unit_id] = "SUPERSEDED"
                result.reasons[unit_id] = "implicit_revision_prior:" + new_key

    def uncertain_revision(old_id, old_key, new_key):
        """The U10 prior's conditions hold under an uncertain judgement: apply the policy."""
        if unknown_policy == "prior":
            supersede_revision(old_id, old_key, new_key)
            return
        targets = [old_id]
        if revision_prior_scope == "document" and old_key in by_key:
            targets = [e["id"] for e in by_key[old_key].evidence()]
        for unit_id in targets:
            if result.statuses.get(unit_id) != "ELIGIBLE":
                continue
            if unknown_policy == "tentative":
                # Both stay in force and visible; the older text carries the inferred relation.
                result.units[unit_id] = {
                    **result.units[unit_id],
                    "validity_note": "tentatively_superseded_by:" + new_key,
                }
        record = {
            "older_unit": old_id,
            "newer_document": new_key,
            "reason": "semantic_uncertainty",
            "prior_conditions_met": True,
        }
        result.unresolved.append(record)
        if unknown_policy == "tentative":
            result.tentative.append(record | {"provenance": "inferred"})

    def quiet_revision(old_key, new_key):
        """U10-1: an uncertain judgement between a later same-family document and an older one
        whose text it overlaps at least `revision_prior` counts as a replacement — the overlap
        is the prior, a confident CONFLICTS/UNRELATED from the judge overrides it."""
        if not revision_prior or old_key not in by_key or new_key not in by_key:
            return False  # None after resolution, or an explicit 0, switches the prior off
        old, newer = by_key[old_key], by_key[new_key]
        return (
            old.family == newer.family
            and old.published_at < newer.published_at
            and old.authority == newer.authority
            and not newer.supersedes
            and overlap(old, newer) >= revision_prior
        )

    if revision_overlap is not None:
        for newer in current_docs:
            if newer.supersedes:
                continue
            for old in current_docs:
                if (
                    old.id == newer.id
                    or old.family != newer.family
                    or old.published_at >= newer.published_at
                    or old.authority != newer.authority
                    or overlap(old, newer) < revision_overlap
                ):
                    continue
                for evidence in old.evidence():
                    if result.statuses[evidence["id"]] == "ELIGIBLE":
                        result.statuses[evidence["id"]] = "SUPERSEDED"
                        result.reasons[evidence["id"]] = "implicit_revision:" + newer.key
    if gateway is None or not result.complete:
        return result
    pairs = []
    for newer in current_docs:
        if newer.supersedes or not any(
            result.statuses[e["id"]] == "ELIGIBLE" for e in newer.evidence()
        ):
            continue
        for evidence in result.active:
            old = by_key[f"{evidence['document_id']}@{evidence['revision']}"]
            if (
                old.id == newer.id
                or old.family != newer.family
                or old.published_at >= newer.published_at
            ):
                continue
            if old.authority == "primary" and newer.authority != "primary":
                continue
            pairs.append({"older": evidence, "newer": newer.model_dump(mode="json")})
    if len(pairs) > 128:
        raise ContractError("temporal_pair_budget_exceeded")
    if pairing == "document":  # experiment W-5: every unit of one older document at once
        return await _judge_by_document(
            result, pairs, question, gateway, threshold, margin, quiet_revision, uncertain_revision
        )
    for offset in range(0, len(pairs), 4):
        batch = pairs[offset : offset + 4]
        answers = await gateway.ask(
            "temporal_semantic_replacement",
            {"as_of": question.as_of.isoformat(), "pairs": batch},
            {
                f"r{i}": {
                    "type": "choice",
                    "instructions": f"{prompt('temporal.gate.select_evidence.instructions.head')}{i}{prompt('temporal.gate.select_evidence.instructions.tail')}",
                    "criteria": {
                        "REPLACES": prompt("temporal.gate.select_evidence.criteria.REPLACES"),
                        "CONFLICTS": prompt("temporal.gate.select_evidence.criteria.CONFLICTS"),
                        "UNRELATED": prompt("temporal.gate.select_evidence.criteria.UNRELATED"),
                        "UNKNOWN": prompt("temporal.gate.select_evidence.criteria.UNKNOWN"),
                    },
                }
                for i in range(len(batch))
            },
        )
        for i, pair in enumerate(batch):
            answer = answers[f"r{i}"]
            old_id = pair["older"]["id"]
            new_key = pair["newer"]["id"] + "@" + str(pair["newer"]["revision"])
            relation = answer["choice"] if confident(answer, threshold, margin) else "UNKNOWN"
            old_key = f"{pair['older']['document_id']}@{pair['older']['revision']}"
            if relation == "UNKNOWN" and quiet_revision(old_key, new_key):
                uncertain_revision(old_id, old_key, new_key)
            elif relation == "REPLACES":
                result.statuses[old_id] = "SUPERSEDED"
                result.reasons[old_id] = "semantic_replacement:" + new_key
            elif relation == "CONFLICTS":
                result.conflicts.append({"older_unit": old_id, "newer_document": new_key})
            elif relation == "UNKNOWN":
                result.unresolved.append(
                    {
                        "older_unit": old_id,
                        "newer_document": new_key,
                        "reason": "semantic_uncertainty",
                    }
                )
    return result


DOCUMENT_PAIRING = prompt("temporal.gate.DOCUMENT_PAIRING")


async def _judge_by_document(
    result,
    pairs,
    question,
    gateway,
    threshold,
    margin,
    quiet_revision=lambda a, b: False,
    supersede_revision=None,
):
    """The same four relations, asked once per (older document, newer document) for all its units."""
    groups = {}
    for pair in pairs:
        older = pair["older"]
        key = (f"{older['document_id']}@{older['revision']}", pair["newer"]["id"])
        groups.setdefault(key, (pair["newer"], []))[1].append(older)
    for (old_key, _), (newer, units) in groups.items():
        state = {
            "as_of": question.as_of.isoformat(),
            "older": {
                "id": old_key,
                "title": units[0]["title"],
                "units": [{"id": u["id"], "text": u["text"]} for u in units],
            },
            "newer": {"id": newer["id"], "title": newer["title"], "text": newer["text"]},
        }
        criteria = {
            "REPLACES": prompt("temporal.gate._judge_by_document.criteria.REPLACES"),
            "CONFLICTS": prompt("temporal.gate._judge_by_document.criteria.CONFLICTS"),
            "UNRELATED": prompt("temporal.gate._judge_by_document.criteria.UNRELATED"),
            "UNKNOWN": prompt("temporal.gate._judge_by_document.criteria.UNKNOWN"),
        }
        answers = await gateway.ask(
            "temporal_semantic_replacement:document",
            state,
            {
                f"r{i}": {
                    "type": "choice",
                    "instructions": f"{prompt('temporal.gate._judge_by_document.instructions.head')}{i}{prompt('temporal.gate._judge_by_document.instructions.tail')}"
                    + DOCUMENT_PAIRING,
                    "criteria": criteria,
                }
                for i in range(len(units))
            },
        )
        new_key = newer["id"] + "@" + str(newer["revision"])
        for i, unit in enumerate(units):
            answer = answers[f"r{i}"]
            relation = answer["choice"] if confident(answer, threshold, margin) else "UNKNOWN"
            if relation == "UNKNOWN" and quiet_revision(old_key, new_key):
                if supersede_revision is not None:
                    supersede_revision(unit["id"], old_key, new_key)
                else:
                    result.statuses[unit["id"]] = "SUPERSEDED"
                    result.reasons[unit["id"]] = "implicit_revision_prior:" + new_key
            elif relation == "REPLACES":
                result.statuses[unit["id"]] = "SUPERSEDED"
                result.reasons[unit["id"]] = "semantic_replacement:" + new_key
            elif relation == "CONFLICTS":
                result.conflicts.append({"older_unit": unit["id"], "newer_document": new_key})
            elif relation == "UNKNOWN":
                result.unresolved.append(
                    {
                        "older_unit": unit["id"],
                        "newer_document": new_key,
                        "reason": "semantic_uncertainty",
                    }
                )
    return result
