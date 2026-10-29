"""The handoff packet (PROD-1, opt-in, parent docs/PROD_1_PROTOCOL.md §2): what a person who takes
over a request the controller did not release needs to decide it — the held drafts with their
citations, why they were held, which claim failed in which sentence, the evidence units with
their validity status, how a second draft compared, and why a writer declined.

The packet is not a release. It is attached as `result.handoff` only to results that are not
`FULL` or `PARTIAL`, says so in `kind` and `released`, and never enters a release record. It shows
a unit's text only when the requester may see that unit now: its document passes the release
check (not revoked, visible to the principal), is not on the deny list, is inside the selection
when there is one, and the validity gate found the unit ELIGIBLE. Every text a stage wrote —
draft, claims, the writer's own output — is withheld when any unit that stage's writer saw is
withheld (the release check's own unit: the writing window). Nothing here calls a model or
writes to a store."""

from __future__ import annotations

VERSION = "handoff-v1"
KIND = "handoff_draft"
# The wording every automatic flag carries (parent docs/PROD_1_PROTOCOL.md §2, rule ⑥): the
# measured precision of the flags was below one half (parent docs/PROD_1_REPORT.md, (b): 14.5%,
# Wilson 95% 10.3-20.1%, 200 flags), so the default says the flag is of low confidence.
FLAG_NOTE = "automatic flag, low confidence"
RELEASED = ("FULL", "PARTIAL")
# The release-time reasons of the engine, as packet reason codes.
RELEASE_REASONS = {
    "revoked_before_release": "release:blocked_documents",
    "permission_changed_before_release": "release:blocked_documents",
    "source_changed": "release:source_changed",
    "scope_changed": "release:scope_changed",
    "access_changed": "release:access_changed",
    "stale_after_retry": "release:stale_after_retry",
    "retry_budget_exhausted": "release:retry_budget_exhausted",
}
SCOPE_REASONS = ("scope_unresolved", "scope_top1_disagree")
# The draft a person is handed: the last written stage that has one (the second stage's evidence
# is the wider), never the partner draft, which only checks the second.
HANDED = ("stage2", "stage1", "stage1_top1")


def _document(unit_id):
    return str(unit_id).split("#")[0].split("@")[0]


def _gates(audits):
    out = {}
    for audit in audits or []:
        for unit, status in ((audit.get("gate") or {}).get("statuses") or {}).items():
            out.setdefault(unit, set()).add(status)
    return out


class Sight:
    """What the requester may see now, per unit (the masking rule of protocol §2)."""

    def __init__(self, *, gates, revoked, hidden, excluded, allowed, unknown=None):
        self.gates, self.revoked, self.hidden = gates, revoked, hidden
        self.excluded, self.allowed, self.unknown = excluded, allowed, unknown

    def withheld(self, unit_id, document_id=None):
        doc = document_id or _document(unit_id)
        if self.unknown:
            return self.unknown
        if self.excluded and doc in self.excluded:
            return "excluded"
        if self.allowed is not None and doc not in self.allowed:
            return "not_selected"
        if doc in self.revoked:
            return "revoked"
        if doc in self.hidden:
            return "not_visible"
        gates = self.gates.get(unit_id) or set()
        if gates != {"ELIGIBLE"}:
            return "gate:" + ("|".join(sorted(gates)) or "unknown")
        return None

    def gate(self, unit_id):
        gates = sorted(self.gates.get(unit_id) or ())
        return gates[0] if len(gates) == 1 else ("|".join(gates) or None)


def sight(release, *, audits, hidden=None, excluded=None, allowed=None, unknown=None):
    """`release` is the release check's answer for every document in the packet (read-only),
    `hidden` the principal's hidden ids when the check did not give them; `unknown` withholds
    every unit (the check could not be read)."""
    revoked, seen = set(), set(hidden or ())
    if release is not None:
        if release.reason == "revoked_before_release":
            revoked = set(release.blocked)
        elif release.reason is not None:
            seen |= set(release.blocked)
        seen |= set(release.hidden or ())
    return Sight(
        gates=_gates(audits),
        revoked=revoked,
        hidden=seen,
        excluded=frozenset(excluded or ()),
        allowed=None if allowed is None else frozenset(allowed),
        unknown=unknown,
    )


def documents(stages):
    """Every document a packet may show a unit of (for the one release check)."""
    out = set()
    for entry in stages:
        for unit in entry.get("units") or []:
            out.add(unit.get("document_id") or _document(unit["id"]))
        for sentence in ((entry.get("writer") or {}).get("raw") or {}).get("sentences") or []:
            out.update(_document(u) for u in sentence.get("evidence_ids") or [])
    return out


def _withheld_stage(entry, view):
    """Why the stage's own texts are withheld: a unit its writer saw is withheld, or its writer
    cited a unit outside them."""
    units = entry.get("units") or []
    hidden = [u["id"] for u in units if view.withheld(u["id"], u.get("document_id"))]
    if hidden:
        return {"withheld": "window_has_withheld_unit", "units": hidden}
    window = {u["id"] for u in units}
    raw = (entry.get("writer") or {}).get("raw") or {}
    outside = sorted(
        {i for s in raw.get("sentences") or [] for i in s.get("evidence_ids") or []} - window
    )
    if outside:
        return {"withheld": "cites_unit_outside_window", "units": outside}
    return None


def _draft(draft, masked):
    if draft is None:
        return None
    if masked:
        return masked | {
            "evidence_ids": sorted(
                {i for s in draft.get("sentences") or [] for i in s.get("evidence_ids") or []}
            )
        }
    return {
        "short_answer": draft.get("short_answer"),
        "sentences": [
            {"text": s.get("text"), "evidence_ids": list(s.get("evidence_ids") or [])}
            for s in draft.get("sentences") or []
        ],
    }


def _control(entry, masked):
    control = entry.get("control")
    if control is None:
        return None
    if entry["stage"] == "conflict_rewrite" and "labels" in control:
        claims = [
            {"text": text, "sentence": i, "label": verdict}
            for i, (text, verdict) in enumerate(
                zip(control.get("claims") or [], control["labels"], strict=True)
            )
        ]
        control = {"claims": claims, "scope": control.get("scope")}
    if masked and control.get("claims"):
        control = {
            **control,
            "claims": [
                {k: v for k, v in claim.items() if k != "text"} | {"text": masked}
                for claim in control["claims"]
            ],
        }
    return control


def stage_reasons(entry):
    """The reason codes one written stage contributes (protocol §2 table)."""
    out = []
    writer = entry.get("writer") or {}
    if writer.get("refused"):
        out.append("writer_refusal")
    if entry.get("reason") in ("no_active_evidence", "evidence_not_answerable"):
        out.append(entry["reason"])
    control = entry.get("control") or {}
    if "error" in control:
        out.append("error:" + str(control["error"]))
    elif control.get("blocked"):
        out.append("draft_blocked:" + str(control["blocked"]))
    elif control.get("claims") is not None and entry["stage"] != "conflict_rewrite":
        if any(c.get("label") != "SUPPORTED" for c in control["claims"]):
            out.append("claim_not_supported")
        elif not control.get("eligible"):
            out.append("dependency_or_coverage")
        for name, signal in (control.get("signals") or {}).items():
            if not signal.get("passed"):
                out.append("profile:" + name)
        if control.get("certified_kind") is False:
            out.append("profile:kind_not_certified")
    elif (
        entry["stage"] == "conflict_rewrite"
        and entry.get("status") not in RELEASED
        and control.get("claims") is not None
    ):
        out.append("claim_not_supported")
    comparison = entry.get("comparison") or {}
    if comparison and not comparison.get("agreed"):
        out.append("consistency_disagreement")
    if entry.get("status") == "ERROR" and not any(r.startswith("error:") for r in out):
        out.append("error:" + str(entry.get("reason")))
    return out


def result_reasons(result):
    out = []
    reason = result.get("reason")
    if result.get("status") == "ERROR":
        error = result.get("error") or {}
        codes = [error.get("code"), *(error.get("codes") or [])]
        out += ["error:" + str(c) for c in codes if c]
        if not out:
            out.append("error:" + str(reason))
    if reason in SCOPE_REASONS:
        out.append(reason)
    if reason in RELEASE_REASONS:
        out.append(RELEASE_REASONS[reason])
    return out


def _flags(stage, control):
    out = []
    claims = (control or {}).get("claims") or []
    if stage in ("partner_draft", "conflict_rewrite"):
        return out
    failing = {}
    for i, claim in enumerate(claims):
        if claim.get("label") != "SUPPORTED":
            failing.setdefault(claim.get("sentence"), []).append(i)
    for sentence, which in sorted(failing.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        out.append(
            {
                "stage": stage,
                "sentence": sentence,
                "kind": "claim_not_supported",
                "detail": {"claims": which, "labels": [claims[i].get("label") for i in which]},
                "automatic": True,
                "note": FLAG_NOTE,
            }
        )
    answered = ((control or {}).get("signals") or {}).get("answered")
    if answered and not answered.get("passed"):
        out.append(
            {
                "stage": stage,
                "sentence": None,
                "kind": "answered_below",
                "detail": {"score": answered.get("score"), "threshold": answered.get("threshold")},
                "automatic": True,
                "note": FLAG_NOTE,
            }
        )
    return out


def _sentences(draft):
    return [
        (s.get("text"), tuple(s.get("evidence_ids") or ()))
        for s in (draft or {}).get("sentences") or []
    ]


def packet(result, stages, view):
    """The packet for a result that was not released; `stages` is the cascade's handoff trail and
    `view` the requester's `Sight` at this instant."""
    out_stages, by_name, masks, units, flags, reasons = [], {}, {}, {}, [], []
    for entry in stages:
        masked = _withheld_stage(entry, view)
        masks[entry["stage"]] = masked
        writer = entry.get("writer")
        if writer is not None and writer.get("raw") is not None and masked:
            writer = {**writer, "raw": masked}
        own = stage_reasons(entry)
        control = _control(entry, masked)
        item = {
            "stage": entry["stage"],
            "evidence_ids": list(entry.get("evidence_ids") or []),
            "writer": writer,
            "draft": _draft(entry.get("draft"), masked),
            "control": control,
            "status": entry.get("status"),
            "reason": entry.get("reason"),
            "reasons": own,
        }
        if entry.get("draft_before_recovery") is not None:
            item["draft_before_recovery"] = _draft(entry["draft_before_recovery"], masked)
        if entry.get("comparison") is not None:
            item["comparison"] = dict(entry["comparison"])
        out_stages.append(item)
        by_name.setdefault(entry["stage"], entry)
        for unit in entry.get("units") or []:
            units.setdefault(unit["id"], unit)
        flags += _flags(entry["stage"], entry.get("control"))
        for code in own:
            if code not in reasons:
                reasons.append(code)
    for code in reversed(result_reasons(result)):
        if code in reasons:
            reasons.remove(code)
        reasons.insert(0, code)
    handed = next(
        (name for name in HANDED if (by_name.get(name) or {}).get("draft") is not None), None
    )
    comparison = None
    if "stage2" in by_name:
        first, second = by_name.get("stage1") or {}, by_name["stage2"]
        mate = by_name.get("partner_draft") or {}
        short = lambda e: (
            None
            if e.get("draft") is None
            else (masks.get(e["stage"]) or (e["draft"].get("short_answer")))
        )
        rule = mate.get("comparison") or {}
        before, after = _sentences(first.get("draft")), _sentences(second.get("draft"))
        hidden = masks.get("stage1") or masks.get("stage2")
        listed = lambda rows: hidden or [{"text": t, "evidence_ids": list(i)} for t, i in rows]
        comparison = {
            "first": short(first) if first else None,
            "second": short(second),
            "partner": short(mate) if mate else None,
            "judge": rule.get("judge"),
            "theta": rule.get("theta"),
            "agreed": rule.get("agreed"),
            "sentences_added": listed([r for r in after if r[0] not in {t for t, _ in before}]),
            "sentences_dropped": listed([r for r in before if r[0] not in {t for t, _ in after}]),
        }
    evidence = []
    for unit_id, unit in units.items():
        why = view.withheld(unit_id, unit.get("document_id"))
        if why:
            evidence.append({"id": unit_id, "gate": view.gate(unit_id), "withheld": why})
        else:
            evidence.append(
                {
                    "id": unit_id,
                    "title": unit.get("title"),
                    "text": unit.get("text"),
                    "published_at": unit.get("published_at"),
                    "effective_from": unit.get("effective_from"),
                    "effective_until": unit.get("effective_until"),
                    "revision": unit.get("revision"),
                    "gate": view.gate(unit_id),
                }
            )
    return {
        "kind": KIND,
        "released": False,
        "version": VERSION,
        "status": result.get("status"),
        "reason": result.get("reason"),
        "reasons": reasons,
        "stages": out_stages,
        "handed_draft": handed,
        "comparison": comparison,
        "evidence": evidence,
        "flags": flags,
        "premise": "not_checked",
    }
