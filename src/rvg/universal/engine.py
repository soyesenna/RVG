"""The one product path (Universal RVG): time-scoped retrieval → code validity rules → automatic
scope → writer → the fixed controller → one bounded cascade step → FULL / PARTIAL / ABSTAIN /
UNRESOLVED_SCOPE. Every constant comes from `rvg/resources/policies/universal.json`; nothing here is chosen per
knowledge base, language or request."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from ..control import cascade
from ..control.controller import AnswerController, ControlPolicy
from ..control.verify import Asker
from ..errors import ProviderError
from ..gateway import DecisionGateway
from ..storage.segments import unit_ranks
from ..temporal.gate import select_evidence
from . import cache as cache_store
from . import handoff as handoff_packet
from . import scoped as scope_fence
from .config import universal, universal_hash
from .trace import HELD_DRAFTS, node

_U = universal()


def validity_block(arm=None):
    """The policy's validity block, with an experiment arm's overrides when one is given."""
    return {**(_U.get("validity") or {}), **(dict(arm.validity) if arm is not None else {})}


def validity_rules(arm=None):
    """The policy's deterministic validity rules as `select_evidence` keyword arguments (U10:
    the implicit-revision prior and its scope; the hard overlap rule stays off unless set; N2:
    what an uncertain relation does when the prior's conditions hold, "prior" by default)."""
    rules = validity_block(arm)
    return {
        "pairing": rules.get("pairing", "unit"),
        "revision_overlap": rules.get("implicit_revision_overlap"),
        "revision_prior": rules.get("implicit_revision_prior"),
        "revision_prior_scope": rules.get("implicit_revision_prior_scope", "unit"),
        "unknown_policy": rules.get("unknown_revision", "prior"),
    }


def prefer_latest_revision(ranked):
    """U10-b: when several revisions of one family are all in force (the gate could not tell
    them apart), that family's newest revision comes first in the writer's order, so a writer
    that sees only the top of the list never sees the older text alone. Other units keep their
    retrieval order (stable sort); the gate's validity decisions are untouched."""
    stamp = lambda e: (e.get("published_at") or "", e.get("revision") or 0)
    revisions, newest = {}, {}
    for e in ranked:
        revisions.setdefault(e["family"], set()).add(stamp(e))
        newest[e["family"]] = max(newest.get(e["family"], stamp(e)), stamp(e))
    older = lambda e: len(revisions[e["family"]]) > 1 and stamp(e) != newest[e["family"]]
    return sorted(ranked, key=older)


def interleave_revisions(ranked):
    """U10-c: when several revisions of one family are all in force, their units are
    interleaved (newest revision first, then the next, round-robin, each in retrieval order) at
    the position of the family's first unit, so that any prefix of the list a writer sees holds
    both revisions and a conflict between them is visible rather than resolved by order. Other
    units keep their order; the gate's validity decisions are untouched."""
    stamp = lambda e: (e.get("published_at") or "", e.get("revision") or 0)
    by_family = {}
    for e in ranked:
        by_family.setdefault(e["family"], {}).setdefault(stamp(e), []).append(e)
    mixed, emitted = {}, set()
    for family, groups in by_family.items():
        if len(groups) < 2:
            continue
        queues = [list(groups[s]) for s in sorted(groups, reverse=True)]
        order = []
        while any(queues):
            for queue in queues:
                if queue:
                    order.append(queue.pop(0))
        mixed[family] = order
    out = []
    for e in ranked:
        if e["family"] not in mixed:
            out.append(e)
        elif e["family"] not in emitted:
            emitted.add(e["family"])
            out.extend(mixed[e["family"]])
    return out


def rank_units(active, by_unit, order):
    """Units by (seed rank, segment score) and, for ties — one document's units under the same rank
    and score — in document order: the revision, then the character offset. The unit id string
    put u10 before u2 and could leave out the next paragraph of a short note (N6-2,
    parent docs/N6_2_TECHQA_PROTOCOL.md §11)."""
    return sorted(
        active,
        key=lambda e: (
            *by_unit.get(
                e["id"], (order.get(f"{e['document_id']}@{e['revision']}", len(order)), 0.0)
            ),
            e["document_id"],
            e["revision"],
            e.get("start") or 0,
            e["id"],
        ),
    )


def withhold_drafts(result):
    """0.1.0 (parent PUB-2): a result that releases nothing (ABSTAIN, UNRESOLVED_SCOPE, ERROR)
    shows no value of a draft the controller did not let through — its short answer, the claim
    sentences of its assessments, a paired draft's short answer in the trace, any text or
    sentence. The draft stays in the opt-in handoff packet (and a cache entry) built before this.
    An evaluation that set `HELD_DRAFTS` receives what was removed; the result never does."""
    out, held = dict(result), {}
    if out.get("short_answer") is not None:
        held["short_answer"], out["short_answer"] = out["short_answer"], None
    for key, empty in (("text", ""), ("sentences", [])):
        if out.get(key):
            held[key], out[key] = out[key], empty
    for key in ("assessment", "assessment_before_recovery"):
        part = out.get(key)
        if isinstance(part, dict) and part.get("claims"):
            held[f"{key}.claims"], out[key] = part["claims"], {**part, "claims": []}
    trace = out.get("trace")
    paired = (trace or {}).get("consistency") or {}
    if paired.get("partner_short_answer") is not None:
        held["trace.consistency.partner_short_answer"] = paired["partner_short_answer"]
        out["trace"] = {**trace, "consistency": {**paired, "partner_short_answer": None}}
    collector = HELD_DRAFTS.get()
    if collector is not None and held:
        collector.append(held)
    return out


class UniversalEngine:
    def __init__(
        self,
        *,
        retriever,
        decision,
        writer,
        extractor,
        arm=None,
        handoff_packet=False,
        decision_cache=None,
    ):
        """`arm` is an experiment arm (`rvg.universal.experiment.Arm`) that only an
        evaluation runner passes; the product entry points never do. `handoff_packet` (PROD-1,
        opt-in, off by default) attaches `result.handoff` to results that are not released
        (`rvg.universal.handoff`); released results and their release records do not change.
        `decision_cache` (PROD-2, opt-in, off by default) is a `rvg.universal.cache.CacheMode`:
        it looks a question up before writing and/or records the decision after answering."""
        self.retriever, self.decision = retriever, decision
        self.writer, self.extractor = writer, extractor
        self.arm = arm
        self.handoff_packet = bool(handoff_packet)
        self.decision_cache = decision_cache
        answered = dict(arm.cascade).get("answered_threshold") if arm is not None else None
        self.controller = AnswerController(
            decision,
            extractor,
            ControlPolicy() if answered is None else ControlPolicy(answered=float(answered)),
        )

    def release_block(self):
        """U24: the release settings — the arm's block when it names any release key (the keys it
        leaves out at their values before 0.1.0.dev27: `current`, no retry, no vector placement),
        else the policy's `release` block (scoped dependency, vectors placed, vector moves after
        the check, one `slack` retry since 0.1.0.dev27, parent docs/UNIVERSAL_U24_REPORT.md).
        The block is taken whole, never key by key, so an arm stays a valid combination."""
        if self.arm is not None and self.arm.release:
            return dict(self.arm.release)
        return dict(_U.get("release") or {})

    def release_mode(self):
        return self.release_block().get("mode", "current")

    def release_retry(self):
        """RVI-03: semantic retries after a change at the fence — 0 or 1 (the policy's since
        0.1.0.dev27), for `current` and `scoped_dependency` (latest_at_release has its own single
        retry)."""
        return int(self.release_block().get("retry", 0))

    def release_retry_rule(self):
        """RVI-03 F3: when a scoped retry regenerates — "any", "slack" (the policy's since
        0.1.0.dev27) or "cited"."""
        return self.release_block().get("retry_rule", "any")

    def release_revalidate(self):
        """RVI-03 F3: revalidate the first draft's inputs before any regeneration (arm only)."""
        return bool(self.release_block().get("revalidate", False))

    def release_vector_moves(self):
        """FIX-V9-CHURN: "hold" (F2's V9) or "after_check" (the policy's since 0.1.0.dev27)."""
        return self.release_block().get("vector_moves", "hold")

    def release_acl_moves(self):
        """FIX-ACL-WINDOW: "hold" (the policy's before 0.1.0.dev42; an arm may still choose it) or
        "after_check" (the policy's since 0.1.0.dev42, parent docs/UNIVERSAL_U32_REPORT.md)."""
        return self.release_block().get("acl_moves", "hold")

    def fence_hidden(self, auth):
        """FIX-ACL-WINDOW: the principal's hidden ids as of the fence when this arm orders ACL
        rows after the check (`acl_moves = after_check`) and the store gave them; else None (the
        check reads the state, as before)."""
        if self.release_acl_moves() != "after_check":
            return None
        return getattr(auth, "hidden", None)

    def release_vectors(self):
        """RVI-03 F2: whether the scoped fence places vector writes by relatedness — "ignore"
        (the scoped fence as before) or "scoped" (the policy's since 0.1.0.dev27)."""
        return self.release_block().get("vectors", "ignore")

    def conflict_partial(self):
        """N4 (arm only): what a draft the floor blocked for conflict alone gets — "off" (the
        policy) or "rewrite" (once in the conflict form, judged on its cited units)."""
        return dict(self.arm.cascade).get("conflict_partial", "off") if self.arm else "off"

    def writer_schema(self):
        """N5 (arm only): the writing contract — "W0" (the policy's), "W1" structured, "W2"
        structured with quotes bound to the unit text."""
        return dict(self.arm.cascade).get("writer_schema", "W0") if self.arm else "W0"

    def writer_contract(self):
        """N6-2 (arm only): "W0" (the policy's writing instruction) or "form-v1" (the answer form —
        value, list or procedure — decided from the question and the evidence)."""
        return dict(self.arm.cascade).get("writer_contract", "W0") if self.arm else "W0"

    def evidence_window(self):
        """N6-2: "neighbors-v1" (each writing unit with its neighbours in the same revision that
        passed the gate; the policy's since 0.1.0.dev17, parent docs/N6_2_TECHQA_REPORT.md) or
        "none" (the writing pool alone, the policy's before; an arm may still choose it)."""
        policy = _U["cascade"].get("evidence_window", "none")
        return dict(self.arm.cascade).get("evidence_window", policy) if self.arm else policy

    def consistency_partner(self):
        """U22: the evidence of the second-stage partner draft — "widened" (the second-stage
        evidence in reverse; the policy's since 0.1.0.dev18, parent docs/UNIVERSAL_U22_REPORT.md)
        or "first" (the first evidence in reverse, U8; an arm may still choose it)."""
        rule = _U["cascade"].get("consistency") or {}
        policy = "widened" if rule.get("partner") == "widened" else "first"
        return dict(self.arm.cascade).get("consistency_partner", policy) if self.arm else policy

    def consistency_judge(self):
        """U22: how the second-stage and partner short answers are compared — "same_text" (the
        policy's since 0.1.0.dev18), "identical_text" (U8), "normalized", or "off" (no partner
        draft; the second stage is released as it is)."""
        policy = (_U["cascade"].get("consistency") or {}).get("judge", "identical_text")
        return dict(self.arm.cascade).get("consistency_judge", policy) if self.arm else policy

    def write_top(self):
        """U23 (arm only): the first stage's writing units; None is the policy's."""
        return dict(self.arm.cascade).get("write_top") if self.arm is not None else None

    def scope_top1(self):
        """U30: what to do when the scope leaves the top-ranked unit's document out — "off" (the
        policy's before 0.1.0.dev42; an arm may still choose it), "entity-tie-v1" (arm only) or
        "dual-draft-v1" (the policy's since 0.1.0.dev42, parent docs/UNIVERSAL_U32_REPORT.md;
        `rvg.control.cascade.run`)."""
        policy = _U["cascade"].get("scope_top1", "off")
        return dict(self.arm.cascade).get("scope_top1", policy) if self.arm else policy

    async def units(self, tenant, question, text, k, context, gateway, *, at=None, reads=None):
        """Ranked evidence units in force at `question.as_of`, for a search text that may differ
        from the question (the cascade's gap queries). Incomplete validity closure yields none.
        `at` restricts the search to a knowledge snapshot (N3 snapshot release modes). `reads`,
        when given (RVI-03 scoped fence), receives what this search read."""
        probe = question.model_copy(update={"text": text[:2000]})
        snapshot_view = {"at": at} if at is not None else {}
        with node("retrieve", {"text": probe.text, "k": k}) as n:
            docs, retrieval = await self.retriever.search(
                tenant,
                probe,
                context,
                strategy=_U["retrieval"]["strategy"],
                top_k=k,
                expand=True,
                **snapshot_view,
            )
            n.output(retrieval.get("seed_ids"))
        with node("validity_gate", [d.key for d in docs]) as n:
            # RVI-02: validity is decided over the authoritative state, content by the ACL.
            find = getattr(self.retriever, "validity_documents", None)
            plane = await asyncio.to_thread(find, tenant, docs, at=at) if find else []
            evidence = await select_evidence(
                docs,
                question,
                gateway=gateway,
                threshold=_U["floor"]["threshold"],
                margin=_U["floor"]["margin"],
                validity_docs=plane,
                **validity_rules(self.arm),
            )
            n.output(lambda: evidence.audit())
        if reads is not None:
            read = [*docs, *plane]
            reads.append(
                scope_fence.Read(
                    text=probe.text,
                    k=k,
                    ids=frozenset(d.id for d in read),
                    families=frozenset(d.family for d in read),
                    retrieval=retrieval,
                    vector=getattr(self.retriever, "_query_vectors", {}).get(probe.text),
                    keys=tuple(sorted(d.key for d in docs)),
                    plane=tuple(sorted(d.key for d in plane)),
                )
            )
        if not evidence.complete:
            return [], evidence, retrieval
        order = {key: rank for rank, key in enumerate(retrieval["seed_ids"])}
        by_unit = unit_ranks(docs, retrieval["seed_ids"], probe.text)  # K8-1: segment rows
        ranked = rank_units(evidence.active, by_unit, order)
        rules = validity_block(self.arm)
        if rules.get("prefer_latest_revision"):
            ranked = prefer_latest_revision(ranked)
        if rules.get("coexisting_revisions") == "interleave":
            ranked = interleave_revisions(ranked)
        return ranked, evidence, retrieval

    async def answer(self, tenant, question, context, *, _retried=False):
        storage = self.retriever.storage
        store = storage.state
        mode = self.release_mode()
        scoped = mode == "scoped_dependency"
        # RVI-03: read before the snapshot, so every later change is inside the checked range.
        start = scope_fence.begin(store, tenant) if scoped else None
        # The tenant's ACL epoch, read before the knowledge snapshot: every ACL change committed
        # after it is seen by the release check below (release-input consistency, ACL defect).
        access = store.acl_epoch(tenant)
        clock = getattr(context, "clock", time.monotonic)
        began = clock()
        snapshot = store.snapshot(tenant)
        at = snapshot if mode not in ("current", "scoped_dependency") else None
        gateway = DecisionGateway(self.decision, context)
        audits, families, read = [], set(), set()
        reads = [] if scoped else None
        cache = self.decision_cache
        recording = cache is not None and cache.record
        # PROD-1/2: the stage trail, only when the handoff packet or the cache's record is on
        stages = [] if (self.handoff_packet or recording) else None
        arm_stamp = self.arm.stamp() if self.arm is not None else None

        async def lookup(first, second_of):
            found = await cache_store.find(
                cache,
                asker=Asker(self.decision, context),
                question=question,
                first=first,
                second_of=second_of,
                tenant=tenant,
                arm=arm_stamp,
                snapshot=snapshot,
            )
            return found, cache

        async def retrieve(text, k):
            ranked, evidence, retrieval = await self.units(
                tenant, question, text, k, context, gateway, at=at, reads=reads
            )
            families.update(u.get("family") for u in evidence.units.values())
            read.update(u.get("document_id") for u in evidence.units.values())
            audits.append({"query": text, "gate": evidence.audit(), "retrieval": retrieval})
            return ranked

        try:
            with node("answer", {"tenant": tenant, "question": question.model_dump(mode="json")}):
                result = await cascade.run(
                    question,
                    retrieve=retrieve,
                    writer=self.writer,
                    controller=self.controller,
                    asker=Asker(self.decision, context),
                    context=context,
                    extractor=self.extractor,
                    conflict_partial=self.conflict_partial(),
                    writer_schema=self.writer_schema(),
                    writer_contract=self.writer_contract(),
                    evidence_window=self.evidence_window(),
                    consistency_partner=self.consistency_partner(),
                    consistency_judge=self.consistency_judge(),
                    write_top=self.write_top(),
                    **({"scope_top1": self.scope_top1()} if self.scope_top1() != "off" else {}),
                    **({"handoff": stages} if stages is not None else {}),
                    **({"cache": lookup} if cache is not None and cache.lookup else {}),
                )
        except ProviderError as exc:
            # A failed store, provider, schema or budget is ERROR, never a silent ABSTAIN; the
            # stage it failed in is not guessed (the trace's failed node shows it).
            result = {
                **cascade.abstain(exc.code),
                "status": "ERROR",
                "stage": None,
                "error": {
                    "type": type(exc).__name__,
                    "code": exc.code,
                    "http_status": exc.status,
                    "detail": exc.detail,
                },
            }
        failed = [
            (n, (result.get("trace") or {}).get(key) or {})
            for n, key in ((1, "first"), (2, "second"))
            if ((result.get("trace") or {}).get(key) or {}).get("status") == "ERROR"
        ]
        if result["status"] == "ABSTAIN" and failed:
            # A stage failed and nothing was released: that is a failure, not an abstention.
            # Nothing released changes; only the status names what happened.
            stage, _ = failed[0]
            result = {
                **result,
                "status": "ERROR",
                "abstain_reason": result.get("reason"),
                "reason": "stage_failed",
                "error": {"stage": stage, "codes": [s.get("reason") for _, s in failed]},
            }
        changed = None  # RVI-03: why the fence withheld a draft over a change (retry arms)
        fence = verdict = draft = None
        if result["status"] in ("FULL", "PARTIAL"):
            # N3 (all modes): at the release instant none of the answer's documents is revoked
            # and the principal may see all of them; the check is the linearisation point. It
            # reads the knowledge number and the ACL epoch in the same read; a change committed
            # after it is ordered after the release (bytes delivered later are not recalled).
            documents = {u.split("@")[0] for u in result.get("evidence_ids") or []}
            auth = fence = storage.release.authorize(tenant, documents, self.retriever.principal)
            if not auth.ok:
                result = {
                    **cascade.abstain(auth.reason),
                    "stage": result.get("stage", 0),
                    "blocked_documents": list(auth.blocked),
                }
            else:
                result = {
                    **result,
                    "release": {
                        "mode": mode,
                        "knowledge_snapshot": snapshot,
                        "revocation_seq": auth.revocation_seq,
                        "acl_epoch": auth.acl_epoch,
                    },
                }
                cached = (result.get("trace") or {}).get("cache") or {}
                if cached.get("outcome") == "released":  # PROD-2: a release from the cache says so
                    result["release"]["source"] = {
                        "cache": {k: cached.get(k) for k in ("entry", "p", "theta", "match")}
                    }
                if scoped:
                    # RVI-03: only changes that could alter what this attempt read withhold it;
                    # what cannot be mapped falls back to the global rule.
                    vectors = (
                        await self.vector_writes(tenant, question, reads, start, auth)
                        if self.release_vectors() == "scoped"
                        else None
                    )
                    verdict = self.scope_verdict(
                        tenant, question, reads, start, snapshot, auth, vectors=vectors
                    )
                    if verdict.status == scope_fence.CLEAR:
                        result["release"] = {**result["release"], "scope": verdict.receipt()}
                    else:
                        draft = result  # RVI-03 F3: what a revalidation may still release
                        changed = (
                            "scope_changed"
                            if verdict.status == scope_fence.RELATED
                            else "source_changed"
                        )
                        result = {
                            **cascade.abstain(changed),
                            "stage": result.get("stage", 0),
                            "scope": verdict.receipt(),
                        }
        if mode == "current" and result["status"] != "ERROR" and store.snapshot(tenant) != snapshot:
            # The sources moved while the answer was being made: nothing is released.
            result = {**cascade.abstain("source_changed"), "stage": result.get("stage", 0)}
            changed = "source_changed"
        elif (
            mode in ("current", "as_of_snapshot")
            and result["status"] in ("FULL", "PARTIAL")
            and fence.acl_epoch != access
        ):
            # The tenant's ACL moved while the answer was being made: what its searches could
            # see is not what the principal sees at the release instant. Whether the documents
            # that appeared or disappeared were cited is not asked; nothing is released. (A
            # snapshot answer cannot be made to hold for the ACL at S either: ACL has no history.)
            result = {**cascade.abstain("access_changed"), "stage": result.get("stage", 0)}
        elif (
            mode == "latest_at_release"
            and result["status"] in ("FULL", "PARTIAL")
            and await self.moved(
                tenant,
                question,
                context,
                snapshot,
                families,
                audits,
                access=access,
                fence=fence,
                read=read,
            )
        ):
            if _retried:  # at most one fresh attempt, the same bound for every store
                result = {**cascade.abstain("stale_after_retry"), "stage": result.get("stage", 0)}
            else:
                again = await self.answer(tenant, question, context, _retried=True)
                return again | {"retried_after_change": True}
        if changed is not None and self.release_retry():
            first = {"reason": changed, "snapshot": snapshot}
            revalidated, why = None, None
            if _retried:  # one retry at most
                result = {**result, **cascade.abstain("stale_after_retry")}
            elif draft is not None and self.release_revalidate():
                revalidated, why = await self.revalidate(
                    tenant, question, context, draft, reads, start
                )
            if revalidated is not None and revalidated["status"] in ("FULL", "PARTIAL"):
                result = {**revalidated, "revalidated_after_change": True, "first_attempt": first}
            elif revalidated is not None:  # the release check withheld it (revoked, hidden)
                result = {**revalidated, "revalidation": why, "first_attempt": first}
            elif not _retried:
                allowed, declined = self.retry_permitted(verdict, draft, began, clock, context)
                if not allowed:
                    result = {**result, "retry_declined": declined}
                    if why is not None:
                        result["revalidation"] = why
                else:
                    again, failure = await self.retry(tenant, question, context, began, clock)
                    if again is not None:
                        extra = {"revalidation": why} if why is not None else {}
                        return again | {
                            "retried_after_change": True,
                            "first_attempt": first,
                            **extra,
                        }
                    result = {
                        **result,
                        **cascade.abstain("retry_budget_exhausted"),
                        "retry_error": failure,
                    }
        out = {
            **result,
            "as_of": question.as_of.isoformat(),
            "known_at": question.known_at.isoformat(),
            "snapshot": snapshot,
            "config_hash": universal_hash(),
            "arm": self.arm.stamp() if self.arm is not None else None,
            "storage": storage.describe(),
            "audit": audits,
        }
        if self.handoff_packet and out["status"] not in handoff_packet.RELEASED:
            out["handoff"] = handoff_packet.packet(
                out, stages, self.handoff_view(tenant, stages, audits)
            )
        if recording:  # PROD-2: the decision as a cache entry (the result itself is unchanged)
            principal = getattr(self.retriever, "principal", None)
            entry = cache_store.entry_of(
                out,
                stages,
                question,
                tenant,
                arm=arm_stamp,
                snapshot=snapshot,
                scope={
                    "principal": getattr(principal, "principal_id", None),
                    "excluded": len(getattr(self.retriever, "excluded_ids", None) or ()),
                    "allowed": None
                    if getattr(self.retriever, "allowed_ids", None) is None
                    else len(self.retriever.allowed_ids),
                },
                source=cache.source,
            )
            if entry is not None:
                cache.store.put(entry)
        if out["status"] not in handoff_packet.RELEASED:
            out = withhold_drafts(out)
        return out

    def handoff_view(self, tenant, stages, audits):
        """PROD-1: what the requester may see now, for the packet — one read-only release check
        over every document the packet could show, this retriever's deny list and selection, and
        the validity gate's statuses. A check that cannot be read withholds every unit."""
        storage = self.retriever.storage
        principal = getattr(self.retriever, "principal", None)
        try:
            release = storage.release.authorize(tenant, handoff_packet.documents(stages), principal)
            hidden = None if release.hidden is not None else storage.state.hidden(tenant, principal)
        except Exception:  # noqa: BLE001 — fail closed: no unit text when visibility is unknown
            return handoff_packet.sight(None, audits=audits, unknown="visibility_unknown")
        return handoff_packet.sight(
            release,
            audits=audits,
            hidden=hidden,
            excluded=getattr(self.retriever, "excluded_ids", None),
            allowed=getattr(self.retriever, "allowed_ids", None),
        )

    def scope_verdict(
        self, tenant, question, reads, start, snapshot, auth, *, vectors=None, fence_view=True
    ):
        """RVI-03: the scoped verdict over the range the release check read (`rvg.universal.
        scoped`), for this engine's principal, selection, embedding space and strategy. With
        `acl_moves = after_check` the check reads visibility as of the fence (FIX-ACL-WINDOW);
        `fence_view=False` keeps the state's (the revalidation, protocol §2.5)."""
        embedding = getattr(self.retriever, "embedding", None)
        return scope_fence.check(
            self.retriever.storage,
            tenant,
            question=question,
            principal=getattr(self.retriever, "principal", None),
            allowed=getattr(self.retriever, "allowed_ids", None),
            excluded=getattr(self.retriever, "excluded_ids", None),
            reads=reads or [],
            start=start,
            snapshot=snapshot,
            fence_seq=auth.revocation_seq,
            fence_epoch=auth.acl_epoch,
            space=embedding.describe() if embedding is not None else None,
            strategy=_U["retrieval"]["strategy"],
            vectors=vectors,
            fence_hidden=self.fence_hidden(auth) if fence_view else None,
        )

    async def vector_writes(self, tenant, question, reads, start, auth, *, fence_view=True):
        """RVI-03 F2 (`release.vectors = scoped`): the vector writes committed since the attempt
        began, for the scoped check — the outbox `vector` rows up to a mark read now, the
        catalog, whether a search that read rows without vectors would now get another BM25
        fallback, and whether another vector row arrived while this was read."""
        storage = self.retriever.storage
        state = storage.state
        if start.outbox_mark is None or not hasattr(state, "outbox"):
            return scope_fence.VectorWrites(rows=None)
        # FIX-V9-CHURN: with "after_check" the rows after the mark are placed by the check itself,
        # once it has read everything (scoped.check); with "hold" any of them is UNKNOWN (V9).
        after_check = self.release_vector_moves() == "after_check"
        mark = state.outbox_mark(tenant)
        rows = [
            o
            for o in state.outbox(tenant, start.outbox_mark)
            if o["kind"] == "vector" and o["id"] <= mark
        ]
        if not rows:
            return scope_fence.VectorWrites(rows=(), mark=mark, after_check=after_check)
        catalog = await storage.catalogs.get(tenant, run=asyncio.to_thread)
        queried = set()
        changed = await self.fallback_changed(
            tenant,
            question,
            reads or [],
            catalog,
            auth,
            queried=queried,
            hidden=self.fence_hidden(auth) if fence_view else None,
        )
        later = not after_check and any(o["kind"] == "vector" for o in state.outbox(tenant, mark))
        return scope_fence.VectorWrites(
            rows=tuple(rows),
            mark=mark,
            catalog=catalog,
            fallback_changed=changed,
            moved=later,
            after_check=after_check,
            fallback_hashes=frozenset(queried),
        )

    async def fallback_changed(
        self, tenant, question, reads, catalog, auth, *, queried=None, hidden=None
    ):
        """RVI-03 F2: whether a search that read rows without vectors would now get another BM25
        fallback — the same text and top-k over the rows that still have no vector at the
        release check's knowledge number, without the dense seeds it consumed. Which rows have a
        vector is read from the blob store by input hash (the index's view would rebuild its
        matrix after every vector write; the blob store refuses invalid vectors, so the two
        agree). `hidden`: the principal's hidden ids as of the fence (FIX-ACL-WINDOW), else the
        state's are read."""
        searches = [r for r in reads if r.retrieval.get("pending")]
        if not searches:
            return False
        retriever = self.retriever
        state, space = retriever.store, retriever.embedding.describe()
        fence = auth.revocation_seq
        mask = catalog.eligible(
            known_at=question.known_at,
            at=fence,
            revoked=state.revoked(tenant, at=fence),
            hidden=state.hidden(tenant, retriever.principal) if hidden is None else hidden,
            allowed=retriever.allowed_ids,
            excluded=getattr(retriever, "excluded_ids", None) or (),
        )
        rows = np.flatnonzero(mask)
        hashes = {catalog.input_hashes[i] for i in rows}
        if queried is not None:  # FIX-V9-CHURN: what this read, for the rows after the mark
            queried.update(hashes)
        have = await asyncio.to_thread(
            retriever.storage.blobs.has_vectors, tenant, space.fingerprint, hashes
        )
        pending = catalog.docs([i for i in rows if catalog.input_hashes[i] not in have])
        for read in searches:
            ranking = read.retrieval.get("ranking") or []
            dense = {r["key"] for r in ranking if r.get("via") == "dense"}
            consumed = [r["key"] for r in ranking if r.get("via") == "lexical_pending"]
            probe = question.model_copy(update={"text": read.text})
            now = [k for k in retriever._lexical(probe, pending, read.k) if k not in dense]
            if now != consumed:
                return True
        return False

    def retry_permitted(self, verdict, draft, began, clock, context):
        """RVI-03 F3 (parent docs/rvi/RVI03_F3_PROTOCOL.md §3): whether this arm's rule lets a
        withheld change start a regeneration — (allowed, why not)."""
        rule = self.release_retry_rule()
        if rule == "any" or verdict is None:
            return True, None
        unknown = any(outcome == scope_fence.UNKNOWN for _, outcome, _ in verdict.items)
        if rule == "slack":
            try:
                left = context.remaining()
            except ProviderError:
                return False, "slack:no_deadline"
            if left < 1.5 * (clock() - began):
                return False, "slack:deadline"
            if unknown or len(verdict.related) != 1:
                return False, "slack:changes"
            return True, None
        cited = {u.split("@")[0] for u in (draft or {}).get("evidence_ids") or []}
        if verdict.related & cited:
            return True, None
        return False, "cited:untouched"

    async def revalidate(self, tenant, question, context, draft, reads, first):
        """RVI-03 F3 R-c (`release.revalidate`, protocol §3.5): the first attempt's searches and
        validity plane read again at the current state, without a model call. When every input
        it consumed is byte-identical (seeds, documents, BM25 fallback, watermark, plane) and the
        ACL epoch is the one it began with, the first draft goes through the release check again
        over the changes since this read. Returns (released or withheld result, None), or
        (None, why) when a regeneration rule decides."""
        storage = self.retriever.storage
        store = storage.state
        start = scope_fence.begin(store, tenant)
        snapshot = store.snapshot(tenant)
        if start.acl_epoch != first.acl_epoch:
            return None, "acl_epoch_moved"
        find = getattr(self.retriever, "validity_documents", None)
        again = []
        for read in reads or []:
            probe = question.model_copy(update={"text": read.text})
            docs, retrieval = await self.retriever.search(
                tenant,
                probe,
                context,
                strategy=_U["retrieval"]["strategy"],
                top_k=read.k,
                expand=True,
            )
            plane = await asyncio.to_thread(find, tenant, docs) if find else []
            seen = [*docs, *plane]
            again.append(
                scope_fence.Read(
                    text=probe.text,
                    k=read.k,
                    ids=frozenset(d.id for d in seen),
                    families=frozenset(d.family for d in seen),
                    retrieval=retrieval,
                    vector=getattr(self.retriever, "_query_vectors", {}).get(probe.text),
                    keys=tuple(sorted(d.key for d in docs)),
                    plane=tuple(sorted(d.key for d in plane)),
                )
            )
        if not reads or [r.consumed() for r in again] != [r.consumed() for r in reads]:
            return None, "inputs_differ"
        documents = {u.split("@")[0] for u in draft.get("evidence_ids") or []}
        auth = storage.release.authorize(tenant, documents, self.retriever.principal)
        if not auth.ok:
            withheld = {
                **cascade.abstain(auth.reason),
                "stage": draft.get("stage", 0),
                "blocked_documents": list(auth.blocked),
            }
            return withheld, "release_check"
        vectors = (
            await self.vector_writes(tenant, question, again, start, auth, fence_view=False)
            if self.release_vectors() == "scoped"
            else None
        )
        verdict = self.scope_verdict(
            tenant, question, again, start, snapshot, auth, vectors=vectors, fence_view=False
        )
        if verdict.status != scope_fence.CLEAR:
            return None, "changed_again"
        release = {
            "mode": "scoped_dependency",
            "knowledge_snapshot": snapshot,
            "revocation_seq": auth.revocation_seq,
            "acl_epoch": auth.acl_epoch,
            "scope": verdict.receipt(),
            "revalidated": True,
        }
        return {**draft, "release": release}, None

    async def retry(self, tenant, question, context, began, clock):
        """RVI-03 (`release.retry = 1`): one more attempt after a change at the fence, through
        this same method, so evidence packaging and checks are those of the first attempt. It is
        not started when less time is left than the first attempt took, and one that runs out of
        budget is not released: both return `(None, code)` and end in a safe abstention."""
        try:
            left = context.remaining()
        except ProviderError as exc:
            return None, exc.code
        if left < clock() - began:
            return None, "deadline_below_first_attempt"
        again = await self.answer(tenant, question, context, _retried=True)
        error = again.get("error") or {}
        codes = [error.get("code"), *(error.get("codes") or [])]
        if again["status"] == "ERROR" and any(str(c).endswith("_exceeded") for c in codes if c):
            return None, next(str(c) for c in codes if c and str(c).endswith("_exceeded"))
        return again, None

    async def moved(
        self,
        tenant,
        question,
        context,
        snapshot,
        families,
        audits,
        *,
        access=None,
        fence=None,
        read=(),
    ):
        """A relevant change since `snapshot` (latest_at_release): the tenant's ACL epoch at the
        release check differs from the one read before the attempt (any document may have
        appeared or disappeared for the principal); a new revision of a document the attempt
        gave the validity gate, or one that declares a replacement of it, whatever its family or
        ACL (the validity plane reads it); an event whose kind is unknown; an insert or
        revocation in a family the answer retrieved; or a first search that now returns other
        seeds or other documents. No model call: the question's vector is reused."""
        store = self.retriever.storage.state
        if access is not None and fence is not None and fence.acl_epoch != access:
            return True
        if store.snapshot(tenant) == snapshot:
            return False
        if read:
            events = store.events(tenant, snapshot)
            keys = [
                (e["document_id"], e["revision"])
                for e in events
                if e["kind"] == "insert" and e["document_id"] and e["revision"]
            ]
            bodies = store.bodies(tenant, keys) if keys else {}
            for e in events:
                if e["kind"] not in ("insert", "revoke") or not e["document_id"]:
                    return True  # an event this check cannot map
                if e["kind"] != "insert":
                    continue
                document = bodies.get((e["document_id"], e["revision"]))
                if document is None or document.id in read:
                    return True
                if any(r.document_id in read for r in document.supersedes):
                    return True
        if store.changed_families(tenant, snapshot) & families:
            return True
        first = (audits or [{}])[0].get("retrieval") or {}
        _, now = await self.retriever.search(
            tenant,
            question,
            context,
            strategy=_U["retrieval"]["strategy"],
            top_k=_U["retrieval"]["candidates"],
            expand=True,
        )
        return now.get("seed_ids") != first.get("seed_ids") or now.get("documents") != first.get(
            "documents"
        )
