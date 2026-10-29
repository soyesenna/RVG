"""Experiment arms (N-series): a named override of part of the policy, handed by an evaluation
runner to the engine's constructor. A runtime request cannot carry one, the product CLI never
builds one, and every answer made under an arm is stamped with its name and hash. Adopting an arm
means writing its values into `rvg/resources/policies/universal.json` (a new policy hash), not running it."""

from __future__ import annotations

from dataclasses import dataclass

from .trace import canonical_hash

# Only these policy keys may be overridden by an arm; anything else is refused.
OVERRIDABLE = {
    "validity": {
        "implicit_revision_prior",
        "implicit_revision_prior_scope",
        "unknown_revision",
        "coexisting_revisions",
    },
    "release": {
        "mode",
        "retry",
        "vectors",
        "retry_rule",
        "revalidate",
        "vector_moves",
        "acl_moves",
    },
    "cascade": {
        "conflict_partial",
        "writer_schema",
        "writer_contract",
        "evidence_window",
        "consistency_partner",
        "consistency_judge",
        "answered_threshold",
        "write_top",
        "scope_top1",
    },
}
RELEASE_MODES = ("current", "as_of_snapshot", "latest_at_release", "scoped_dependency")
# RVI-03: modes that take `release.retry` (0 or 1); latest_at_release has its own single retry.
RETRY_MODES = ("current", "scoped_dependency")
# RVI-03 F2 (parent docs/rvi/RVI03_F2_PROTOCOL.md §3): whether the scoped fence places vector writes
# (outbox `vector` rows) by relatedness; "ignore" is the scoped fence as before
VECTOR_SCOPES = ("ignore", "scoped")
# RVI-03 F3 (parent docs/rvi/RVI03_F3_PROTOCOL.md §3): when a scoped retry regenerates ("any": on
# every withheld change, as before; "slack": time to spare and one related document; "cited": a
# cited document changed) and whether the first draft is revalidated before any regeneration
RETRY_RULES = ("any", "slack", "cited")
# FIX-V9-CHURN (parent docs/rvi/FIX_V9_CHURN_PROTOCOL.md §2): a vector row that arrives while the
# check reads ("hold": UNKNOWN, as F2's V9; "after_check": ordered after the release unless the
# check read its input)
VECTOR_MOVES = ("hold", "after_check")
# FIX-ACL-WINDOW (parent docs/rvi/FIX_ACL_WINDOW_PROTOCOL.md §2): an ACL row that arrives while the
# check reads ("hold": UNKNOWN, as before; "after_check": the check reads visibility as of the fence
# and orders the row after the release unless it touches a document or family the attempt read)
ACL_MOVES = ("hold", "after_check")
CONFLICT_PARTIAL = ("off", "rewrite")
WRITER_SCHEMAS = ("W0", "W1", "W2")
# N6-2 (parent docs/N6_2_TECHQA_PROTOCOL.md §1): the answer-form contract and the neighbour window
WRITER_CONTRACTS = ("W0", "form-v1")
EVIDENCE_WINDOWS = ("none", "neighbors-v1")
# U22 (parent docs/UNIVERSAL_U22_PROTOCOL.md §1): the second-stage agreement rule — the evidence the
# partner draft is written on and how the two short answers are compared ("off": no partner)
CONSISTENCY_PARTNERS = ("first", "widened")
CONSISTENCY_JUDGES = ("identical_text", "normalized", "same_text", "off")
# U23 (parent docs/UNIVERSAL_U23_PROTOCOL.md §1): the controller's `answered` threshold (0 drops it
# from the gate) and the first stage's writing units
WRITE_TOPS = (3, 5, 10)
# U30 (parent docs/UNIVERSAL_U30_PROTOCOL.md §12): when the scope leaves the top-ranked unit's
# document out — "off" (the policy), S2 "entity-tie-v1", S3 "dual-draft-v1"
SCOPE_TOP1 = ("off", "entity-tie-v1", "dual-draft-v1")


@dataclass(frozen=True)
class Arm:
    name: str
    validity: tuple[tuple[str, object], ...] = ()
    release: tuple[tuple[str, object], ...] = ()
    cascade: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        unknown = {k for k, _ in self.validity} - OVERRIDABLE["validity"]
        unknown |= {k for k, _ in self.release} - OVERRIDABLE["release"]
        unknown |= {k for k, _ in self.cascade} - OVERRIDABLE["cascade"]
        if unknown or not self.name.strip():
            raise ValueError(f"arm_override_not_allowed:{sorted(unknown)}")
        mode = dict(self.release).get("mode", "current")
        if mode not in RELEASE_MODES:
            raise ValueError("unknown_release_mode")
        retry = dict(self.release).get("retry", 0)
        if retry not in (0, 1) or isinstance(retry, bool) or (retry and mode not in RETRY_MODES):
            raise ValueError("unknown_release_retry")
        vectors = dict(self.release).get("vectors", "ignore")
        if vectors not in VECTOR_SCOPES or (vectors != "ignore" and mode != "scoped_dependency"):
            raise ValueError("unknown_release_vectors")
        moves = dict(self.release).get("vector_moves", "hold")
        if moves not in VECTOR_MOVES or (moves != "hold" and vectors != "scoped"):
            raise ValueError("unknown_release_vector_moves")
        acl = dict(self.release).get("acl_moves", "hold")
        if acl not in ACL_MOVES or (acl != "hold" and mode != "scoped_dependency"):
            raise ValueError("unknown_release_acl_moves")
        rule = dict(self.release).get("retry_rule", "any")
        revalidate = dict(self.release).get("revalidate", False)
        if rule not in RETRY_RULES or not isinstance(revalidate, bool):
            raise ValueError("unknown_release_retry_rule")
        if (rule != "any" or revalidate) and (mode != "scoped_dependency" or retry != 1):
            raise ValueError("unknown_release_retry_rule")
        if dict(self.cascade).get("conflict_partial", "off") not in CONFLICT_PARTIAL:
            raise ValueError("unknown_conflict_partial")
        if dict(self.cascade).get("writer_schema", "W0") not in WRITER_SCHEMAS:
            raise ValueError("unknown_writer_schema")
        if dict(self.cascade).get("writer_contract", "W0") not in WRITER_CONTRACTS:
            raise ValueError("unknown_writer_contract")
        if dict(self.cascade).get("evidence_window", "none") not in EVIDENCE_WINDOWS:
            raise ValueError("unknown_evidence_window")
        if dict(self.cascade).get("consistency_partner", "first") not in CONSISTENCY_PARTNERS:
            raise ValueError("unknown_consistency_partner")
        if dict(self.cascade).get("consistency_judge", "identical_text") not in CONSISTENCY_JUDGES:
            raise ValueError("unknown_consistency_judge")
        answered = dict(self.cascade).get("answered_threshold", 0.0)
        if (
            isinstance(answered, bool)
            or not isinstance(answered, (int, float))
            or not 0 <= answered <= 1
        ):
            raise ValueError("answered_threshold_out_of_range")
        if dict(self.cascade).get("write_top", 3) not in WRITE_TOPS:
            raise ValueError("unknown_write_top")
        if dict(self.cascade).get("scope_top1", "off") not in SCOPE_TOP1:
            raise ValueError("unknown_scope_top1")

    @classmethod
    def of(cls, name, *, validity=None, release=None, cascade=None):
        return cls(
            name,
            tuple(sorted((validity or {}).items())),
            tuple(sorted((release or {}).items())),
            tuple(sorted((cascade or {}).items())),
        )

    @property
    def digest(self):
        body = {"name": self.name, "validity": dict(self.validity), "release": dict(self.release)}
        if self.cascade:  # absent for older arms, so their hashes stay as recorded
            body["cascade"] = dict(self.cascade)
        return canonical_hash(body)

    def stamp(self):
        out = {
            "name": self.name,
            "hash": self.digest,
            "validity": dict(self.validity),
            "release": dict(self.release),
        }
        if self.cascade:
            out["cascade"] = dict(self.cascade)
        return out
