"""Who may see what (N3). A principal comes only from a trusted server or authentication
context; a runtime request body never carries one that is believed. Documents without ACL
entries are visible to every principal of their tenant (the behaviour before N3, so existing
stores are unchanged); a document with entries is visible to a principal named — directly or
through a group — by an `allow` entry and by no `deny` entry. `allowed_ids` only narrows.

An ACL that allows nobody (RVI-07A) is stored as one `allow` entry for `DENY_ALL`, a name no
principal or group may hold: the existing rule then hides the document from everyone, in this
version and in versions before it, without a new entry kind."""

from __future__ import annotations

from dataclasses import dataclass, field

RESERVED_PREFIX = "rvg-reserved:"
DENY_ALL = RESERVED_PREFIX + "deny-all"


@dataclass(frozen=True)
class TrustedPrincipal:
    principal_id: str
    groups: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.principal_id.strip():
            raise ValueError("principal_required")
        if any(n.startswith(RESERVED_PREFIX) for n in (self.principal_id, *self.groups)):
            raise ValueError("reserved_principal_name")

    @property
    def names(self):
        return {self.principal_id, *self.groups}


def hidden_ids(rows, principal):
    """Document ids a principal cannot see, from (id, principal, effect) ACL rows. Without a
    principal only documents that have no ACL entries are visible."""
    allows, denies = {}, {}
    for identifier, who, effect in rows:
        (allows if effect == "allow" else denies).setdefault(identifier, set()).add(who)
    names = principal.names if principal is not None else set()
    out = set()
    for identifier in set(allows) | set(denies):
        allowed = allows.get(identifier)
        if (
            (allowed is not None and not (allowed & names))
            or (denies.get(identifier, set()) & names)
            or principal is None
        ):
            out.add(identifier)
    return out


@dataclass(frozen=True)
class Release:
    """The outcome of the release check: the linearisation point of an answer's publication."""

    ok: bool
    reason: str | None
    revocation_seq: int
    acl_epoch: int
    blocked: tuple[str, ...] = ()
    # FIX-ACL-WINDOW (parent docs/rvi/FIX_ACL_WINDOW_PROTOCOL.md §2.1): the principal's hidden
    # document ids in the same read as `acl_epoch`, so a release check can read visibility as of
    # the fence; None when the store does not give them. Not part of the ticket's identity.
    hidden: frozenset | None = field(default=None, compare=False, repr=False)
