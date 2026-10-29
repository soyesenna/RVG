"""The release fence (N3, PORT-0). An answer is published only if, in one atomic read of the
authoritative state, none of its documents is revoked and the principal may see all of them.
What the read saw is the release ticket: the guarantee holds at that instant; bytes delivered
afterwards are not recalled. The state store decides how the read is atomic (a SQLite read
transaction, a PostgreSQL transaction that shares the tenant row lock with every writer); no
transaction stays open while a model is called."""

from __future__ import annotations


class ReleaseCoordinator:
    def __init__(self, state):
        self.state = state

    def authorize(self, tenant, document_ids, principal):
        return self.state.authorize_release(tenant, set(document_ids), principal)
