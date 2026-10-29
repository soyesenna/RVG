"""PORT-3: moving a live tenant to another storage deployment, and back.

1. `copy`: export the tenant from the source in one read with the outbox mark of that read, and
   import it into an empty tenant of the target (same event numbers, same vector bytes — no
   re-embedding).
2. `catch_up`: apply the source's outbox entries after the mark to the target, in order, with
   the source's event numbers. Every entry is applied at most once by number — a knowledge event
   only when it is the source's next event after the target's last one, an ACL change only when
   the target's epoch is one behind — so a repeated or resumed catch-up changes nothing, and a
   gap is an error, never a silent skip. The ledger file records how far it got. It runs until
   the outbox is empty: under writers faster than the target applies (e.g. a local SQLite source
   into a remote PostgreSQL), it does not end — pass `limit`, and let `switch` (writes paused)
   apply the rest.
3. `switch`: under the deployment pointer's exclusive lock (no writer and no release fence of
   any process in between), a last catch-up, `verify` (digests of every space, event lists, ACL
   epoch, outbox drained), an optional shadow comparison, then the pointer names the target —
   one atomic file replace with a new generation. A check that fails leaves the pointer as it was.
4. In-flight requests finish on the deployment they pinned; their release fence
   (`PinnedRelease`) holds the pointer's shared lock and refuses (`deployment_changed`) once the
   pointer has moved, because the old deployment no longer sees revocations or ACL changes.
5. Rollback is the same procedure in the other direction, from the target's outbox mark
   recorded at the switch; the old deployment is kept (retention) until then.

There is no fallback: a request never moves to another deployment by itself; a failing
deployment's requests end in ERROR and say which deployment they ran on."""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from ..errors import ContractError
from ..temporal.access import Release
from .release import ReleaseCoordinator


class DeploymentPointer:
    """A small file naming the active deployment of a tenant, replaced atomically. Writers and
    release fences hold its shared lock for the duration of one write or one fence; a switch
    holds the exclusive lock. A turnstile lock in front keeps a steady stream of shared holders
    from starving the switch."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._gate = self.path.with_suffix(".gate")
        self._lock = self.path.with_suffix(".lock")
        for f in (self._gate, self._lock):
            f.touch(exist_ok=True)

    def read(self):
        return json.loads(self.path.read_text())

    def initialize(self, deployment, **detail):
        if self.path.exists():
            raise ContractError("deployment_pointer_exists")
        self._write({"generation": 1, "deployment": deployment, **detail})

    def _write(self, record):
        tmp = self.path.with_suffix(f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(record, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def _held(self, path, mode):
        fd = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(fd, mode)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @contextmanager
    def shared(self):
        """The active record, stable while the block runs (one write or one release fence)."""
        with self._held(self._gate, fcntl.LOCK_SH):
            fd = os.open(self._lock, os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            yield self.read()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @contextmanager
    def exclusive(self):
        """No shared holder in any process while the block runs; returns the waiting time."""
        started = time.monotonic()
        with self._held(self._gate, fcntl.LOCK_EX), self._held(self._lock, fcntl.LOCK_EX):
            yield {"record": self.read(), "waited_seconds": time.monotonic() - started}

    def switch(self, deployment, *, expect_generation, **detail):
        """Call under `exclusive()`: the pointer names `deployment` with the next generation,
        only if nobody switched since `expect_generation` was read."""
        current = self.read()
        if current["generation"] != expect_generation:
            raise ContractError("deployment_generation_conflict")
        record = {"generation": expect_generation + 1, "deployment": deployment, **detail}
        self._write(record)
        return record


class PinnedRelease(ReleaseCoordinator):
    """The release fence of a request pinned to one deployment: while the pointer still names
    that generation, the deployment's own atomic check (under the pointer's shared lock, so no
    switch interleaves); after a switch, nothing is released."""

    def __init__(self, state, pointer, generation):
        super().__init__(state)
        self.pointer, self.generation = pointer, generation

    def authorize(self, tenant, document_ids, principal):
        with self.pointer.shared() as record:
            if record["generation"] != self.generation:
                seen = self.state.authorize_release(tenant, set(), principal)
                return Release(False, "deployment_changed", seen.revocation_seq, seen.acl_epoch)
            return self.state.authorize_release(tenant, set(document_ids), principal)


def pin(pointer, deployments):
    """A request's view: the active deployment (a `Storage`) with its release fence pinned to the
    pointer's generation. The read needs no lock — the file is replaced atomically, and a switch
    after it is caught by the fence."""
    record = pointer.read()
    storage = deployments[record["deployment"]]
    return replace(
        storage, release=PinnedRelease(storage.state, pointer, record["generation"])
    ), record


def write_active(pointer, deployments, change):
    """One write to whichever deployment is active, under the pointer's shared lock (a switch
    waits for it; it never lands on a deployment the pointer has left). Returns the change's
    value and the record it was written under."""
    with pointer.shared() as record:
        return change(deployments[record["deployment"]]), record


class Migration:
    """Copy and catch-up of one tenant from `source` to `target` (state stores). The ledger
    (a JSON file) records the source outbox mark the target has reached."""

    def __init__(self, source, target, tenant, ledger):
        self.source, self.target, self.tenant = source, target, tenant
        self.ledger = Path(ledger)
        self._pending = deque()  # the source's event numbers after the target's last one

    def _next_source_event(self, have):
        while self._pending and self._pending[0] <= have:
            self._pending.popleft()
        if not self._pending:
            self._pending.extend(e["seq"] for e in self.source.events(self.tenant, since=have))
        return self._pending[0] if self._pending else None

    def _save(self, **fields):
        state = self.state() | fields
        tmp = self.ledger.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True))
        os.replace(tmp, self.ledger)

    def state(self):
        return json.loads(self.ledger.read_text()) if self.ledger.exists() else {}

    def copy(self):
        export, mark = self.source.export_with_mark(self.tenant)
        self.target.import_tenant(self.tenant, **export)
        self._save(applied=mark, copied_at_mark=mark)
        return mark

    def resume_from(self, mark):
        """Start a catch-up into a target that already holds the tenant (rollback)."""
        self._save(applied=mark)

    def catch_up(self, *, batch=500, limit=None, on_entry=None):
        """Apply the source's outbox after the ledger's mark; returns the entries applied and
        skipped (already applied, or a vector superseded by a later entry)."""
        applied = skipped = 0
        mark = self.state()["applied"]
        while limit is None or applied + skipped < limit:
            entries = self.source.outbox(self.tenant, since=mark)
            if limit is not None:
                entries = entries[: limit - applied - skipped]
            if not entries:
                break
            for n, entry in enumerate(entries, 1):
                if self._apply(entry):
                    applied += 1
                else:
                    skipped += 1
                mark = entry["id"]
                if on_entry is not None:
                    on_entry(entry)
                if n % batch == 0:
                    self._save(applied=mark)
            self._save(applied=mark)
        return {"applied": applied, "skipped": skipped, "mark": mark}

    def _apply(self, entry):
        kind, payload, tenant = entry["kind"], entry["payload"], self.tenant
        if kind in ("insert", "revoke"):
            number, have = entry["knowledge_seq"], self.target.snapshot(tenant)
            if have >= number:
                return False
            if self._next_source_event(have) != number:
                raise ContractError("migration_event_gap")
            if kind == "insert":
                key = (payload["id"], payload["revision"])
                self.target.upsert(tenant, self.source.bodies(tenant, [key])[key], event_seq=number)
            else:
                self.target.revoke(tenant, payload["id"], event_seq=number)
            if self.target.snapshot(tenant) != number:
                raise ContractError("migration_event_number_mismatch")
            return True
        if kind == "acl":
            epoch, have = payload["epoch"], self.target.acl_epoch(tenant)
            if have >= epoch:
                return False
            if have != epoch - 1:
                raise ContractError("migration_acl_gap")
            self.target.set_access(tenant, payload["id"], payload["principal"], payload["effect"])
            if self.target.acl_epoch(tenant) != epoch:
                raise ContractError("migration_acl_epoch_mismatch")
            return True
        if kind == "vector":
            space, key = payload["space"], payload["input_hash"]
            meta = self.source.vector_meta(tenant, space).get(key)
            if meta is None or meta[1] != payload["seq"]:
                return False  # superseded: a later entry carries the current bytes
            value = self.source.vectors(tenant, space, [key])[key]
            return bool(self.target.put_vector_blob(tenant, space, key, value))
        raise ContractError(f"migration_unknown_outbox_kind:{kind}")

    def verify(self, spaces, *, full=True):
        """Source and target hold the same tenant: event lists, ACL epoch, digest of the
        knowledge and of every space's vectors, and no source outbox entry left to apply. With
        `full=False` (under the switch lock) vectors are compared by their stored sha256 instead
        of their bytes, and the ACL entries by the epoch alone."""
        checks = {
            "events_equal": self.source.events(self.tenant) == self.target.events(self.tenant),
            "acl_epoch_equal": self.source.acl_epoch(self.tenant)
            == self.target.acl_epoch(self.tenant),
            "knowledge_digest_equal": self.source.digest(self.tenant)
            == self.target.digest(self.tenant),
            "outbox_drained": self.source.outbox_mark(self.tenant) == self.state()["applied"],
        }
        for space in spaces:
            if full:
                same = self.source.digest(self.tenant, space) == self.target.digest(
                    self.tenant, space
                )
            else:
                same = self.source.vector_checksums(
                    self.tenant, space
                ) == self.target.vector_checksums(self.tenant, space)
            checks[f"vectors_equal:{space[:12]}"] = same
        if full:
            acl = lambda state: sorted(
                (a["id"], a["principal"], a["effect"])
                for a in state.export_tenant(self.tenant)["acl"]
            )
            checks["acl_equal"] = acl(self.source) == acl(self.target)
        return checks


def switch(pointer, migration, deployment, *, spaces, shadow=None, full=True, **detail):
    """The switch of the module's step 3. Returns the new pointer record and what was checked;
    on a failed check raises `migration_verification_failed` with the pointer unchanged."""
    with pointer.exclusive() as held:
        started = time.monotonic()
        caught = migration.catch_up()
        checks = migration.verify(spaces, full=full)
        if shadow is not None:
            checks["shadow_equal"] = bool(shadow())
        if not all(checks.values()):
            raise ContractError(
                "migration_verification_failed:" + ",".join(k for k, v in checks.items() if not v)
            )
        record = pointer.switch(
            deployment,
            expect_generation=held["record"]["generation"],
            target_outbox_mark=migration.target.outbox_mark(migration.tenant),
            **detail,
        )
        paused = time.monotonic() - started
    return {
        "record": record,
        "checks": checks,
        "final_catch_up": caught,
        "waited_seconds": held["waited_seconds"],
        "writes_paused_seconds": paused,
    }
