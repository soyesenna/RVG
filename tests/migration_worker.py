"""A writer in its own process for the PORT-3 tests: knowledge events and vectors written to
whichever deployment the pointer names, each acknowledged with the record it was written under."""

import json
import time

import numpy as np
from test_n3_release import DIMS, SPACE, doc

from rvg.storage.migration import DeploymentPointer, write_active
from rvg.temporal.store import TemporalRetriever


def open_state(spec):
    if spec["kind"] == "sqlite":
        from rvg.storage.sqlite import SQLiteState

        return SQLiteState(spec["path"])
    from rvg.storage.postgres import PostgresState

    return PostgresState(spec["dsn"], schema=spec["schema"])


def writer(pointer_path, specs, count, interval, log_path, start_at, done=None, after=None):
    """`count` writes at most. With `done` (a shared counter) it counts acknowledged writes; with
    `after` it stops `after` writes into a generation newer than its first one (the switch), so
    the writes on both sides of a switch never depend on how fast the machine is."""
    pointer = DeploymentPointer(pointer_path)
    states = {name: open_state(spec) for name, spec in specs.items()}
    rng = np.random.default_rng(11)
    log, first, moved = [], None, 0
    while time.monotonic() < start_at:
        time.sleep(0.001)
    for k in range(count):
        kind = ("insert", "insert", "revoke", "acl", "vector")[k % 5]
        d = doc(f"w{k:04d}", f"Written note {k} with figure {k}.")
        text = TemporalRetriever.input_text(d)
        vector = rng.normal(size=DIMS)

        def change(state, kind=kind, d=d, text=text, vector=vector, k=k):
            if kind == "insert":
                state.upsert("t", d)
                state.put_vector("t", SPACE.fingerprint, text, vector)
                return {"document": d.id, "knowledge_seq": state.snapshot("t")}
            if kind == "revoke":
                target = f"w{k - 2:04d}"
                state.revoke("t", target)
                return {"document": target, "knowledge_seq": state.snapshot("t")}
            if kind == "acl":
                target = f"w{k - 3:04d}"
                state.set_access("t", target, "alice", "deny")
                return {"document": target, "acl_epoch": state.acl_epoch("t")}
            target = doc(f"w{k - 4:04d}", f"Written note {k - 4} with figure {k - 4}.")
            text = TemporalRetriever.input_text(target)
            state.put_vector("t", SPACE.fingerprint, text, vector)
            return {"document": target.id, "text": text, "vector": vector.tolist()}

        value, record = write_active(pointer, states, change)
        log.append(
            {"k": k, "kind": kind, **value}
            | {"deployment": record["deployment"], "generation": record["generation"]}
        )
        if done is not None:
            with done.get_lock():
                done.value += 1
        first = record["generation"] if first is None else first
        moved += record["generation"] != first
        if after is not None and moved >= after:
            break
        time.sleep(interval)
    with open(log_path, "w") as f:
        json.dump(log, f)
    for state in states.values():
        state.close()
