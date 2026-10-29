"""U32 (0.1.0.dev42, parent docs/UNIVERSAL_U32_PROTOCOL.md): the policy's `release.acl_moves` is
`after_check` and its `cascade.scope_top1` is `dual-draft-v1`. An engine without an arm, or whose
arm leaves the key out, runs them; the rollback arm R — dev41's release block whole and
`scope_top1: off` — runs dev41's values. Either key can be rolled back alone."""

import pytest
from test_fix_acl_window import during_the_check
from test_rvi03_scoped_fence import PG, World, ask, reasons

from rvg.universal import universal
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

DEV41_RELEASE = {
    "mode": "scoped_dependency",
    "vectors": "scoped",
    "vector_moves": "after_check",
    "retry": 1,
    "retry_rule": "slack",
    "revalidate": False,
}
R = Arm.of("u32-R", release=DEV41_RELEASE, cascade={"scope_top1": "off"})


@pytest.fixture(params=["sqlite", "postgres"])
def world(request, tmp_path):
    if request.param == "postgres" and not PG:
        pytest.skip("no PostgreSQL")
    w = World(request.param, tmp_path)
    yield w
    w.cleanup()


def engine(arm=None):
    return UniversalEngine(retriever=None, decision=None, writer=None, extractor=None, arm=arm)


def test_the_policy_holds_both_keys_as_a_valid_arm():
    policy = universal()
    assert policy["release"] == {**DEV41_RELEASE, "acl_moves": "after_check"}
    assert policy["cascade"]["scope_top1"] == "dual-draft-v1"
    Arm.of("policy", release=policy["release"], cascade={"scope_top1": "dual-draft-v1"})


def test_an_engine_without_the_keys_runs_the_policys():
    for arm in (None, Arm.of("w", cascade={"consistency_partner": "widened"})):
        e = engine(arm)
        assert e.release_acl_moves() == "after_check"
        assert e.scope_top1() == "dual-draft-v1"


def test_the_rollback_arm_runs_dev41s_values_and_each_key_rolls_back_alone():
    e = engine(R)
    assert e.release_block() == DEV41_RELEASE
    assert e.release_acl_moves() == "hold" and e.scope_top1() == "off"
    acl_only = engine(Arm.of("a", release=DEV41_RELEASE))
    assert acl_only.release_acl_moves() == "hold" and acl_only.scope_top1() == "dual-draft-v1"
    scope_only = engine(Arm.of("s", cascade={"scope_top1": "off"}))
    assert scope_only.release_acl_moves() == "after_check" and scope_only.scope_top1() == "off"
    # an arm naming any release key keeps its block whole: without acl_moves it holds
    assert engine(Arm.of("v", release={"mode": "scoped_dependency"})).release_acl_moves() == "hold"


def test_the_fence_hidden_ids_reach_the_check_without_an_arm():
    class Auth:
        hidden = frozenset({"x"})

    assert engine().fence_hidden(Auth()) == {"x"}
    assert engine(R).fence_hidden(Auth()) is None


def test_an_unrelated_acl_row_during_the_check_is_the_next_event_by_default(world, monkeypatch):
    during_the_check(monkeypatch, lambda: world.store.replace_access("t", "far", {"bob": "deny"}))
    result, _ = ask(world, arm=None)
    monkeypatch.undo()
    assert result["status"] == "FULL" and result["short_answer"] == "500"
    assert "acl:CLEAR:after_check" in reasons(result)
    during_the_check(monkeypatch, lambda: world.store.replace_access("t", "far", {"carol": "deny"}))
    held, _ = ask(world, arm=R)
    assert held["reason"] == "source_changed"
    assert "acl:UNKNOWN:acl_moved_during_check" in reasons(held)
