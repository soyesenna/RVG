"""U23 arms (parent docs/UNIVERSAL_U23_PROTOCOL.md §1): the controller's `answered` threshold
and the first stage's writing units are opt-in experiment keys; with them off the controller and
the cascade are the policy's. No model call."""

import pytest

from rvg.control.controller import ControlPolicy
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm


def test_the_arm_keys_are_accepted_and_bad_values_refused():
    arm = Arm.of("WT5AN6", cascade={"write_top": 5, "answered_threshold": 0.6})
    assert dict(arm.cascade) == {"answered_threshold": 0.6, "write_top": 5}
    for bad in ({"write_top": 4}, {"answered_threshold": 1.5}, {"answered_threshold": True}):
        with pytest.raises(ValueError):
            Arm.of("bad", cascade=bad)


def test_the_answered_threshold_is_the_profiles_unless_set():
    assert ControlPolicy().limits()["*"]["answered"] == 0.84
    assert ControlPolicy(answered=0.6).limits()["*"]["answered"] == 0.6
    off = ControlPolicy(answered=0.0).limits()["*"]
    assert "answered" not in off and off["qualifiers"] == 0.3


def test_the_engine_passes_the_arm_values():
    plain = UniversalEngine(retriever=None, decision=None, writer=None, extractor=None)
    assert plain.controller.policy.answered is None and plain.write_top() is None
    arm = Arm.of("WT5AN0", cascade={"write_top": 5, "answered_threshold": 0.0})
    eng = UniversalEngine(retriever=None, decision=None, writer=None, extractor=None, arm=arm)
    assert eng.controller.policy.answered == 0.0 and eng.write_top() == 5


def test_write_top_sets_the_first_writing_pool():
    from test_cascade import Writer, retriever, run, unit

    ranked = [unit(i, "ACME") for i in range(1, 21)]
    policy, five = Writer(), Writer()
    run(retrieve=retriever(ranked), writer=policy)
    run(retrieve=retriever(ranked), writer=five, write_top=5)
    assert len(policy.seen[0]) == 3 and len(five.seen[0]) == 5
