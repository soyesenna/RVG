"""U22 arms (parent docs/UNIVERSAL_U22_PROTOCOL.md §1): the partner evidence and the judge of the
second-stage agreement rule are opt-in experiment keys; with them off the rule is the policy's
(U8: first evidence in reverse, identical text). No model call."""

import pytest

from rvg.control import cascade
from rvg.control.grade import normalized, normalized_equal, same_text
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm


def test_the_arm_keys_are_accepted_and_unknown_values_refused():
    arm = Arm.of(
        "PWS", cascade={"consistency_partner": "widened", "consistency_judge": "same_text"}
    )
    engine = UniversalEngine(retriever=None, decision=None, writer=None, extractor=None, arm=arm)
    assert (engine.consistency_partner(), engine.consistency_judge()) == ("widened", "same_text")
    plain = UniversalEngine(retriever=None, decision=None, writer=None, extractor=None)
    assert (plain.consistency_partner(), plain.consistency_judge()) == ("widened", "same_text")
    u8 = Arm.of(
        "U8", cascade={"consistency_partner": "first", "consistency_judge": "identical_text"}
    )
    with_u8 = UniversalEngine(retriever=None, decision=None, writer=None, extractor=None, arm=u8)
    assert (with_u8.consistency_partner(), with_u8.consistency_judge()) == (
        "first",
        "identical_text",
    )
    with pytest.raises(ValueError):
        Arm.of("bad", cascade={"consistency_partner": "all"})
    with pytest.raises(ValueError):
        Arm.of("bad", cascade={"consistency_judge": "jev"})


def test_the_normalized_judge_ignores_case_width_group_separators_and_trailing_zeros():
    assert normalized("Ｐｒｉｃｅ: 1,234.50 USD") == "price 1234 5 usd"
    assert normalized_equal("1,234 units", "1234 Units")
    assert normalized_equal("12.0%", "12 %")
    assert not normalized_equal("12%", "15%")
    assert not normalized_equal("", "")
    assert not normalized_equal("Main Street", "Indian Fields Road")


def test_the_same_text_judge_is_u12b():
    assert same_text("44", "44 aircraft")
    assert not same_text("12%", "15%")
    assert same_text("Indian Fields Road (New York Route 32)", "Indian Fields Road, Route 32")
    assert not same_text("", "x")


def test_identical_text_is_the_policy_judge():
    assert cascade.same_short_answer("Paris.", "paris", "identical_text")
    assert not cascade.same_short_answer("Paris", "Paris, France", "identical_text")
    assert cascade.same_short_answer("Paris", "Paris, France", "same_text")
    with pytest.raises(ValueError):
        cascade.same_short_answer("a", "a", "off")


def widened_case():
    from test_cascade import retriever, unit

    # u1..u5 carry no answer, u6 does: the first stage (u1..u3) abstains, the widened one is FULL
    return retriever([unit(i, "") for i in range(1, 6)] + [unit(6, "", "Revenue was 42.")])


def test_a_widened_partner_is_written_on_the_second_stage_evidence_in_reverse():
    from test_cascade import Writer, run

    writer = Writer()
    found = run(retrieve=widened_case(), writer=writer, step="widen", consistency_partner="widened")
    assert writer.seen[-1] == ["u6", "u5", "u4", "u3", "u2", "u1"]
    assert found["status"] == "FULL" and found["stage"] == 2
    assert found["trace"]["consistency"] == {
        "partner_short_answer": "42",
        "agreed": True,
        "partner": "widened",
        "judge": "identical_text",
    }


def test_off_writes_no_partner_and_releases_the_second_stage():
    from test_cascade import Writer, run

    policy, off = Writer(), Writer()
    run(retrieve=widened_case(), writer=policy, step="widen")
    found = run(retrieve=widened_case(), writer=off, step="widen", consistency_judge="off")
    assert len(off.seen) == len(policy.seen) - 1  # no partner draft
    assert found["status"] == "FULL" and found["stage"] == 2
    assert "consistency" not in found["trace"]


@pytest.mark.parametrize(
    "judge, released", [("identical_text", False), ("normalized", False), ("same_text", True)]
)
def test_the_judge_decides_on_the_first_evidence_partner(judge, released):
    from test_cascade import Writer, run

    # the partner (first evidence reversed) says "Revenue 42" where the second draft says "42"
    found = run(
        retrieve=widened_case(),
        writer=Writer(partner_answer="Revenue 42"),
        step="widen",
        consistency_judge=judge,
    )
    assert (found["stage"] == 2) is released
    assert found["trace"]["consistency"]["agreed"] is released


def test_without_the_keys_the_rule_is_the_policys():
    import asyncio

    from test_cascade import QUESTION, Contained, Controller, Writer, retriever, unit

    from rvg.control.verify import Asker
    from rvg.runtime import CallContext
    from rvg.universal import universal

    assert universal()["cascade"]["consistency"] == {
        "partner": "widened",
        "judge": "same_text",
        "theta": 0.7,
    }
    ctx = CallContext.with_timeout(5, max_decision_calls=20, max_generation_calls=9)
    writer = Writer()
    found = asyncio.run(
        cascade.run(
            QUESTION,
            retrieve=retriever(
                [unit(i, "") for i in range(1, 6)] + [unit(6, "", "Revenue was 42.")]
            ),
            writer=writer,
            controller=Controller(),
            asker=Asker(Contained(0.9), ctx),
            context=ctx,
            step="widen",
            evidence_window="none",
        )
    )
    assert writer.seen[-1][0] == "u6"  # the partner saw the second-stage evidence in reverse
    assert found["stage"] == 2 and found["trace"]["consistency"]["judge"] == "same_text"
