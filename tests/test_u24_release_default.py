"""U24 (0.1.0.dev27, parent docs/UNIVERSAL_U24_REPORT.md): the policy's `release` block is the
default of an engine whose arm names no release key; an arm that names any release key keeps its
whole block, the keys it leaves out at their values before dev27. Existing arms therefore behave and
hash as they did; `release.mode = current` is the experiment rollback."""

import pytest
from test_rvi03_scoped_fence import CURRENT, World, ask, related, unrelated

from rvg.universal import universal
from rvg.universal.engine import UniversalEngine
from rvg.universal.experiment import Arm

CAND = {
    "mode": "scoped_dependency",
    "vectors": "scoped",
    "vector_moves": "after_check",
    "retry": 1,
    "retry_rule": "slack",
    "revalidate": False,
}
# U32 (dev42) added ACL moves after the check to the policy block
POLICY = {**CAND, "acl_moves": "after_check"}
SCOPED_VECTORS = {"mode": "scoped_dependency", "vectors": "scoped"}


def settings(arm):
    engine = UniversalEngine(retriever=None, decision=None, writer=None, extractor=None, arm=arm)
    return {
        "mode": engine.release_mode(),
        "vectors": engine.release_vectors(),
        "vector_moves": engine.release_vector_moves(),
        "retry": engine.release_retry(),
        "retry_rule": engine.release_retry_rule(),
        "revalidate": engine.release_revalidate(),
    }


BEFORE = {  # an arm's keys it leaves out, as every release accessor read them before dev27
    "mode": "current",
    "vectors": "ignore",
    "vector_moves": "hold",
    "retry": 0,
    "retry_rule": "any",
    "revalidate": False,
}


def test_the_policy_release_block_is_the_u24_candidate_and_a_valid_arm():
    assert universal()["release"] == POLICY
    Arm.of("policy", release=universal()["release"])  # a combination an arm may hold


def test_an_engine_without_release_keys_runs_the_policy_block():
    assert settings(None) == CAND
    assert settings(Arm.of("u22-PWS", cascade={"consistency_partner": "widened"})) == CAND


@pytest.mark.parametrize(
    "release",
    [
        {"mode": "current"},
        {"mode": "current", "retry": 1},
        {"mode": "latest_at_release"},
        SCOPED_VECTORS,
        {**SCOPED_VECTORS, "retry": 1, "retry_rule": "slack"},
        {**SCOPED_VECTORS, "vector_moves": "after_check"},
    ],
)
def test_an_arm_with_release_keys_keeps_its_whole_block(release):
    assert settings(Arm.of("arm", release=release)) == BEFORE | release


def test_existing_arm_digests_are_unchanged():
    """The digests of these arms as 0.1.0.dev26 computed them."""
    recorded = {
        Arm.of("rvi03-current", release={"mode": "current"}): "81224e547e0341d2",
        Arm.of("rvi03-f3-A2V", release=SCOPED_VECTORS): "94c957ca52cc4242",
        Arm.of(
            "rvi03-f3-A3a-slack", release={**SCOPED_VECTORS, "retry": 1, "retry_rule": "slack"}
        ): "94dc53058bbbe9a0",
        Arm.of("fix-v9-A2V9", release={**SCOPED_VECTORS, "vector_moves": "after_check"}): (
            "170b95acd41fc4f0"
        ),
        Arm.of("u24-CAND", release={k: v for k, v in CAND.items() if k != "revalidate"}): (
            "f941c466762d5c8a"
        ),
        Arm.of(
            "u22-PWS", cascade={"consistency_partner": "widened", "consistency_judge": "same_text"}
        ): "93fa657f44832736",
        Arm.of("n4-rewrite", cascade={"conflict_partial": "rewrite"}): "4883bf2d201171e5",
    }
    for arm, prefix in recorded.items():
        assert arm.digest.startswith(prefix), arm.name


def test_no_change_the_default_releases_with_a_clear_receipt():
    result, _ = ask(World(), arm=None)
    assert result["status"] == "FULL" and result["short_answer"] == "500"
    assert result["release"]["mode"] == "scoped_dependency"
    assert result["release"]["scope"]["status"] == "CLEAR"
    assert result["release"]["scope"]["checked"] == 0


@pytest.mark.parametrize("kind", ["insert", "revoke"])
def test_the_default_releases_through_an_unrelated_change_that_current_withholds(kind):
    w = World()
    result, _ = ask(w, unrelated(w, kind), arm=None)
    assert result["status"] == "FULL" and result["short_answer"] == "500"
    assert result["release"]["scope"]["status"] == "CLEAR"
    w = World()
    result, _ = ask(w, unrelated(w, kind), arm=CURRENT)
    assert result["status"] == "ABSTAIN" and result["reason"] == "source_changed"


def test_the_default_regenerates_after_a_related_revision_that_current_withholds():
    w = World()
    result, _ = ask(w, related(w, "revision"), arm=None)
    assert result["status"] == "FULL" and result["short_answer"] == "650"
    assert result.get("retried_after_change")
    w = World()
    result, _ = ask(w, related(w, "revision"), arm=CURRENT)
    assert result["status"] == "ABSTAIN" and result["reason"] == "source_changed"
