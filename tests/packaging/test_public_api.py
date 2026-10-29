"""The public API re-exports the runtime objects themselves, reports the pinned policy and an
honest capability table, and answers end to end through the real provider code."""

import asyncio
import json
import subprocess
import sys

import pytest
from smoke import answer

import rvg
from rvg import api

POLICY_SHA256 = "24777fc8dfe3a16fb6225238147dc015b498e973464c25b40fa6e1f81753bae8"


def test_every_public_name_is_the_runtime_object_itself():
    from rvg.temporal.store import TemporalStore
    from rvg.universal.engine import UniversalEngine

    assert set(rvg.__all__) - {"__version__"} == set(api.__all__)
    for name in api.__all__:
        assert getattr(rvg, name) is getattr(api, name)
    assert rvg.UniversalEngine is UniversalEngine and rvg.TemporalStore is TemporalStore
    with pytest.raises(AttributeError):
        rvg.not_a_public_name  # noqa: B018


def test_the_policy_is_the_pinned_resource_and_a_copy_is_handed_out():
    assert rvg.policy_hash() == POLICY_SHA256
    policy = rvg.policy()
    policy["floor"]["threshold"] = 0.0
    assert rvg.policy()["floor"]["threshold"] == 0.85


def test_capabilities_name_what_is_missing_instead_of_substituting_it():
    caps = rvg.capabilities()
    json.dumps(caps)
    assert caps["policy"]["sha256"] == POLICY_SHA256
    assert caps["decision_providers"]["local_judge"]["status"] == "NOT_IMPLEMENTED"
    assert caps["vector_indexes"]["approximate"]["status"] == "NOT_IMPLEMENTED"
    assert caps["release_modes"]["scoped_dependency"]["status"] == "ADOPTED"  # U24 (dev27)
    assert caps["release_modes"]["current"]["status"] == "IMPLEMENTED"
    assert caps["statuses"] == ["FULL", "PARTIAL", "ABSTAIN", "UNRESOLVED_SCOPE", "ERROR"]


def test_the_engine_answers_through_the_public_api_and_stamps_the_policy():
    result = asyncio.run(answer())
    assert result["status"] == "FULL"
    assert result["config_hash"] == POLICY_SHA256
    assert result["short_answer"] == "500" and "a@1#u0" in result["evidence_ids"]


def test_the_command_line_reports_the_policy_and_the_capabilities(tmp_path):
    run = lambda *args: (
        subprocess.run(
            [sys.executable, "-m", "rvg", *args],
            capture_output=True,
            text=True,
            check=True,
            cwd=tmp_path,
        ).stdout
    )
    assert json.loads(run("policy"))["sha256"] == POLICY_SHA256
    caps = json.loads(run("capabilities"))
    assert caps["decision_providers"]["local_judge"]["status"] == "NOT_IMPLEMENTED"
    assert list(tmp_path.iterdir()) == []  # describing the build writes nothing
