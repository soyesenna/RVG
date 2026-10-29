"""0.1.0 (parent docs/PUB_2_REPORT.md, PUB_2C_REPORT.md): what a developer met first. The command line builds a
knowledge base (`rvg ingest`: documents, access lists, events from JSON Lines) and asks it as a
principal (`--principal`, `--group`), with help in English; `PROPOSER_PROVIDER=openrouter` with a
Claude Code model name fails when the session opens, not at every answer; the D-DIM-DECL note is
logged once per process at INFO; `.env.example` is English and names the Jev default."""

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import rvg
from rvg.config import Settings
from rvg.factory import generation_provider
from rvg.universal import config as universal_config

ROOT = Path(__file__).resolve().parents[1]
HANGUL = re.compile("[가-힣]")


def cli(tmp_path, *args):
    env = {k: v for k, v in os.environ.items() if k.lower() not in Settings.model_fields}
    env.pop("RVG_MODE", None)
    return subprocess.run(
        [sys.executable, "-m", "rvg", "--root", str(tmp_path), "--mode", "mock", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )


def document(identifier, sentences):
    text, units, offset = " ".join(sentences), [], 0
    for k, sentence in enumerate(sentences):
        units.append({"id": f"u{k}", "start": offset, "end": offset + len(sentence)})
        offset += len(sentence) + 1
    return {
        "id": identifier,
        "family": identifier,
        "revision": 1,
        "title": identifier.upper(),
        "text": text,
        "source": f"cli://{identifier}",
        "published_at": "2026-09-01T00:00:00Z",
        "observed_at": "2026-09-01T00:00:00Z",
        "units": units,
    }


def knowledge_base(tmp_path):
    lines = [
        {
            "document": document("a", ["The budget of project A is 500.", "It ends in October."]),
            "access": {"allow": ["alice", "staff"]},
        },
        document("b", ["Project B has no budget yet.", "It is led by Kim."]),
        {"type": "revoke", "document_id": "b"},
        {"id": "broken"},
    ]
    (tmp_path / "kb.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def seeds(tmp_path, *who):
    done = cli(
        tmp_path,
        "universal-query",
        "What is the budget of project A?",
        "--store",
        "s.sqlite3",
        "--as-of",
        "2026-09-20T00:00:00Z",
        "--include-audit",
        *who,
    )
    result = json.loads(done.stdout)
    return [s for audit in result["audit"] for s in audit["retrieval"]["seed_ids"]]


def test_ingest_builds_a_knowledge_base_that_a_principal_asks(tmp_path):
    knowledge_base(tmp_path)
    done = cli(tmp_path, "ingest", "kb.jsonl", "--store", "s.sqlite3")
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["documents"] == {"STORED": 2, "REJECTED": 1}
    assert report["events"] == {"APPLIED": 1}
    assert [(r["line"], r["reason"]) for r in report["rejected"]] == [(4, "family:missing")]
    assert report["vectors"]["status"] == "DONE" and report["vectors"]["pending"] == 0
    assert seeds(tmp_path, "--principal", "alice") == ["a@1"]
    assert seeds(tmp_path, "--principal", "carol", "--group", "staff") == ["a@1"]
    assert seeds(tmp_path, "--principal", "bob") == []  # a's access list, b revoked
    assert seeds(tmp_path) == []


def test_bad_command_line_input_is_named_without_its_content(tmp_path):
    (tmp_path / "kb.jsonl").write_text('{"id": "a"}\nsecret-not-json\n')
    done = cli(tmp_path, "ingest", "kb.jsonl", "--store", "s.sqlite3")
    assert done.returncode == 2 and "line 2: not JSON" in done.stderr
    assert "secret" not in done.stderr + done.stdout
    alone = cli(tmp_path, "universal-query", "q", "--store", "s.sqlite3", "--group", "staff")
    assert alone.returncode == 2 and "--group needs --principal" in alone.stderr


def test_the_help_is_english(tmp_path):
    for args in (("--help",), ("universal-query", "--help"), ("ingest", "--help")):
        done = cli(tmp_path, *args)
        assert done.returncode == 0 and not HANGUL.search(done.stdout), args


def test_openrouter_generation_needs_openrouter_model_ids_when_it_starts():
    settings = Settings(
        _env_file=None, rvg_mode="live", proposer_provider="openrouter", openrouter_api_key="k"
    )
    with pytest.raises(rvg.ConfigurationError, match="CONFIG_PROPOSER_MODEL"):
        generation_provider(settings, None, role="surface")
    with pytest.raises(rvg.ConfigurationError, match="CONFIG_PROPOSER_MODEL"):
        rvg.open_session(rvg.TemporalStore(":memory:"), settings)
    good = Settings(
        _env_file=None,
        rvg_mode="live",
        proposer_provider="openrouter",
        openrouter_api_key="k",
        surface_proposer_model="qwen/qwen3.8-27b",
    )
    assert type(generation_provider(good, None, role="surface")).__name__ == (
        "OpenRouterGenerationProvider"
    )


def test_the_dimension_note_is_logged_once_per_process_at_info(caplog, monkeypatch):
    class Space:
        def describe(self):
            return type("S", (), {"dimensions": 8})()

    monkeypatch.setattr(universal_config, "_NOTED", set())
    with caplog.at_level(logging.DEBUG, logger="rvg.policy"):
        for _ in range(3):
            assert universal_config.warn_dimensions(Space())
    notes = [r for r in caplog.records if r.name == "rvg.policy"]
    assert len(notes) == 1 and notes[0].levelno == logging.INFO


def test_the_example_settings_file_is_english_and_names_the_jev_default():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert not HANGUL.search(text)
    assert Settings.model_fields["jev_provider"].default == "openrouter"
    assert "# JEV_PROVIDER=openrouter" in text and "Vercel AI Gateway" not in text


# Hangul the package keeps on purpose: the mock judge's tokenizer of Korean text (a feature).
KOREAN_ALLOWED = {("rvg/providers/mock.py", "[a-z0-9]+|[가-힣]+")}


def test_the_package_says_nothing_to_a_developer_in_korean():
    """PUB-2c: configuration errors and the handoff packet's flag note were Korean. No string
    literal of the package is Korean now (comments and test data are not strings it shows)."""
    import ast

    package = ROOT / "src"
    found = []
    for path in sorted((package / "rvg").rglob("*.py")):
        rel = str(path.relative_to(package))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and HANGUL.search(node.value)
                and (rel, node.value) not in KOREAN_ALLOWED
            ):
                found.append((rel, node.lineno, node.value[:40]))
    assert found == []


def test_a_missing_key_names_the_setting_in_english():
    from rvg.universal.handoff import FLAG_NOTE

    live = Settings(_env_file=None, rvg_mode="live")
    with pytest.raises(rvg.ConfigurationError, match="OPENROUTER_API_KEY is required for the Jev"):
        rvg.decision_provider(live, None)
    with pytest.raises(rvg.ConfigurationError, match="OPENROUTER_API_KEY is required for embed"):
        rvg.embedding_provider(live, None)
    assert FLAG_NOTE == "automatic flag, low confidence"
