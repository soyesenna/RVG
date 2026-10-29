"""SEP-04 A3 (G04-SETTINGS): how `Settings` resolves its inputs. The first group records the
behaviour of the extraction baseline and must hold before and after the alias change; the second
covers the one whitelisted alias `RVG_MODE` <-> `DEEPJEV_MODE`; the third pins what a library
caller relies on: `Settings(...)` reads arguments and the environment but never a `.env` file, and
`.env.example` lists every setting at its default."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from rvg import config
from rvg.config import Settings, load_settings
from rvg.errors import ConfigurationError

DEFAULTS_SHA256 = "a86fa435d17eaa6ef90aa75bdd2fd4671ad81d705cb315946e84b7dfd9ee49bd"
# 0.1.0.dev11 (parent docs/rvi/RVI_CYCLE3_DECISIONS_4.md §1): the decision defaults moved to
# OpenRouter jev-1.14. Every other default is still the baseline's.
DECISION_DEFAULTS = {"jev_provider": "openrouter", "openrouter_jev_model": "typesafe/jev-1.14"}
BASELINE_DECISION_DEFAULTS = {
    "jev_provider": "vercel_gateway",
    "openrouter_jev_model": "typesafe/jev-1.13",
}
# 0.1.0.dev16 (parent docs/UNIVERSAL_U21_REPORT.md, U21 ADOPT): writer and extractor Claude Haiku
# through the Claude Code CLI.
GENERATION_DEFAULTS = {
    "proposer_provider": "claude_code",
    "semantic_proposer_model": "claude-haiku-5-5",
    "surface_proposer_model": "claude-haiku-5-5",
}
BASELINE_GENERATION_DEFAULTS = {
    "proposer_provider": "openrouter",
    "semantic_proposer_model": "qwen/qwen3.8-27b",
    "surface_proposer_model": "qwen/qwen3.8-27b",
}
# 0.1.0 (parent docs/PUB_2_REPORT.md): the embedding dimensions every measurement used.
EMBEDDING_DEFAULTS = {"embedding_dimensions": 1536}
BASELINE_EMBEDDING_DEFAULTS = {"embedding_dimensions": 3072}
DUMPED_FIELDS = 61  # model_dump(exclude={"project_root"}) keys of the baseline


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in list(os.environ):
        if key.lower() in Settings.model_fields or key.upper() in ("RVG_MODE",):
            monkeypatch.delenv(key)


def dotenv(tmp_path, text):
    (tmp_path / ".env").write_text(text, encoding="utf-8")
    return tmp_path


# --- baseline behaviour -------------------------------------------------------------------------


def test_defaults_are_the_baseline_defaults():
    values = Settings(_env_file=None).model_dump(mode="json", exclude={"project_root"})
    assert {key: values[key] for key in DECISION_DEFAULTS} == DECISION_DEFAULTS
    assert {key: values[key] for key in GENERATION_DEFAULTS} == GENERATION_DEFAULTS
    assert {key: values[key] for key in EMBEDDING_DEFAULTS} == EMBEDDING_DEFAULTS
    baseline = (
        values
        | BASELINE_DECISION_DEFAULTS
        | BASELINE_GENERATION_DEFAULTS
        | BASELINE_EMBEDDING_DEFAULTS
    )
    canonical = json.dumps(baseline, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == DEFAULTS_SHA256
    assert len(values) == DUMPED_FIELDS and values["deepjev_mode"] == "mock"


def test_sources_keep_their_priority_argument_environment_dotenv_default(tmp_path, monkeypatch):
    root = dotenv(tmp_path, "DEEPJEV_MODE=live\nHTTP_MAX_ATTEMPTS=5\n")
    assert load_settings(root).deepjev_mode == "live"  # .env over the default
    monkeypatch.setenv("DEEPJEV_MODE", "mock")
    settings = load_settings(root)
    assert settings.deepjev_mode == "mock"  # environment over .env
    assert settings.http_max_attempts == 5  # .env still fills what the environment leaves
    assert load_settings(root, deepjev_mode="live").deepjev_mode == "live"  # argument first


def test_empty_strings_of_optional_fields_are_none(tmp_path):
    root = dotenv(tmp_path, "RUN_MAX_COST_USD=\nPROPOSER_REASONING_ENABLED=\n")
    settings = load_settings(root)
    assert settings.run_max_cost_usd is None and settings.proposer_reasoning_enabled is None


def test_an_invalid_value_names_the_field_and_never_the_value(tmp_path):
    root = dotenv(tmp_path, "DEEPJEV_MODE=sk-or-v1-not-a-mode\n")
    with pytest.raises(ConfigurationError) as caught:
        load_settings(root)
    assert "deepjev_mode" in str(caught.value) and "sk-or" not in str(caught.value)


def test_a_named_env_file_that_does_not_exist_is_an_error(tmp_path):
    with pytest.raises(ConfigurationError):
        load_settings(tmp_path, env_file=tmp_path / "missing.env")


def cli(tmp_path, *args, env=None):
    base = {k: v for k, v in os.environ.items() if k.lower() not in Settings.model_fields}
    base.pop("RVG_MODE", None)
    return subprocess.run(
        [sys.executable, "-m", "rvg", "--root", str(tmp_path), *args],
        capture_output=True,
        text=True,
        env=base | (env or {}),
        cwd=tmp_path,
        check=False,
    )


def test_the_command_line_mode_is_above_the_environment(tmp_path):
    query = ("universal-query", "q", "--store", str(tmp_path / "s.sqlite3"))
    live = cli(tmp_path, "--mode", "live", *query, env={"DEEPJEV_MODE": "mock"})
    assert live.returncode == 2 and "OPENROUTER_API_KEY" in live.stderr  # live needs a key
    mock = cli(tmp_path, "--mode", "mock", *query, env={"DEEPJEV_MODE": "live"})
    assert "OPENROUTER_API_KEY" not in mock.stderr  # mock never asks for one


# --- the whitelisted alias ----------------------------------------------------------------------


def test_the_whitelist_is_one_runtime_key():
    assert config.ENV_ALIASES == {"rvg_mode": "deepjev_mode"}


def test_the_new_name_alone_equals_the_legacy_name_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPJEV_MODE", "live")
    legacy = load_settings(tmp_path)
    monkeypatch.delenv("DEEPJEV_MODE")
    monkeypatch.setenv("RVG_MODE", "live")
    new = load_settings(tmp_path)
    assert new.model_dump(exclude={"project_root"}) == legacy.model_dump(exclude={"project_root"})
    assert new.deepjev_mode == "live"
    root = dotenv(tmp_path, "RVG_MODE=live\n")
    monkeypatch.delenv("RVG_MODE")
    assert load_settings(root).deepjev_mode == "live"


def test_both_names_with_one_value_converge(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPJEV_MODE", "live")
    monkeypatch.setenv("RVG_MODE", "live")
    assert load_settings(tmp_path).deepjev_mode == "live"
    monkeypatch.delenv("DEEPJEV_MODE")
    monkeypatch.delenv("RVG_MODE")
    assert (
        load_settings(dotenv(tmp_path, "RVG_MODE=mock\nDEEPJEV_MODE=mock\n")).deepjev_mode == "mock"
    )


@pytest.mark.parametrize("where", ["environment", "dotenv"])
def test_both_names_with_different_values_in_one_source_fail(tmp_path, monkeypatch, where):
    if where == "environment":
        monkeypatch.setenv("DEEPJEV_MODE", "mock")
        monkeypatch.setenv("RVG_MODE", "live")
    else:
        dotenv(tmp_path, "DEEPJEV_MODE=mock\nRVG_MODE=live\n")
    with pytest.raises(ConfigurationError) as caught:
        load_settings(tmp_path)
    message = str(caught.value)
    assert message.startswith("CONFIG_ALIAS_CONFLICT") and where in message
    assert "live" not in message and "mock" not in message  # names only, never values


def test_across_sources_the_existing_priority_decides(tmp_path, monkeypatch):
    root = dotenv(tmp_path, "DEEPJEV_MODE=live\n")
    monkeypatch.setenv("RVG_MODE", "mock")
    assert load_settings(root).deepjev_mode == "mock"  # environment over .env
    root = dotenv(tmp_path, "RVG_MODE=mock\n")
    monkeypatch.delenv("RVG_MODE")
    monkeypatch.setenv("DEEPJEV_MODE", "live")
    assert load_settings(root).deepjev_mode == "live"
    assert load_settings(root, deepjev_mode="mock").deepjev_mode == "mock"  # argument first


def test_an_invalid_new_name_value_is_reported_on_the_field(tmp_path, monkeypatch):
    monkeypatch.setenv("RVG_MODE", "production")
    with pytest.raises(ConfigurationError) as caught:
        load_settings(tmp_path)
    assert "deepjev_mode" in str(caught.value) and "production" not in str(caught.value)


def test_research_legacy_fields_have_no_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("RVG_DB_PATH", "elsewhere.sqlite3")
    assert str(load_settings(tmp_path).deepjev_db_path) == "data/deepjev.sqlite3"


# --- explicit construction and the example file -------------------------------------------------

EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"
EXAMPLE_LINE = re.compile(r"^(?:# )?([A-Z][A-Z0-9_]*)=(.*)$")


def test_the_constructor_reads_arguments_and_environment_but_no_dotenv(tmp_path, monkeypatch):
    dotenv(tmp_path, "DEEPJEV_MODE=live\nHTTP_MAX_ATTEMPTS=5\n")
    monkeypatch.chdir(tmp_path)
    assert Settings().deepjev_mode == "mock" and Settings().http_max_attempts == 3
    monkeypatch.setenv("HTTP_MAX_ATTEMPTS", "4")
    assert Settings().http_max_attempts == 4  # the caller's environment
    assert Settings(http_max_attempts=2).http_max_attempts == 2  # the caller's argument first
    assert load_settings(tmp_path).deepjev_mode == "live"  # a file is read only when named


def test_an_argument_is_a_field_or_an_alias_and_nothing_else(monkeypatch):
    """0.1.0 (parent PUB-2): `Settings(rvg_mode="live")` was ignored and stayed mock."""
    monkeypatch.delenv("DEEPJEV_MODE", raising=False)
    monkeypatch.delenv("RVG_MODE", raising=False)
    for kwargs in ({"rvg_mode": "live"}, {"RVG_MODE": "live"}, {"deepjev_mode": "live"}):
        assert Settings(_env_file=None, **kwargs).deepjev_mode == "live", kwargs
    assert Settings(_env_file=None, rvg_mode="live", deepjev_mode="live").deepjev_mode == "live"
    with pytest.raises(ConfigurationError, match="CONFIG_UNKNOWN_SETTING: rvg_modee"):
        Settings(_env_file=None, rvg_modee="live")
    with pytest.raises(ConfigurationError, match="CONFIG_ALIAS_CONFLICT"):
        Settings(_env_file=None, rvg_mode="live", deepjev_mode="mock")
    monkeypatch.setenv("EXPERIMENT_MODE", "anything")  # unrelated environment names: ignored
    assert Settings(_env_file=None).deepjev_mode == "mock"


def example_entries() -> dict[str, str]:
    entries = {}
    for line in EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = EXAMPLE_LINE.match(line)
        if match:
            value = match[2]
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            entries[config.ENV_ALIASES.get(match[1].lower(), match[1].lower())] = value
    return entries


def test_the_example_file_lists_every_setting_and_copying_it_changes_nothing():
    entries = example_entries()
    assert set(entries) == set(Settings.model_fields) - {"project_root"}
    defaults = Settings(_env_file=None)
    for field, value in entries.items():
        parsed = getattr(Settings(_env_file=None, **{field: value}), field)
        assert parsed == getattr(defaults, field), field
