"""0.1.0 (parent docs/PUB_2B_REPORT.md): mock mode — the default of `rvg.Settings()` and of
`.env.example` — answers end to end without a network or a model: the mock writer drafts from
the first evidence unit that states a number (the rules of the HTTP-level fake), the mock judge
and embeddings are deterministic. The README's example gives FULL "500" to a principal who may
see the budget and ABSTAIN to one who may not, through the library and the command line. Live
settings never get a mock provider."""

import asyncio
import json

import httpx
import pytest
from test_cli_usability import cli

import rvg
from rvg.factory import decision_provider, embedding_provider, generation_provider
from rvg.providers.generation import MockGenerationProvider
from rvg.providers.mock import MockDecisionProvider, MockEmbeddingProvider

AS_OF = "2026-09-20T00:00:00Z"
QUESTION = {"text": "What is the budget of project A?", "as_of": AS_OF, "known_at": AS_OF}


def revision(identifier, sentences):
    text, units, offset = " ".join(sentences), [], 0
    for i, part in enumerate(sentences):
        units.append({"id": f"u{i}", "start": offset, "end": offset + len(part)})
        offset += len(part) + 1
    return {
        "id": identifier,
        "family": identifier,
        "revision": 1,
        "title": identifier.upper(),
        "text": text,
        "source": f"example://{identifier}",
        "published_at": "2026-09-01T00:00:00Z",
        "observed_at": "2026-09-01T00:00:00Z",
        "units": units,
    }


BUDGET = revision("budget", ["The budget of project A is 500.", "It ends soon."])
LEAD = revision("lead", ["Project B is led by Kim."])


@pytest.fixture
def no_network(monkeypatch):
    for name in ("DEEPJEV_MODE", "RVG_MODE"):
        monkeypatch.delenv(name, raising=False)

    async def refuse(self, request, **kwargs):
        raise AssertionError(f"mock mode sent a request to {request.url.host}")

    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)


def test_the_library_answers_the_readme_example_in_default_mock_mode(tmp_path, no_network):
    async def go():
        storage = {"schema_version": 1, "state": {"backend": "sqlite", "path": "state.sqlite3"}}
        settings = rvg.Settings(_env_file=None)
        assert settings.deepjev_mode == "mock"
        async with rvg.open_session(storage, settings, root=tmp_path) as session:
            await session.ingest(
                [{"document": BUDGET, "access": {"allow": ["staff"]}}, LEAD],
                rvg.RequestContext(tenant="example"),
            )
            out = {}
            for name, principal in (
                ("staff", rvg.TrustedPrincipal("alice", groups=("staff",))),
                ("other", rvg.TrustedPrincipal("bob")),
                ("anonymous", None),
            ):
                envelope = await session.answer(
                    QUESTION, rvg.RequestContext(tenant="example", principal=principal)
                )
                out[name] = envelope["result"]
            return out

    out = asyncio.run(go())
    assert (out["staff"]["status"], out["staff"]["short_answer"]) == ("FULL", "500")
    assert out["staff"]["sentences"][0]["evidence_ids"] == ["budget@1#u0"]
    for name in ("other", "anonymous"):
        assert out[name]["status"] == "ABSTAIN" and out[name]["short_answer"] is None, name


def test_the_command_line_answers_the_readme_example_in_mock_mode(tmp_path):
    lines = [{"document": BUDGET, "access": {"allow": ["staff"]}}, LEAD]
    (tmp_path / "kb.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    done = cli(tmp_path, "ingest", "kb.jsonl", "--store", "s.sqlite3", "--tenant", "example")
    assert done.returncode == 0, done.stderr

    def ask(*who):
        run = cli(
            tmp_path,
            "universal-query",
            QUESTION["text"],
            "--store",
            "s.sqlite3",
            "--tenant",
            "example",
            "--as-of",
            AS_OF,
            *who,
        )
        assert run.returncode == 0, run.stderr
        result = json.loads(run.stdout)
        return result["status"], result.get("short_answer")

    assert ask("--principal", "alice", "--group", "staff") == ("FULL", "500")
    assert ask("--principal", "bob") == ("ABSTAIN", None)


def test_live_settings_never_get_a_mock_provider():
    live = rvg.Settings(_env_file=None, rvg_mode="live", openrouter_api_key="not-a-real-key")
    mock = rvg.Settings(_env_file=None, rvg_mode="mock")
    assert not isinstance(decision_provider(live, None), MockDecisionProvider)
    assert not isinstance(generation_provider(live, None, role="surface"), MockGenerationProvider)
    assert not isinstance(embedding_provider(live, None), MockEmbeddingProvider)
    assert isinstance(decision_provider(mock, None), MockDecisionProvider)
    assert isinstance(generation_provider(mock, None, role="surface"), MockGenerationProvider)
