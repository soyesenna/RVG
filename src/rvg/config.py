from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from .errors import ConfigurationError
from .runtime import CallContext

# SEP-04: new environment names accepted for legacy keys the runtime reads (a whitelist; the field
# keeps its name). Inside one source, both names with different values is an error; the new name
# alone is the legacy name; sources keep their priority (arguments > environment > .env).
ENV_ALIASES = {"rvg_mode": "deepjev_mode"}


class _AliasedSource(PydanticBaseSettingsSource):
    """An environment or `.env` source that reads the whitelisted new names as their legacy key."""

    def __init__(self, inner, label):
        super().__init__(inner.settings_cls)
        self.inner, self.label = inner, label

    def _set_current_state(self, state):
        super()._set_current_state(state)
        self.inner._set_current_state(state)

    def _set_settings_sources_data(self, states):
        super()._set_settings_sources_data(states)
        self.inner._set_settings_sources_data(states)

    def get_field_value(self, field, field_name):
        return self.inner.get_field_value(field, field_name)

    def __call__(self):
        values = self.inner()
        raw = self.inner.env_vars  # names lower-cased: the settings are case-insensitive
        for new, legacy in ENV_ALIASES.items():
            if new not in raw:
                continue
            if legacy in raw and raw[legacy] != raw[new]:
                raise ConfigurationError(
                    f"CONFIG_ALIAS_CONFLICT: {new.upper()} and {legacy.upper()} differ in the "
                    f"{self.label}"
                )
            values[legacy] = raw[new]
        return values


class Settings(BaseSettings):
    """Runtime settings. `Settings(...)` reads its arguments, then the environment, then the
    defaults — never a `.env` file: a library caller builds it from its own configuration and
    passes it on. `load_settings(root)` adds `root/.env` for standalone use (`.env.example`).
    An argument must be a field name or a whitelisted alias (`rvg_mode`), in any case; anything
    else is a `ConfigurationError` (0.1.0). Unrelated names in the environment or `.env` are
    ignored."""

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
        hide_input_in_errors=True,
        env_file_encoding="utf-8",
        enable_decoding=False,
    )
    deepjev_mode: Literal["mock", "live"] = "mock"
    jev_provider: Literal["vercel_gateway", "typesafe_direct", "openrouter", "mock"] = "openrouter"
    jev_model: str = "typesafe-ai/jev"
    jev_fallback_enabled: bool = True
    openrouter_jev_model: str = "typesafe/jev-1.14"
    openrouter_decisions_url: str = "https://openrouter.ai/api/alpha/decisions"
    ai_gateway_api_key: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    vercel_ai_gateway_base_url: str = "https://ai-gateway.vercel.sh/v4/ai"
    typesafe_api_key: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    typesafe_base_url: str = "https://api.typesafe.ai/v1"
    embedding_provider: Literal["openrouter", "mock"] = "openrouter"
    openrouter_api_key: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "google/gemini-embedding-2"
    # 1,536 since 0.1.0 (parent PUB-2): the dimensions every measurement used (3,072 before)
    embedding_dimensions: int = Field(default=1536, ge=1, le=65536)
    embedding_encoding_format: Literal["float"] = "float"
    embedding_max_input_tokens: int = Field(default=8192, ge=1)
    embedding_batch_size: int = Field(default=2, ge=1, le=256)
    embedding_upstream_only: str = ""  # comma-separated OpenRouter upstream slugs; empty = any
    embedding_normalize: bool = True
    embedding_similarity_metric: Literal["cosine"] = "cosine"
    embedding_query_input_type: str | None = None
    embedding_document_input_type: str | None = None
    embedding_query_prefix: str = "task: search result | query: "
    embedding_document_prefix: str = "title: none | text: "
    embedding_space_version: int = Field(default=2, ge=1)
    # "claude_code": structured generation through the Claude Code CLI on the user's Claude
    # subscription (parent contract U21); the binary is a transport option (`claude_code_binary`).
    # The default since 0.1.0.dev16 (U21 ADOPT); "openrouter" with qwen/qwen3.8-27b was the
    # default through dev15.
    proposer_provider: Literal["openrouter", "claude_code", "mock"] = "claude_code"
    semantic_proposer_model: str = "claude-haiku-5-5"
    surface_proposer_model: str = "claude-haiku-5-5"
    proposer_temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    proposer_reasoning_enabled: bool | None = False
    proposer_response_mode: Literal["json_schema", "json_object"] = "json_schema"
    semantic_proposer_max_tokens: int = Field(default=2048, ge=32, le=16384)
    surface_proposer_max_tokens: int = Field(default=1024, ge=32, le=16384)
    run_max_generation_calls: int = Field(default=4, ge=0)
    run_max_semantic_proposer_calls: int = Field(default=2, ge=0)
    run_max_surface_proposer_calls: int = Field(default=2, ge=0)
    run_max_schema_repairs: int = Field(default=1, ge=0, le=1)
    service_tokens_json: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    service_requests_per_minute: int = Field(default=30, ge=1, le=1000)
    service_max_concurrency: int = Field(default=4, ge=1, le=32)
    http_timeout_seconds: float = Field(default=20, gt=0, allow_inf_nan=False)
    http_max_attempts: int = Field(default=3, ge=1, le=10)
    run_deadline_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    run_max_jev_calls: int = Field(default=12, ge=1)
    run_max_embedding_calls: int = Field(default=4, ge=1)
    run_max_http_attempts: int = Field(default=24, ge=1)
    run_max_cost_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    choice_policy: Literal["probability_margin", "provider_confidence"] = "probability_margin"
    choice_min_probability: float = Field(default=0.60, ge=0, le=1)
    choice_min_margin: float = Field(default=0.15, ge=0, le=1)
    choice_min_provider_confidence: float = Field(default=0.60, ge=0, le=1)
    relevance_threshold: float = Field(default=0.65, ge=0, le=1)
    support_threshold: float = Field(default=0.85, ge=0, le=1)
    retrieval_mode: Literal["lexical", "dense", "hybrid"] = "hybrid"
    document_search_strategy: Literal["lexical", "dense", "hybrid"] = "dense"
    retrieval_candidate_k: int = Field(default=16, ge=1, le=64)
    retrieval_expanded_k: int = Field(default=32, ge=1, le=64)
    retrieval_rrf_k: int = Field(default=60, ge=1)
    evidence_batch_size: int = Field(default=8, ge=1, le=32)
    max_semantic_actions: int = Field(default=4, ge=1, le=32)
    deepjev_task: str = "grounded_explain"
    deepjev_db_path: Path = Path("data/deepjev.sqlite3")
    deepjev_runs_dir: Path = Path("runs")
    deepjev_vector_dir: Path = Path("data/vectors/gemini-embedding-2-3072-v2")
    project_root: Path = Field(default_factory=Path.cwd, exclude=True)

    def __init__(self, **values):
        names = {name.lower() for name in type(self).model_fields}
        unknown = sorted(
            key
            for key in values
            if not key.startswith("_") and key.lower() not in names | set(ENV_ALIASES)
        )
        if unknown:
            raise ConfigurationError("CONFIG_UNKNOWN_SETTING: " + ", ".join(unknown))
        for new, legacy in ENV_ALIASES.items():
            given = [key for key in values if key.lower() == new]
            if not given:
                continue
            value = values.pop(given[0])
            same = [key for key in values if key.lower() == legacy]
            if same and values[same[0]] != value:
                raise ConfigurationError(
                    f"CONFIG_ALIAS_CONFLICT: {new} and {legacy} differ in the arguments"
                )
            values.setdefault(same[0] if same else legacy, value)
        super().__init__(**values)

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        # The default sources in their default order; only the whitelisted aliases are added.
        return (
            init_settings,
            _AliasedSource(env_settings, "environment"),
            _AliasedSource(dotenv_settings, "dotenv"),
            file_secret_settings,
        )

    @field_validator(
        "run_max_cost_usd",
        "embedding_query_input_type",
        "embedding_document_input_type",
        "proposer_reasoning_enabled",
        mode="before",
    )
    @classmethod
    def empty_optional(cls, value):
        return None if value == "" else value

    @field_validator(
        "vercel_ai_gateway_base_url",
        "openrouter_base_url",
        "typesafe_base_url",
        "openrouter_decisions_url",
    )
    @classmethod
    def secure_url(cls, value: str) -> str:
        url = urlsplit(value)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("base URL must be HTTPS without credentials, query or fragment")
        return value.rstrip("/")

    def context(self) -> CallContext:
        return CallContext.with_timeout(
            self.run_deadline_seconds,
            max_http_attempts=self.run_max_http_attempts,
            max_decision_calls=self.run_max_jev_calls,
            max_embedding_calls=self.run_max_embedding_calls,
            max_generation_calls=self.run_max_generation_calls,
            max_semantic_proposer_calls=self.run_max_semantic_proposer_calls,
            max_surface_proposer_calls=self.run_max_surface_proposer_calls,
            max_schema_repairs=self.run_max_schema_repairs,
            max_cost_usd=self.run_max_cost_usd,
        )

    def resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path


def load_settings(root: Path, *, env_file: Path | None = None, **overrides) -> Settings:
    """Standalone use (the `rvg` command line, scripts): `Settings` over arguments, the
    environment and `root/.env` (or the named `env_file`, which must exist), in that order.
    Relative paths resolve against `root`; nothing is read from the installed package."""
    root = root.resolve()
    path = env_file if env_file is not None else root / ".env"
    if not path.is_absolute():
        path = root / path
    if env_file is not None and not path.is_file():
        raise ConfigurationError(f"The settings file {path} (env_file, --env-file) does not exist.")
    try:
        return Settings(_env_file=path, project_root=root, **overrides)
    except ValidationError as exc:
        # Pydantic's .errors() contains input values: expose field names only.
        fields = sorted({str(error["loc"][0]) for error in exc.errors()})
        raise ConfigurationError(
            "Invalid settings (values not shown): " + ", ".join(fields)
        ) from None
