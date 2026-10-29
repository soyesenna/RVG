import re
from dataclasses import replace

from .config import Settings
from .contracts import DecisionProvider, EmbeddingProvider, EmbeddingSpace
from .errors import ConfigurationError
from .providers.decision import (
    OpenRouterDecisionProvider,
    TypeSafeDecisionProvider,
    VercelDecisionProvider,
)
from .providers.embedding import OpenRouterEmbeddingProvider
from .providers.fallback import RateLimitFallbackProvider
from .providers.generation import (
    ClaudeCodeGenerationProvider,
    GenerationProvider,
    MockGenerationProvider,
    OpenRouterGenerationProvider,
)
from .providers.mock import MockDecisionProvider, MockEmbeddingProvider
from .transport import JsonTransport

# An OpenRouter model id: `vendor/model` (e.g. `qwen/qwen3.8-27b`), no spaces.
OPENROUTER_MODEL_ID = re.compile(r"[^\s/]+/[^\s/]+")


def generation_provider(
    settings: Settings, transport: JsonTransport, *, role: str
) -> GenerationProvider:
    if role not in {"semantic", "surface"}:
        raise ConfigurationError(
            f"Unsupported generation role {role!r}: use 'semantic' or 'surface'."
        )
    if settings.deepjev_mode == "mock" or settings.proposer_provider == "mock":
        return MockGenerationProvider()
    model = (
        settings.semantic_proposer_model if role == "semantic" else settings.surface_proposer_model
    )
    if settings.proposer_provider == "claude_code":
        return ClaudeCodeGenerationProvider(model=model, transport=transport)
    if not OPENROUTER_MODEL_ID.fullmatch(model):
        # 0.1.0 (parent PUB-2): caught when the session or the CLI starts, not as an
        # upstream_http_error at every answer (the defaults are Claude Code names)
        name = "SEMANTIC_PROPOSER_MODEL" if role == "semantic" else "SURFACE_PROPOSER_MODEL"
        raise ConfigurationError(
            f"CONFIG_PROPOSER_MODEL: PROPOSER_PROVIDER=openrouter needs an OpenRouter model id "
            f"(vendor/model) in {name}, not {model!r}"
        )
    if not settings.openrouter_api_key.get_secret_value():
        raise ConfigurationError(
            "OPENROUTER_API_KEY is required for the writer and extractor on OpenRouter "
            "(PROPOSER_PROVIDER=openrouter)."
        )
    return OpenRouterGenerationProvider(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        transport=transport,
        default_response_mode=settings.proposer_response_mode,
        default_reasoning_enabled=settings.proposer_reasoning_enabled,
    )


def decision_provider(settings: Settings, transport: JsonTransport) -> DecisionProvider:
    if settings.deepjev_mode == "mock" or settings.jev_provider == "mock":
        return MockDecisionProvider()

    def openrouter():
        if not settings.openrouter_api_key.get_secret_value():
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required for the Jev judge on OpenRouter "
                "(JEV_PROVIDER=openrouter, or the fallback of JEV_PROVIDER=vercel_gateway)."
            )
        return OpenRouterDecisionProvider(
            model=settings.openrouter_jev_model,
            api_key=settings.openrouter_api_key,
            endpoint=settings.openrouter_decisions_url,
            transport=transport,
        )

    if settings.jev_provider == "openrouter":
        return openrouter()
    if settings.jev_provider == "vercel_gateway":
        key = settings.ai_gateway_api_key
        cls = VercelDecisionProvider
        base_url = settings.vercel_ai_gateway_base_url
        key_name = "AI_GATEWAY_API_KEY"
    elif settings.jev_provider == "typesafe_direct":
        key = settings.typesafe_api_key
        cls = TypeSafeDecisionProvider
        base_url = settings.typesafe_base_url
        key_name = "TYPESAFE_API_KEY"
    else:
        raise ConfigurationError(
            f"Unsupported JEV_PROVIDER {settings.jev_provider!r}: use openrouter, "
            "vercel_gateway, typesafe_direct or mock."
        )
    if not key.get_secret_value():
        raise ConfigurationError(
            f"{key_name} is required for the Jev judge with JEV_PROVIDER={settings.jev_provider}."
        )
    if settings.jev_provider == "vercel_gateway" and settings.jev_fallback_enabled:
        secondary = openrouter()  # Fail configuration before any primary paid call.
        primary = VercelDecisionProvider(
            model=settings.jev_model,
            api_key=key,
            base_url=base_url,
            transport=transport,
            fail_fast_on_429=True,
        )
        return RateLimitFallbackProvider(primary, secondary)
    return cls(model=settings.jev_model, api_key=key, base_url=base_url, transport=transport)


def embedding_provider(settings: Settings, transport: JsonTransport) -> EmbeddingProvider:
    space = EmbeddingSpace(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        base_url=settings.openrouter_base_url,
        dimensions=settings.embedding_dimensions,
        normalize=settings.embedding_normalize,
        similarity_metric=settings.embedding_similarity_metric,
        encoding_format=settings.embedding_encoding_format,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
        query_input_type=settings.embedding_query_input_type,
        document_input_type=settings.embedding_document_input_type,
        version=settings.embedding_space_version,
    )
    if settings.deepjev_mode == "mock" or settings.embedding_provider == "mock":
        return MockEmbeddingProvider(
            replace(
                space, provider="mock", model="mock-not-embedding-model", base_url="mock://local"
            )
        )
    if settings.embedding_provider != "openrouter":
        raise ConfigurationError(
            f"Unsupported EMBEDDING_PROVIDER {settings.embedding_provider!r}: use openrouter or "
            "mock."
        )
    if not settings.openrouter_api_key.get_secret_value():
        raise ConfigurationError(
            "OPENROUTER_API_KEY is required for embeddings on OpenRouter "
            "(EMBEDDING_PROVIDER=openrouter)."
        )
    if space.model == "google/gemini-embedding-2":
        if not 128 <= space.dimensions <= 3072 or settings.embedding_max_input_tokens > 8192:
            raise ConfigurationError(
                "Gemini Embedding 2 takes EMBEDDING_DIMENSIONS from 128 to 3072 and "
                "EMBEDDING_MAX_INPUT_TOKENS up to 8192."
            )
        if space.query_input_type is not None or space.document_input_type is not None:
            raise ConfigurationError(
                "Gemini Embedding 2 takes no input_type: leave EMBEDDING_QUERY_INPUT_TYPE and "
                "EMBEDDING_DOCUMENT_INPUT_TYPE empty and use the text prefixes "
                "(EMBEDDING_QUERY_PREFIX, EMBEDDING_DOCUMENT_PREFIX)."
            )
    return OpenRouterEmbeddingProvider(
        space=space,
        api_key=settings.openrouter_api_key,
        transport=transport,
        batch_size=settings.embedding_batch_size,
        upstream_only=tuple(
            s.strip() for s in settings.embedding_upstream_only.split(",") if s.strip()
        ),
        max_input_tokens=settings.embedding_max_input_tokens,
    )
