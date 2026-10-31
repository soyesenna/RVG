"""RVG: the Universal RVG runtime as an installable library.

One product path — time-scoped retrieval, deterministic validity rules, automatic scope, a
writer, the fixed answer controller and one bounded cascade step — under one policy for every
knowledge base and language. Importing this package reads no file, opens no connection and
creates no database: the public names (`rvg.api`) load on first use."""

__version__ = "0.1.1"

__all__ = [
    "API_VERSION",
    "OBSERVATION_VERSION",
    "AccessEvent",
    "AnswerDraft",
    "BudgetExceeded",
    "CacheMode",
    "CallContext",
    "ConfigurationError",
    "ContractError",
    "DecisionCache",
    "DocumentAccess",
    "IngestItem",
    "JsonTransport",
    "OperationError",
    "PromptResourceError",
    "ProviderError",
    "RequestContext",
    "RevisionEvent",
    "RevokeEvent",
    "RuntimeSession",
    "Settings",
    "Storage",
    "StorageConfig",
    "TemporalDocument",
    "TemporalQuestion",
    "TemporalRetriever",
    "TemporalStore",
    "TrustedPrincipal",
    "UniversalEngine",
    "__version__",
    "build_candidate_codes",
    "capabilities",
    "decision_provider",
    "embedding_provider",
    "export_external_set",
    "generation_provider",
    "load_config",
    "load_settings",
    "open_session",
    "open_storage",
    "policy",
    "policy_hash",
    "write_external_set",
]
_PUBLIC = frozenset(__all__) - {"__version__"}


def __getattr__(name):
    if name in _PUBLIC:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module 'rvg' has no attribute {name!r}")


def __dir__():
    return __all__
