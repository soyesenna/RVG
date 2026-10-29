"""Record/replay and mock tooling for development and evaluation runs; the product path never
imports it. `fake_openrouter` (an HTTP-level stand-in for the three OpenRouter endpoints) is a
submodule imported on its own."""

from ..providers.generation import MockGenerationProvider
from ..providers.mock import MockDecisionProvider, MockEmbeddingProvider
from ..universal.replay import RecordingTransport, ReplayMiss, ReplayTransport, Tape
from .records import answer_one, compare, manifest, runtime_question, signature

__all__ = [
    "MockDecisionProvider",
    "MockEmbeddingProvider",
    "MockGenerationProvider",
    "RecordingTransport",
    "ReplayMiss",
    "ReplayTransport",
    "Tape",
    "answer_one",
    "compare",
    "manifest",
    "runtime_question",
    "signature",
]
