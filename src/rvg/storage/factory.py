"""A deployment's storage, assembled by dependency injection from `storage.json` (PORT-0).

The storage file names backends and resources only: which state store, which vector index,
where derived files go and which environment variable holds a connection string. It is separate
from the semantic policy (`rvg/resources/policies/universal.json`) and never selects anything by knowledge base,
domain or language. Credentials are never written in it — only the name of the environment
variable that holds them."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..contracts import Contract
from ..errors import ConfigurationError
from ..retrieval.data import canonical_hash
from .catalog import CatalogCache
from .local_exact import LocalExactIndex
from .release import ReleaseCoordinator


class StateConfig(Contract):
    backend: Literal["sqlite", "postgres"]
    path: str | None = None  # sqlite: file path (relative to the deployment root) or ":memory:"
    dsn_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")  # postgres
    schema_name: str | None = Field(default=None, pattern=r"^[a-z_][a-z0-9_]{0,62}$")


class QueryProjectionConfig(Contract):
    method: Literal["truncate_l2"]
    query_dimensions: int = Field(ge=2, le=65536)


class ExternalVectorsConfig(Contract):
    # N1F-5: the manifest of a read-only float16 vector set (relative to the deployment root)
    manifest: str
    score_dtype: Literal["float64", "float32"] = "float64"
    threads: int = Field(default=1, ge=1, le=64)
    # E9 (opt-in): queries of a space with more dimensions of the same model, cut and normalised
    query_projection: QueryProjectionConfig | None = None


class VectorConfig(Contract):
    index: Literal["local_exact", "pgvector", "qdrant"] = "local_exact"
    # local_exact: "beside" = `<sqlite path>.dense`; a path: that directory; null: resident.
    matrix_dir: str | None = "beside"
    # local_exact physical layout: "auto" (the default since 0.1.0.dev41, FIX-MATRIX-GEN:
    # "base_delta" with a matrix directory, "rebuild" for a resident matrix or the N1F-4
    # candidate stage), "rebuild" (E0H) or an N1F-2 incremental layout.
    layout: Literal["auto", "rebuild", "base_delta", "segments", "paged"] = "auto"
    recipe: dict | None = None
    # pgvector / qdrant: where the service is (environment variable names only) and what the
    # ranking guarantees: "native" (exact over the service's float32 vectors) or "canonical"
    # (candidates rescored from the canonical float64 vectors with a completeness proof).
    precision: Literal["native", "canonical"] = "canonical"
    dsn_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    url_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    schema_name: str | None = Field(default=None, pattern=r"^[a-z_][a-z0-9_]{0,62}$")
    prefix: str = Field(default="rvg", pattern=r"^[a-z][a-z0-9_]{0,30}$")
    overlay_limit: int = Field(default=20000, ge=0)
    # local_exact + "rebuild" only (N1F-4, opt-in): TurboQuant candidates rescored canonically.
    candidate_stage: Literal["none", "turboquant_mse4"] = "none"
    # N1F-5 (opt-in): "external_float16" ranks a read-only float16 vector set in place
    # (`vectors.external`), with the state store's vectors for the inputs the set lacks.
    source: Literal["blobs", "external_float16"] = "blobs"
    external: ExternalVectorsConfig | None = None


class StorageConfig(Contract):
    schema_version: Literal[1]
    state: StateConfig
    vectors: VectorConfig = VectorConfig()
    # N1F-6 (opt-in): "columnar" keeps the catalog in fixed-width columns and reads documents
    # from the state store when a query needs them (for knowledge bases of millions of rows)
    catalog: Literal["resident", "columnar"] = "resident"

    def fingerprint(self) -> str:
        body = self.model_dump(mode="json")
        if body["catalog"] == "resident":
            del body["catalog"]  # off: every existing hash stays as it was
        if body["vectors"]["candidate_stage"] == "none":
            del body["vectors"]["candidate_stage"]  # off: every existing hash stays as it was
        if body["vectors"]["source"] == "blobs":
            del body["vectors"]["source"]  # N1F-5 off: the same
        if body["vectors"]["external"] is None:
            del body["vectors"]["external"]
        # FIX-MATRIX-GEN: the layout the deployment really uses, so "auto" changes the hash only
        # where the physical layout changed (a local exact index with a matrix directory)
        body["vectors"]["layout"] = resolved_layout(self.vectors)
        return canonical_hash(body)


def resolved_layout(vectors) -> str:
    """The physical layout of a vector configuration: `layout` unless "auto" — then
    "base_delta" for a local exact index with a matrix directory and no candidate stage,
    "rebuild" otherwise (FIX-MATRIX-GEN)."""
    if vectors.layout != "auto":
        return vectors.layout
    if getattr(vectors, "source", "blobs") != "blobs":
        return "rebuild"  # N1F-5: the state store's vectors stay a resident overlay
    incremental = (
        vectors.index == "local_exact"
        and vectors.matrix_dir is not None
        and vectors.candidate_stage == "none"
    )
    return "base_delta" if incremental else "rebuild"


@dataclass
class Storage:
    state: object
    blobs: object
    index: object
    catalogs: CatalogCache
    release: ReleaseCoordinator
    config_hash: str | None = None
    projector: object | None = None
    _described: dict | None = field(default=None, repr=False)

    def describe(self):
        """The deployment as recorded with every result: backends, the index's precision
        class and the storage configuration's hash (no paths, no credentials)."""
        if self._described is None:
            state, index = self.state.describe(), self.index.capabilities()
            body = {
                "state": state["backend"],
                "index": index["index"],
                "precision": index["search"],
                "config_hash": self.config_hash,
            }
            self._described = body | {
                "fingerprint": canonical_hash({"state": state, "index": index, **body})
            }
        return dict(self._described)

    def close(self):
        """The state store, the vector index (a local index lets go of its generations' files
        and memory maps, 0.1.0) and a vector service's connection."""
        self.state.close()
        close = getattr(self.index, "close", None)
        if close is not None:
            close()
        backend = getattr(self.index, "backend", None)
        if backend is not None:
            backend.close()


def local_storage(
    state,
    *,
    matrix_dir=None,
    config_hash=None,
    index=None,
    catalogs=None,
    layout="auto",
    recipe=None,
    candidate_stage="none",
    catalog=None,
):
    """A state store that is also its vector blob store, searched by the local exact index
    (with the opt-in TurboQuant candidate stage in front when `candidate_stage` names it).
    `layout="auto"` (FIX-MATRIX-GEN, the default since 0.1.0.dev41): the incremental
    `base_delta` layout with a matrix directory — a vector write appends to a small delta
    instead of rewriting the whole matrix — and the full rebuild for a resident matrix or with
    the candidate stage (N1F-4, rebuild only). Search results are the same bit for bit."""
    if layout == "auto":
        layout = "base_delta" if matrix_dir is not None and candidate_stage == "none" else "rebuild"
    if index is None and layout != "rebuild":
        from .incremental import IncrementalExactIndex

        index = IncrementalExactIndex(state, matrix_dir=matrix_dir, layout=layout, recipe=recipe)
    if candidate_stage != "none":
        from .quantized import candidate_index

        exact = index if index is not None else LocalExactIndex(state, matrix_dir=matrix_dir)
        try:
            index = candidate_index(candidate_stage, exact)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from None
    return Storage(
        state=state,
        blobs=state,
        index=index if index is not None else LocalExactIndex(state, matrix_dir=matrix_dir),
        catalogs=catalogs if catalogs is not None else CatalogCache(state, layout=catalog),
        release=ReleaseCoordinator(state),
        config_hash=config_hash,
    )


def external_storage(
    state,
    manifest,
    *,
    score_dtype="float64",
    threads=1,
    projection=None,
    candidate_stage="none",
    config_hash=None,
    catalog=None,
):
    """A state store beside a read-only external float16 vector set (N1F-5, opt-in): the set is
    ranked in place, the state store's vectors (inputs the set lacks) in the same ranking; with
    the candidate stage, TurboQuant codes of the set's rows in front (built explicitly)."""
    from .external import ExternalCandidateIndex, ExternalFloat16Index, ExternalVectorSet

    try:
        index = ExternalFloat16Index(
            ExternalVectorSet(manifest),
            state,
            score_dtype=score_dtype,
            threads=threads,
            projection=projection,
        )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from None
    if candidate_stage == "turboquant_mse4":
        index = ExternalCandidateIndex(index)
    elif candidate_stage != "none":
        raise ConfigurationError("unknown_candidate_stage")
    return Storage(
        state=state,
        blobs=index.blobs,
        index=index,
        catalogs=CatalogCache(state, layout=catalog),
        release=ReleaseCoordinator(state),
        config_hash=config_hash,
    )


def remote_storage(
    state, backend, *, precision="canonical", overlay_limit=20000, config_hash=None, catalog=None
):
    """A state store that is its vector blob store, projected into a vector service."""
    from .projection import Projector
    from .remote import RemoteIndex

    catalogs = CatalogCache(state, layout=catalog)
    projector = Projector(state, state, backend, catalogs)
    index = RemoteIndex(
        state, state, backend, precision=precision, projector=projector, overlay_limit=overlay_limit
    )
    storage = Storage(
        state, state, index, catalogs, ReleaseCoordinator(state), config_hash=config_hash
    )
    storage.projector = projector
    return storage


def as_storage(value):
    if isinstance(value, Storage):
        return value
    storage = getattr(value, "storage", None)
    if isinstance(storage, Storage):
        return storage
    raise TypeError("storage_required")


def load_config(source) -> StorageConfig:
    if isinstance(source, StorageConfig):
        return source
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).read_text(encoding="utf-8"))
    return StorageConfig.model_validate(source)


def open_storage(source, *, root=None, environ=None):
    """Open the deployment a storage configuration names."""
    config = load_config(source)
    root = Path(root) if root is not None else Path.cwd()
    environ = os.environ if environ is None else environ
    matrix = config.vectors.matrix_dir
    if config.state.backend == "sqlite":
        from .sqlite import SQLiteState

        if not config.state.path:
            raise ConfigurationError("sqlite_path_required")
        path = config.state.path if config.state.path == ":memory:" else root / config.state.path
        state = SQLiteState(path)
        if matrix == "beside":
            matrix = None if path == ":memory:" else Path(f"{path}.dense")
    else:
        if not config.state.dsn_env or config.state.dsn_env not in environ:
            raise ConfigurationError("postgres_dsn_environment_variable_missing")
        if matrix == "beside" and config.vectors.index == "local_exact":
            raise ConfigurationError("matrix_dir_required_for_postgres")
        from .postgres import PostgresState

        state = PostgresState(environ[config.state.dsn_env], schema=config.state.schema_name)
    vectors = config.vectors
    if vectors.candidate_stage != "none" and vectors.index != "local_exact":
        raise ConfigurationError("candidate_stage_requires_local_exact")
    if vectors.source == "external_float16":
        if vectors.index != "local_exact":
            raise ConfigurationError("external_vectors_require_local_exact")
        if vectors.external is None:
            raise ConfigurationError("external_vectors_manifest_required")
        external = vectors.external
        manifest = Path(external.manifest)
        manifest = manifest if manifest.is_absolute() else root / manifest
        projection = external.query_projection
        return external_storage(
            state,
            manifest,
            score_dtype=external.score_dtype,
            threads=external.threads,
            projection=None if projection is None else projection.model_dump(),
            candidate_stage=vectors.candidate_stage,
            config_hash=config.fingerprint(),
            catalog=config.catalog,
        )
    if vectors.external is not None:
        raise ConfigurationError("external_vectors_need_source_external_float16")
    if vectors.index == "pgvector":
        from .pgvector import PgvectorBackend

        if not vectors.dsn_env or vectors.dsn_env not in environ:
            raise ConfigurationError("pgvector_dsn_environment_variable_missing")
        backend = PgvectorBackend(
            environ[vectors.dsn_env], schema=vectors.schema_name or "rvg_index"
        )
        return remote_storage(
            state,
            backend,
            precision=vectors.precision,
            overlay_limit=vectors.overlay_limit,
            config_hash=config.fingerprint(),
            catalog=config.catalog,
        )
    if vectors.index == "qdrant":
        from .qdrant import QdrantBackend

        if not vectors.url_env or vectors.url_env not in environ:
            raise ConfigurationError("qdrant_url_environment_variable_missing")
        key = environ.get(vectors.api_key_env) if vectors.api_key_env else None
        backend = QdrantBackend(environ[vectors.url_env], prefix=vectors.prefix, api_key=key)
        return remote_storage(
            state,
            backend,
            precision=vectors.precision,
            overlay_limit=vectors.overlay_limit,
            config_hash=config.fingerprint(),
            catalog=config.catalog,
        )
    if matrix is not None:
        matrix = Path(matrix) if Path(matrix).is_absolute() else root / matrix
    return local_storage(
        state,
        matrix_dir=matrix,
        config_hash=config.fingerprint(),
        layout=vectors.layout,
        recipe=vectors.recipe,
        candidate_stage=vectors.candidate_stage,
        catalog=config.catalog,
    )
