"""Logical identities every storage backend shares (K8-0, docs/K8_0_IDENTITY_MODEL.md).

A knowledge base holds documents; a document has immutable revisions; a revision is split into
retrieval segments by a versioned segmentation spec; a segment's vector is computed from an
embedding input built by a versioned input spec and is addressed by that input's hash. Today's
data is the special case "one segment = the whole revision" (`SEGMENTATION`), and the engine's
evidence keys stay `{document}@{revision}#{unit}`.

A backend's own identifiers (a SQLite rowid, a PostgreSQL sequence value, a Qdrant point id)
never leave its adapter. A backend that needs a fixed-format id derives it from these keys
(`VectorKey.point_id`) and keeps the reverse mapping itself."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..retrieval.data import canonical_hash

SEGMENTATION = "whole-revision/1"  # one retrieval segment = the whole revision
EMBEDDING_INPUT = "title-newline-text/1"  # the input a document vector is computed from
_POINTS = uuid.UUID("5d0c2f55-3a8e-4f0e-9d7b-8f1c6f0a4e21")


def embedding_input(document) -> str:
    """The text a document's vector is computed from (`EMBEDDING_INPUT`)."""
    return document.title + "\n" + document.text


def input_hash(text: str) -> str:
    return canonical_hash(text)


@dataclass(frozen=True, order=True)
class UnitKey:
    """One retrieval segment of one revision. Two revisions are two keys: they are never
    collapsed as duplicates, whatever their text."""

    tenant: str
    document_id: str
    revision: int
    segment_id: str = "s0"
    segmentation: str = SEGMENTATION

    @property
    def revision_key(self) -> str:
        return f"{self.document_id}@{self.revision}"

    def point_id(self, space: str) -> str:
        """The id of this segment's point in a vector index of `space`: one point per segment
        (two segments with the same input are two points), so every filter is per segment."""
        return str(
            uuid.uuid5(
                _POINTS,
                canonical_hash(
                    [
                        self.tenant,
                        self.document_id,
                        self.revision,
                        self.segment_id,
                        self.segmentation,
                        space,
                    ]
                ),
            )
        )


@dataclass(frozen=True)
class VectorKey:
    """A vector: the embedding of one input in one embedding space. Content addressed, so the
    same input in two revisions or documents is embedded once."""

    tenant: str
    space: str
    input_hash: str
    input_spec: str = EMBEDDING_INPUT

    def point_id(self) -> str:
        return str(
            uuid.uuid5(
                _POINTS, canonical_hash([self.tenant, self.space, self.input_spec, self.input_hash])
            )
        )
